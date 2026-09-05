"""Compare VQ-VAE and FSQ with their frozen priors on Imagenette-128.

Tokenizer evidence:

* MSE/PSNR and reconstruction feature-distribution distance;
* quantization error, active tokens, marginal entropy, and fixed capacity.

System evidence:

* held-out prior cross-entropy and effective bits per image;
* prior-sampled feature-distribution distance and sampling time;

The feature metric uses torchvision Inception-v3 with an optional fixed random
projection.  It is a consistent Fréchet proxy for this comparison, not the
canonical TensorFlow FID implementation.

Data:
    data/imagenette-128/val/<WNID>/*.JPEG, prepared by
    tool_scripts/download_dataset.py --dataset imagenette.

Checkpoints:
    output/vae/vq_vae/{tokenizer.pth,pixelcnn_prior.pth}: VQ-VAE system
    output/vae/fsq/{tokenizer.pth,pixelcnn_prior.pth}: FSQ system
    At least one tokenizer is required; a missing prior disables generation.

Outputs:
    output/vae/discrete_tokenizer_evaluation/metrics.json: system comparison
    output/vae/discrete_tokenizer_evaluation/<system>_real_and_reconstruction.png
    output/vae/discrete_tokenizer_evaluation/<system>_prior_samples.png

Evaluation data -- Imagenette validation:
Available images:                       3,925
Batch size:                                64
Reconstruction examples:                1,024
Generated examples:                       100
Generation batch size:                     10
Sampling temperature:                     1.0
Projected Inception feature dimensions:    256

Default dimensions:
Evaluation input:                     128x128 RGB
Generated image:                      128x128 RGB
Latent token grid:                      16x16 indices

Model size:
VQ-VAE tokenizer / prior:               1.71 M / 1.84 M parameters
VQ-VAE system total:                    3.55 M parameters
FSQ tokenizer / prior:                  1.60 M / 2.12 M parameters
FSQ system total:                       3.72 M parameters
Frozen Inception-v3 evaluator:         25.11 M parameters
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from dl_utils.data.imagenette import (
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    make_imagenette_loader,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import model_state_fingerprint
from dl_utils.vae.image_quality import (
    FeatureMoments,
    TorchvisionInceptionFeatures,
    collect_reference_feature_moments,
    evaluate_conditional_generation,
    frechet_distance,
)
from dl_utils.vae.quantization import (
    VQVAE,
    FSQAutoencoder,
    TokenUsageAccumulator,
)
from dl_utils.vae.token_prior import PixelCNNPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "imagenette-128"
IMAGE_SIZE = IMAGENETTE_IMAGE_SIZE
NUM_CLASSES = IMAGENETTE_NUM_CLASSES


@dataclass
class DiscreteSystem:
    name: str
    tokenizer: VQVAE | FSQAutoencoder
    prior: PixelCNNPrior | None
    image_size: int

    @property
    def has_complete_prior(self) -> bool:
        return self.prior is not None

    @property
    def latent_grid_size(self) -> int:
        return self.image_size // (2**self.tokenizer.downsample_steps)

    def reconstruct_and_tokens(
        self, x: Tensor
    ) -> tuple[Tensor, list[tuple[str, Tensor, int]], dict[str, Tensor]]:
        if isinstance(self.tokenizer, VQVAE):
            reconstruction, indices, _, diagnostics = self.tokenizer(x)
            return (
                reconstruction,
                [("tokens", indices, self.tokenizer.quantizer.codebook_size)],
                {"quantization_mse": diagnostics["quantization_mse"]},
            )
        reconstruction, indices, diagnostics = self.tokenizer(x)
        return (
            reconstruction,
            [("tokens", indices, self.tokenizer.quantizer.codebook_size)],
            {"quantization_mse": diagnostics["quantization_mse"]},
        )

    def prior_losses(
        self,
        token_levels: list[tuple[str, Tensor, int]],
        labels: Tensor,
    ) -> dict[str, Tensor]:
        if not self.has_complete_prior:
            return {}
        assert self.prior is not None
        indices = token_levels[0][1]
        class_labels = labels if self.prior.num_classes > 0 else None
        return {
            "tokens": F.cross_entropy(self.prior(indices, labels=class_labels), indices)
        }

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor,
        temperature: float,
    ) -> Tensor:
        if not self.has_complete_prior:
            raise RuntimeError("the system does not have its complete prior")
        assert self.prior is not None
        class_labels = labels if self.prior.num_classes > 0 else None
        indices = self.prior.sample(
            count,
            self.latent_grid_size,
            self.latent_grid_size,
            device=device,
            labels=class_labels,
            temperature=temperature,
        )
        return self.tokenizer.decode_indices(indices)


def make_test_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    return make_imagenette_loader(
        args.data_dir,
        args.batch_size,
        device,
        split="val",
        image_size=IMAGE_SIZE,
        horizontal_flip=False,
        num_workers=args.workers,
        shuffle=False,
        preprocessed=True,
        drop_last=False,
    )


def _load_prior(
    path: Path,
    device: torch.device,
    *,
    expected_model_name: str,
    tokenizer_interface_id: str,
) -> PixelCNNPrior:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != expected_model_name:
        raise ValueError(
            f"{path} has model_name={checkpoint.get('model_name')!r}; "
            f"expected {expected_model_name!r}"
        )
    if (
        checkpoint.get("dataset") != "imagenette-128"
        or checkpoint.get("image_size") != IMAGE_SIZE
        or checkpoint.get("conditioning") != "class_conditional"
    ):
        raise ValueError(f"{path} is not an Imagenette-128 conditional prior")
    prior = PixelCNNPrior(**checkpoint["model_config"]).to(device)
    prior.load_state_dict(checkpoint["state_dict"])
    if checkpoint.get("tokenizer_interface_id") != tokenizer_interface_id:
        raise ValueError(f"{path} was trained for a different tokenizer")
    return prior.eval().requires_grad_(False)


def load_single_level_system(
    *,
    name: str,
    tokenizer_path: Path,
    prior_path: Path,
    device: torch.device,
) -> DiscreteSystem | None:
    if not tokenizer_path.exists():
        return None
    checkpoint = torch.load(tokenizer_path, map_location=device, weights_only=True)
    model_name = checkpoint.get("model_name")
    if (
        checkpoint.get("dataset") != "imagenette-128"
        or checkpoint.get("image_size") != IMAGE_SIZE
    ):
        raise ValueError(f"{tokenizer_path} is not an Imagenette-128 tokenizer")
    if model_name == "vq_vae_tokenizer":
        tokenizer: VQVAE | FSQAutoencoder = VQVAE(**checkpoint["model_config"])
        expected_prior_name = "vq_vae_pixelcnn_prior"
    elif model_name == "fsq_tokenizer":
        tokenizer = FSQAutoencoder(**checkpoint["model_config"])
        expected_prior_name = "fsq_pixelcnn_prior"
    else:
        raise ValueError(f"{tokenizer_path} is not a supported tokenizer")
    tokenizer.load_state_dict(checkpoint["state_dict"])
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    if checkpoint.get("interface_id") != tokenizer_interface_id:
        raise ValueError(f"{tokenizer_path} has no valid tokenizer interface identity")
    prior = None
    if prior_path.exists():
        prior = _load_prior(
            prior_path,
            device,
            expected_model_name=expected_prior_name,
            tokenizer_interface_id=tokenizer_interface_id,
        )
    return DiscreteSystem(
        name,
        tokenizer.to(device).eval().requires_grad_(False),
        prior,
        IMAGE_SIZE,
    )


def load_systems(
    args: argparse.Namespace, device: torch.device
) -> list[DiscreteSystem]:
    systems = [
        load_single_level_system(
            name="vq_vae",
            tokenizer_path=args.vq_vae_tokenizer,
            prior_path=args.vq_vae_prior,
            device=device,
        ),
        load_single_level_system(
            name="fsq",
            tokenizer_path=args.fsq_tokenizer,
            prior_path=args.fsq_prior,
            device=device,
        ),
    ]
    return [system for system in systems if system is not None]


@torch.inference_mode()
def evaluate_tokenizer(
    system: DiscreteSystem,
    loader: DataLoader,
    feature_extractor: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    reconstruction_moments = FeatureMoments(feature_dim)
    usage: dict[str, TokenUsageAccumulator] = {}
    vocabulary_sizes: dict[str, int] = {}
    distortion_sum = 0.0
    element_count = 0
    quantization_totals: dict[str, float] = {}
    prior_totals: dict[str, float] = {}
    examples = 0
    comparison = None
    level_positions: dict[str, int] = {}
    for x, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        reconstruction, levels, quantization = system.reconstruct_and_tokens(x)
        if comparison is None:
            comparison = torch.cat((x[:16], reconstruction[:16])).cpu()
        reconstruction_moments.update(feature_extractor(reconstruction))
        distortion_sum += float((reconstruction - x).square().sum())
        element_count += x.numel()
        for name, indices, vocabulary_size in levels:
            if name not in usage:
                usage[name] = TokenUsageAccumulator(vocabulary_size)
                vocabulary_sizes[name] = vocabulary_size
                level_positions[name] = indices.shape[1] * indices.shape[2]
            usage[name].update(indices)
        for name, value in quantization.items():
            quantization_totals[name] = (
                quantization_totals.get(name, 0.0) + float(value) * x.shape[0]
            )
        for name, value in system.prior_losses(levels, labels).items():
            prior_totals[name] = prior_totals.get(name, 0.0) + float(value) * x.shape[0]
        examples += x.shape[0]

    mse = distortion_sum / element_count
    level_metrics: dict[str, object] = {}
    marginal_bits_per_image = 0.0
    fixed_bits_per_image = 0
    for name, accumulator in usage.items():
        statistics = accumulator.statistics()
        entropy_bits = float(statistics["token_entropy_nats"]) / math.log(2)
        positions = level_positions[name]
        marginal_bits_per_image += positions * entropy_bits
        fixed_bits_per_image += positions * math.ceil(math.log2(vocabulary_sizes[name]))
        level_metrics[name] = {
            "vocabulary_size": vocabulary_sizes[name],
            "positions": positions,
            "active_codes": int(statistics["active_codes"]),
            "usage_fraction": float(statistics["usage_fraction"]),
            "perplexity": float(statistics["perplexity"]),
            "marginal_entropy_bits_per_token": entropy_bits,
        }

    prior_metrics: dict[str, object] | None = None
    if prior_totals:
        assert system.prior is not None
        per_level = {
            name: value / examples / math.log(2) for name, value in prior_totals.items()
        }
        prior_bits_per_image = sum(
            level_positions[name] * bits for name, bits in per_level.items()
        )
        prior_metrics = {
            "bits_per_token_by_level": per_level,
            "bits_per_image": prior_bits_per_image,
            "parameter_count": sum(
                parameter.numel() for parameter in system.prior.parameters()
            ),
        }
    assert comparison is not None
    return {
        "examples": examples,
        "mse": mse,
        "psnr_for_minus_one_to_one_range": (10.0 * math.log10(4.0 / max(mse, 1e-12))),
        "projected_inception_reconstruction_frechet": frechet_distance(
            real_moments, reconstruction_moments
        ),
        "quantization_mse": {
            name: value / examples for name, value in quantization_totals.items()
        },
        "token_levels": level_metrics,
        "marginal_entropy_bits_per_image": marginal_bits_per_image,
        "fixed_length_bits_per_image": fixed_bits_per_image,
        "prior": prior_metrics,
    }, comparison


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_test_loader(args, device)
    systems = load_systems(args, device)
    if not systems:
        raise FileNotFoundError("no tokenizer checkpoint exists; run 6.0 or 6.1 first")
    projection_dim = (
        None if args.inception_projection_dim == 0 else args.inception_projection_dim
    )
    feature_extractor = TorchvisionInceptionFeatures(
        projection_dim=projection_dim,
        projection_seed=args.feature_seed,
    ).to(device)
    feature_dim = feature_extractor.feature_dim
    real_reconstruction_moments, real_generation_moments = (
        collect_reference_feature_moments(
            loader,
            feature_extractor,
            feature_dim=feature_dim,
            reconstruction_examples=args.max_examples,
            generation_examples=args.generation_examples,
            device=device,
        )
    )
    out_dir = OUTPUT_ROOT / "discrete_tokenizer_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_results: dict[str, object] = {}
    results: dict[str, object] = {
        "protocol": {
            "dataset": "imagenette-128 validation",
            "conditioning": "class_conditional",
            "max_reconstruction_examples": args.max_examples,
            "generation_examples": args.generation_examples,
            "feature_extractor": "torchvision Inception-v3 pool features",
            "fixed_random_projection_dimension": projection_dim,
            "fid_warning": (
                "These numbers use torchvision preprocessing and are not "
                "canonical TensorFlow FID values."
            ),
        },
        "models": model_results,
    }
    for system in systems:
        metrics, comparison = evaluate_tokenizer(
            system,
            loader,
            feature_extractor,
            real_reconstruction_moments,
            feature_dim=feature_dim,
            max_examples=args.max_examples,
            device=device,
        )
        save_image(
            comparison.mul(0.5).add(0.5),
            out_dir / f"{system.name}_real_and_reconstruction.png",
            nrow=16,
        )
        if system.has_complete_prior:
            generation, images = evaluate_conditional_generation(
                system,
                feature_extractor,
                real_generation_moments,
                examples=args.generation_examples,
                batch_size=args.generation_batch_size,
                num_classes=NUM_CLASSES,
                temperature=args.temperature,
                device=device,
            )
            metrics["generation"] = generation
            save_image(
                images.mul(0.5).add(0.5),
                out_dir / f"{system.name}_prior_samples.png",
                nrow=8,
            )
        else:
            metrics["generation"] = None
        model_results[system.name] = metrics
        print(
            f"{system.name}: MSE={metrics['mse']:.4f}, "
            f"rFID-proxy="
            f"{metrics['projected_inception_reconstruction_frechet']:.2f}, "
            f"entropy={metrics['marginal_entropy_bits_per_image']:.1f} "
            "bits/image"
        )
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved evaluation to {out_dir}")


class TinyFeatures(nn.Module):
    feature_dim = 3

    def forward(self, images: Tensor) -> Tensor:
        return images.mean(dim=(2, 3))


def smoke_test() -> None:
    set_seed(7)
    device = torch.device("cpu")
    loader = DataLoader(
        TensorDataset(
            torch.rand(12, 3, IMAGE_SIZE, IMAGE_SIZE).mul(2).sub(1),
            torch.arange(12) % NUM_CLASSES,
        ),
        batch_size=4,
    )
    tokenizer = VQVAE(
        hidden_channels=32,
        embedding_dim=8,
        codebook_size=16,
        downsample_steps=3,
    )
    prior = PixelCNNPrior(
        16,
        hidden_channels=16,
        layers=2,
        num_classes=NUM_CLASSES,
    )
    system = DiscreteSystem(
        "smoke",
        tokenizer.eval(),
        prior.eval(),
        IMAGE_SIZE,
    )
    features = TinyFeatures()
    real, generated_reference = collect_reference_feature_moments(
        loader,
        features,
        feature_dim=features.feature_dim,
        reconstruction_examples=8,
        generation_examples=4,
        device=device,
    )
    metrics, comparison = evaluate_tokenizer(
        system,
        loader,
        features,
        real,
        feature_dim=features.feature_dim,
        max_examples=8,
        device=device,
    )
    generation, images = evaluate_conditional_generation(
        system,
        features,
        generated_reference,
        examples=4,
        batch_size=2,
        num_classes=NUM_CLASSES,
        temperature=1.0,
        device=device,
    )
    assert comparison.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert metrics["fixed_length_bits_per_image"] == 1_024
    assert torch.isfinite(torch.tensor(generation["projected_inception_frechet"]))
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=1_024)
    parser.add_argument("--generation-examples", type=int, default=100)
    parser.add_argument("--generation-batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--inception-projection-dim", type=int, default=256)
    parser.add_argument("--feature-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--vq-vae-tokenizer",
        type=Path,
        default=OUTPUT_ROOT / "vq_vae" / "tokenizer.pth",
    )
    parser.add_argument(
        "--vq-vae-prior",
        type=Path,
        default=OUTPUT_ROOT / "vq_vae" / "pixelcnn_prior.pth",
    )
    parser.add_argument(
        "--fsq-tokenizer",
        type=Path,
        default=OUTPUT_ROOT / "fsq" / "tokenizer.pth",
    )
    parser.add_argument(
        "--fsq-prior",
        type=Path,
        default=OUTPUT_ROOT / "fsq" / "pixelcnn_prior.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
