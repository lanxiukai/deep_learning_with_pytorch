"""Compare discrete tokenizers and their frozen priors on held-out CIFAR-10.

Tokenizer evidence:

* MSE/PSNR and reconstruction feature-distribution distance;
* quantization error, active tokens, marginal entropy, and fixed capacity.

System evidence:

* held-out prior cross-entropy and effective bits per image;
* prior-sampled feature-distribution distance and sampling time;
* VQ-VAE-2 fixed-top/resampled-bottom versus regenerated-top interventions.

The feature metric uses torchvision Inception-v3 with an optional fixed random
projection.  It is a consistent Fréchet proxy for this comparison, not the
canonical TensorFlow FID implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from dl_utils.data.cifar10 import (
    make_cifar10_loader,
    normalized_cifar10_transform,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import model_state_fingerprint
from dl_utils.vae.image_quality import (
    FeatureMoments,
    TorchvisionInceptionFeatures,
    frechet_distance,
)
from dl_utils.vae.quantization import (
    VQVAE2,
    VQVAE32,
    FSQAutoencoder32,
    TokenUsageAccumulator,
)
from dl_utils.vae.token_prior import PixelCNNPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"


@dataclass
class DiscreteSystem:
    name: str
    tokenizer: VQVAE32 | VQVAE2 | FSQAutoencoder32
    priors: tuple[PixelCNNPrior, ...]

    @property
    def has_complete_prior(self) -> bool:
        required = 2 if isinstance(self.tokenizer, VQVAE2) else 1
        return len(self.priors) == required

    def reconstruct_and_tokens(
        self, x: Tensor
    ) -> tuple[Tensor, list[tuple[str, Tensor, int]], dict[str, Tensor]]:
        if isinstance(self.tokenizer, VQVAE32):
            reconstruction, indices, _, diagnostics = self.tokenizer(x)
            return reconstruction, [
                ("tokens", indices, self.tokenizer.quantizer.codebook_size)
            ], {
                "quantization_mse": diagnostics["quantization_mse"]
            }
        if isinstance(self.tokenizer, FSQAutoencoder32):
            reconstruction, indices, diagnostics = self.tokenizer(x)
            return reconstruction, [
                ("tokens", indices, self.tokenizer.quantizer.codebook_size)
            ], {
                "quantization_mse": diagnostics["quantization_mse"]
            }
        reconstruction, top, bottom, _, diagnostics = self.tokenizer(x)
        return reconstruction, [
            ("top", top, self.tokenizer.top_quantizer.codebook_size),
            ("bottom", bottom, self.tokenizer.bottom_quantizer.codebook_size),
        ], {
            "top_quantization_mse": diagnostics["top_quantization_mse"],
            "bottom_quantization_mse": diagnostics[
                "bottom_quantization_mse"
            ],
        }

    def prior_losses(
        self, token_levels: list[tuple[str, Tensor, int]]
    ) -> dict[str, Tensor]:
        if not self.has_complete_prior:
            return {}
        if isinstance(self.tokenizer, VQVAE2):
            top = token_levels[0][1]
            bottom = token_levels[1][1]
            top_prior, bottom_prior = self.priors
            condition = self.tokenizer.top_condition_from_indices(top)
            return {
                "top": F.cross_entropy(top_prior(top), top),
                "bottom": F.cross_entropy(
                    bottom_prior(bottom, condition), bottom
                ),
            }
        indices = token_levels[0][1]
        return {
            "tokens": F.cross_entropy(self.priors[0](indices), indices)
        }

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        temperature: float,
    ) -> Tensor:
        if not self.has_complete_prior:
            raise RuntimeError("the system does not have its complete prior")
        if isinstance(self.tokenizer, VQVAE2):
            top_prior, bottom_prior = self.priors
            top = top_prior.sample(
                count,
                4,
                4,
                device=device,
                temperature=temperature,
            )
            condition = self.tokenizer.top_condition_from_indices(top)
            bottom = bottom_prior.sample(
                count,
                8,
                8,
                device=device,
                condition=condition,
                temperature=temperature,
            )
            return self.tokenizer.decode_indices(top, bottom)
        indices = self.priors[0].sample(
            count,
            8,
            8,
            device=device,
            temperature=temperature,
        )
        return self.tokenizer.decode_indices(indices)


def make_test_loader(
    args: argparse.Namespace, device: torch.device
) -> DataLoader:
    return make_cifar10_loader(
        PROJECT_ROOT / "data" / "cifar10",
        args.batch_size,
        device,
        train=False,
        transform=normalized_cifar10_transform(horizontal_flip=False),
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
    )


def _load_prior(
    path: Path,
    device: torch.device,
    *,
    expected_model_name: str,
    tokenizer_interface_id: str,
) -> PixelCNNPrior:
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )
    if checkpoint.get("model_name") != expected_model_name:
        raise ValueError(
            f"{path} has model_name={checkpoint.get('model_name')!r}; "
            f"expected {expected_model_name!r}"
        )
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
    checkpoint = torch.load(
        tokenizer_path, map_location=device, weights_only=True
    )
    model_name = checkpoint.get("model_name")
    if model_name == "vq_vae_tokenizer":
        tokenizer: VQVAE32 | FSQAutoencoder32 = VQVAE32(
            **checkpoint["model_config"]
        )
        expected_prior_name = "vq_vae_pixelcnn_prior"
    elif model_name == "fsq_tokenizer":
        tokenizer = FSQAutoencoder32(**checkpoint["model_config"])
        expected_prior_name = "fsq_pixelcnn_prior"
    else:
        raise ValueError(f"{tokenizer_path} is not a supported tokenizer")
    tokenizer.load_state_dict(checkpoint["state_dict"])
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    if checkpoint.get("interface_id") != tokenizer_interface_id:
        raise ValueError(
            f"{tokenizer_path} has no valid tokenizer interface identity"
        )
    priors: tuple[PixelCNNPrior, ...] = ()
    if prior_path.exists():
        prior = _load_prior(
            prior_path,
            device,
            expected_model_name=expected_prior_name,
            tokenizer_interface_id=tokenizer_interface_id,
        )
        priors = (prior,)
    return DiscreteSystem(
        name,
        tokenizer.to(device).eval().requires_grad_(False),
        priors,
    )


def load_vq_vae_2_system(
    args: argparse.Namespace, device: torch.device
) -> DiscreteSystem | None:
    if not args.vq_vae_2_tokenizer.exists():
        return None
    checkpoint = torch.load(
        args.vq_vae_2_tokenizer,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("model_name") != "vq_vae_2_tokenizer":
        raise ValueError("VQ-VAE-2 tokenizer checkpoint has the wrong type")
    tokenizer = VQVAE2(**checkpoint["model_config"]).to(device)
    tokenizer.load_state_dict(checkpoint["state_dict"])
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    if checkpoint.get("interface_id") != tokenizer_interface_id:
        raise ValueError(
            "VQ-VAE-2 tokenizer has no valid interface identity"
        )
    priors = []
    if args.vq_vae_2_top_prior.exists() and args.vq_vae_2_bottom_prior.exists():
        for path, expected_model_name in (
            (
                args.vq_vae_2_top_prior,
                "vq_vae_2_top_prior",
            ),
            (
                args.vq_vae_2_bottom_prior,
                "vq_vae_2_bottom_conditional_prior",
            ),
        ):
            prior = _load_prior(
                path,
                device,
                expected_model_name=expected_model_name,
                tokenizer_interface_id=tokenizer_interface_id,
            )
            priors.append(prior)
    return DiscreteSystem(
        "vq_vae_2",
        tokenizer.eval().requires_grad_(False),
        tuple(priors),
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
        load_vq_vae_2_system(args, device),
        load_single_level_system(
            name="fsq",
            tokenizer_path=args.fsq_tokenizer,
            prior_path=args.fsq_prior,
            device=device,
        ),
    ]
    return [system for system in systems if system is not None]


@torch.inference_mode()
def real_feature_moments(
    loader: DataLoader,
    feature_extractor: nn.Module,
    *,
    feature_dim: int,
    max_examples: int,
    generation_examples: int,
    device: torch.device,
) -> tuple[FeatureMoments, FeatureMoments]:
    full = FeatureMoments(feature_dim)
    generation_reference = FeatureMoments(feature_dim)
    examples = 0
    generation_count = 0
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        features = feature_extractor(x)
        full.update(features)
        if generation_count < generation_examples:
            count = min(
                x.shape[0], generation_examples - generation_count
            )
            generation_reference.update(features[:count])
            generation_count += count
        examples += x.shape[0]
    return full, generation_reference


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
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        reconstruction, levels, quantization = (
            system.reconstruct_and_tokens(x)
        )
        if comparison is None:
            comparison = torch.cat((x[:16], reconstruction[:16])).cpu()
        reconstruction_moments.update(
            feature_extractor(reconstruction)
        )
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
                quantization_totals.get(name, 0.0)
                + float(value) * x.shape[0]
            )
        for name, value in system.prior_losses(levels).items():
            prior_totals[name] = (
                prior_totals.get(name, 0.0)
                + float(value) * x.shape[0]
            )
        examples += x.shape[0]

    mse = distortion_sum / element_count
    level_metrics: dict[str, object] = {}
    marginal_bits_per_image = 0.0
    fixed_bits_per_image = 0
    for name, accumulator in usage.items():
        statistics = accumulator.statistics()
        entropy_bits = (
            float(statistics["token_entropy_nats"]) / math.log(2)
        )
        positions = level_positions[name]
        marginal_bits_per_image += positions * entropy_bits
        fixed_bits_per_image += positions * math.ceil(
            math.log2(vocabulary_sizes[name])
        )
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
        per_level = {
            name: value / examples / math.log(2)
            for name, value in prior_totals.items()
        }
        prior_bits_per_image = sum(
            level_positions[name] * bits
            for name, bits in per_level.items()
        )
        prior_metrics = {
            "bits_per_token_by_level": per_level,
            "bits_per_image": prior_bits_per_image,
            "parameter_count": sum(
                parameter.numel()
                for prior in system.priors
                for parameter in prior.parameters()
            ),
        }
    assert comparison is not None
    return {
        "examples": examples,
        "mse": mse,
        "psnr_for_minus_one_to_one_range": (
            10.0 * math.log10(4.0 / max(mse, 1e-12))
        ),
        "projected_inception_reconstruction_frechet": frechet_distance(
            real_moments, reconstruction_moments
        ),
        "quantization_mse": {
            name: value / examples
            for name, value in quantization_totals.items()
        },
        "token_levels": level_metrics,
        "marginal_entropy_bits_per_image": marginal_bits_per_image,
        "fixed_length_bits_per_image": fixed_bits_per_image,
        "prior": prior_metrics,
    }, comparison


@torch.inference_mode()
def evaluate_generation(
    system: DiscreteSystem,
    feature_extractor: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    examples: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> tuple[dict[str, float], Tensor]:
    generated_moments = FeatureMoments(feature_dim)
    batches = []
    elapsed = 0.0
    generated = 0
    while generated < examples:
        count = min(batch_size, examples - generated)
        start = time.perf_counter()
        images = system.sample(
            count, device=device, temperature=temperature
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        generated_moments.update(feature_extractor(images))
        if sum(len(batch) for batch in batches) < 64:
            batches.append(images[: min(64 - generated, count)].cpu())
        generated += count
    return {
        "projected_inception_generation_frechet": frechet_distance(
            real_moments, generated_moments
        ),
        "seconds": elapsed,
        "images_per_second": examples / elapsed,
        "temperature": temperature,
    }, torch.cat(batches)[:64]


@torch.inference_mode()
def save_vq_vae_2_interventions(
    system: DiscreteSystem,
    loader: DataLoader,
    out_dir: Path,
    *,
    variants: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    if not isinstance(system.tokenizer, VQVAE2):
        return {}
    if not system.has_complete_prior:
        return {}
    top_prior, bottom_prior = system.priors
    x, _ = next(iter(loader))
    x = x[:8].to(device)
    _, _, real_top, _, _, _ = system.tokenizer.encode(x)
    repeated_top = real_top[:, None, ...].expand(
        -1, variants, -1, -1
    ).reshape(-1, 4, 4)
    condition = system.tokenizer.top_condition_from_indices(repeated_top)
    bottom = bottom_prior.sample(
        len(repeated_top),
        8,
        8,
        device=device,
        condition=condition,
        temperature=temperature,
    )
    fixed_top_images = system.tokenizer.decode_indices(
        repeated_top, bottom
    ).reshape(8, variants, 3, 32, 32)
    save_image(
        fixed_top_images.flatten(0, 1).mul(0.5).add(0.5),
        out_dir / "vq_vae_2_fixed_top_resampled_bottom.png",
        nrow=variants,
    )

    changed_top = top_prior.sample(
        8 * variants,
        4,
        4,
        device=device,
        temperature=temperature,
    )
    changed_condition = system.tokenizer.top_condition_from_indices(
        changed_top
    )
    changed_bottom = bottom_prior.sample(
        len(changed_top),
        8,
        8,
        device=device,
        condition=changed_condition,
        temperature=temperature,
    )
    changed_top_images = system.tokenizer.decode_indices(
        changed_top, changed_bottom
    ).reshape(8, variants, 3, 32, 32)
    save_image(
        changed_top_images.flatten(0, 1).mul(0.5).add(0.5),
        out_dir / "vq_vae_2_regenerated_top_and_bottom.png",
        nrow=variants,
    )
    return {
        "fixed_top_pixel_standard_deviation": float(
            fixed_top_images.std(dim=1).mean()
        ),
        "changed_top_pixel_standard_deviation": float(
            changed_top_images.std(dim=1).mean()
        ),
    }


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_test_loader(args, device)
    systems = load_systems(args, device)
    if not systems:
        raise FileNotFoundError(
            "no tokenizer checkpoint exists; run 6.0, 6.1, or 6.2 first"
        )
    projection_dim = (
        None if args.inception_projection_dim == 0
        else args.inception_projection_dim
    )
    feature_extractor = TorchvisionInceptionFeatures(
        projection_dim=projection_dim,
        projection_seed=args.feature_seed,
    ).to(device)
    feature_dim = feature_extractor.feature_dim
    real_reconstruction_moments, real_generation_moments = (
        real_feature_moments(
            loader,
            feature_extractor,
            feature_dim=feature_dim,
            max_examples=args.max_examples,
            generation_examples=args.generation_examples,
            device=device,
        )
    )
    out_dir = OUTPUT_ROOT / "discrete_tokenizer_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_results: dict[str, object] = {}
    results: dict[str, object] = {
        "protocol": {
            "dataset": "CIFAR-10 test",
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
            generation, images = evaluate_generation(
                system,
                feature_extractor,
                real_generation_moments,
                feature_dim=feature_dim,
                examples=args.generation_examples,
                batch_size=args.generation_batch_size,
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
        metrics["hierarchy_intervention"] = (
            save_vq_vae_2_interventions(
                system,
                loader,
                out_dir,
                variants=args.hierarchy_variants,
                temperature=args.temperature,
                device=device,
            )
        )
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
            torch.rand(12, 3, 32, 32).mul(2).sub(1),
            torch.arange(12) % 10,
        ),
        batch_size=4,
    )
    tokenizer = VQVAE32(
        hidden_channels=32, embedding_dim=8, codebook_size=16
    )
    prior = PixelCNNPrior(16, hidden_channels=16, layers=2)
    system = DiscreteSystem(
        "smoke",
        tokenizer.eval(),
        (prior.eval(),),
    )
    features = TinyFeatures()
    real, generated_reference = real_feature_moments(
        loader,
        features,
        feature_dim=features.feature_dim,
        max_examples=8,
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
    generation, images = evaluate_generation(
        system,
        features,
        generated_reference,
        feature_dim=features.feature_dim,
        examples=4,
        batch_size=2,
        temperature=1.0,
        device=device,
    )
    assert comparison.shape == (8, 3, 32, 32)
    assert images.shape == (4, 3, 32, 32)
    assert metrics["fixed_length_bits_per_image"] == 256
    assert torch.isfinite(
        torch.tensor(generation["projected_inception_generation_frechet"])
    )
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=2_048)
    parser.add_argument("--generation-examples", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hierarchy-variants", type=int, default=6)
    parser.add_argument(
        "--inception-projection-dim", type=int, default=256
    )
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
        "--vq-vae-2-tokenizer",
        type=Path,
        default=OUTPUT_ROOT / "vq_vae_2" / "tokenizer.pth",
    )
    parser.add_argument(
        "--vq-vae-2-top-prior",
        type=Path,
        default=OUTPUT_ROOT / "vq_vae_2" / "top_prior.pth",
    )
    parser.add_argument(
        "--vq-vae-2-bottom-prior",
        type=Path,
        default=OUTPUT_ROOT / "vq_vae_2" / "bottom_prior.pth",
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
