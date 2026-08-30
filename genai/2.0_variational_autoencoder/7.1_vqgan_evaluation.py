"""Evaluate trained VQGAN systems under one held-out CIFAR-10 protocol.

This script reloads the tokenizer, PatchGAN discriminator, and causal
Transformer prior from their stage artifacts. It reports:

* paired pixel, SSIM, and true LPIPS v0.1 fidelity;
* reconstruction feature-distribution distance and PatchGAN logits;
* quantization error, token utilization, entropy, and fixed capacity;
* held-out prior likelihood and complete prior-to-image generation.

Pass ``--vqgan-tokenizer`` more than once to compare fixed, adaptive, and
loss-ablation runs. Without explicit paths, the script discovers both the
legacy flat VQGAN artifact and named run directories. A trained VQ-VAE system
is included as a direct system baseline when its checkpoint exists; because
the tokenizer architectures differ, that cross-script comparison is not a
causal loss ablation.

The projected torchvision Inception distance is a consistent Fréchet proxy
for this lesson, not canonical TensorFlow FID. Unified cross-family generation
evaluation remains outside this model-specific entry.

Data:
    data/cifar10 test split, prepared by tool_scripts/download_dataset.py.
    CIFAR-10 class labels are ignored.

Checkpoints:
    output/vae/vqgan/**/tokenizer.pth: discovered VQGAN runs
    output/vae/vqgan/**/transformer_prior.pth: matching optional priors
    output/vae/vq_vae/{tokenizer.pth,pixelcnn_prior.pth}: optional baseline
    At least one VQGAN tokenizer is required.

Outputs:
    output/vae/vqgan_evaluation/metrics.json: detailed and summary metrics
    output/vae/vqgan_evaluation/<system>_real_and_reconstruction.png
    output/vae/vqgan_evaluation/<system>_prior_samples.png

Evaluation data -- CIFAR-10 test:
Available images:                      10,000
Batch size:                                64
Reconstruction examples:                2,048
Generated examples:                       256
Generation batch size:                     16
Sampling temperature:                     1.0
Projected Inception feature dimensions:    256

Default dimensions:
Evaluation input:                       32x32 RGB
Generated image:                        32x32 RGB
Latent token grid:                       8x8 indices

Model size per default VQGAN system:
VQGAN tokenizer:                         1.63 M parameters
Patch discriminator:                     1.25 M parameters
Transformer prior:                       3.44 M parameters
Stored VQGAN total:                      6.32 M parameters
Optional VQ-VAE baseline total:          3.02 M parameters
Frozen Inception-v3 evaluator:          25.11 M parameters
Frozen LPIPS-VGG evaluator:             14.72 M parameters
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

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
from dl_utils.training.checkpoints import (
    atomic_torch_save,
    model_state_fingerprint,
)
from dl_utils.vae.image_quality import (
    FeatureMoments,
    TorchvisionInceptionFeatures,
    frechet_distance,
    structural_similarity_index,
)
from dl_utils.vae.perceptual_autoencoder import (
    PatchDiscriminator32,
    RandomFeaturePerceptualLoss,
    VQPerceptualAutoencoder32,
    build_perceptual_loss,
)
from dl_utils.vae.quantization import VQVAE32, TokenUsageAccumulator
from dl_utils.vae.token_prior import CausalTransformerPrior, PixelCNNPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"

Tokenizer = VQPerceptualAutoencoder32 | VQVAE32
TokenPrior = CausalTransformerPrior | PixelCNNPrior


@dataclass
class EvaluatedSystem:
    name: str
    family: str
    tokenizer: Tokenizer
    prior: TokenPrior | None
    discriminator: PatchDiscriminator32 | None
    tokenizer_checkpoint: Path
    prior_checkpoint: Path | None
    training_config: dict[str, object]

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.quantizer.codebook_size

    @property
    def has_complete_prior(self) -> bool:
        return self.prior is not None

    def reconstruct_and_tokens(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        reconstruction, indices, _, diagnostics = self.tokenizer(x)
        return reconstruction, indices, diagnostics

    def prior_loss(self, indices: Tensor) -> Tensor | None:
        if isinstance(self.prior, CausalTransformerPrior):
            logits, targets = self.prior.teacher_forcing(indices)
            return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if isinstance(self.prior, PixelCNNPrior):
            return F.cross_entropy(self.prior(indices), indices)
        return None

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        temperature: float,
    ) -> Tensor:
        if self.prior is None:
            raise RuntimeError("the system does not have a complete prior")
        if isinstance(self.prior, CausalTransformerPrior):
            side = math.isqrt(self.prior.sequence_length)
            if side * side != self.prior.sequence_length:
                raise ValueError("VQGAN token sequence must form a square grid")
            indices = self.prior.sample(
                count, device=device, temperature=temperature
            ).reshape(count, side, side)
        elif isinstance(self.prior, PixelCNNPrior):
            indices = self.prior.sample(
                count,
                8,
                8,
                device=device,
                temperature=temperature,
            )
        else:
            raise TypeError("unsupported token-prior type")
        return self.tokenizer.decode_indices(indices)


def make_test_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
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


def _load_checkpoint(path: Path, expected_model_name: str) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} does not contain a checkpoint mapping")
    if checkpoint.get("model_name") != expected_model_name:
        raise ValueError(
            f"{path} has model_name={checkpoint.get('model_name')!r}; "
            f"expected {expected_model_name!r}"
        )
    return checkpoint


def _training_config(checkpoint: dict[str, object]) -> dict[str, object]:
    value = checkpoint.get("training_config", {})
    if not isinstance(value, dict):
        raise TypeError("training_config must be a mapping")
    return dict(value)


def _vqgan_label(checkpoint: dict[str, object], tokenizer_path: Path) -> str:
    training_config = _training_config(checkpoint)
    run_name = str(training_config.get("run_name", "default"))
    if run_name != "default":
        return f"vqgan_{run_name}"
    weighting = training_config.get("adversarial_weighting")
    if weighting in {"fixed", "adaptive"}:
        return f"vqgan_{weighting}"
    if tokenizer_path.parent.name != "vqgan":
        return f"vqgan_{tokenizer_path.parent.name}"
    return "vqgan_default"


def load_vqgan_system(tokenizer_path: Path, device: torch.device) -> EvaluatedSystem:
    checkpoint = _load_checkpoint(tokenizer_path, "vqgan_tokenizer")
    if checkpoint.get("perceptual_loss") != "lpips-v0.1-vgg":
        raise ValueError(f"{tokenizer_path} was not trained with LPIPS v0.1 VGG")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("VQGAN model_config must be a mapping")
    tokenizer = VQPerceptualAutoencoder32(**model_config)
    tokenizer.load_state_dict(checkpoint["state_dict"], strict=True)
    interface_id = model_state_fingerprint(tokenizer)
    if checkpoint.get("interface_id") != interface_id:
        raise ValueError(f"{tokenizer_path} has no valid tokenizer interface identity")

    discriminator_config = checkpoint.get("discriminator_config")
    discriminator_state = checkpoint.get("discriminator_state_dict")
    if discriminator_config is None or discriminator_state is None:
        raise ValueError(
            f"{tokenizer_path} does not contain its trained PatchGAN artifact"
        )
    if not isinstance(discriminator_config, dict) or not isinstance(
        discriminator_state, dict
    ):
        raise TypeError("PatchGAN configuration and state must be mappings")
    discriminator = PatchDiscriminator32(**discriminator_config)
    discriminator.load_state_dict(discriminator_state, strict=True)

    prior_path = tokenizer_path.with_name("transformer_prior.pth")
    prior: CausalTransformerPrior | None = None
    if prior_path.is_file():
        prior_checkpoint = _load_checkpoint(prior_path, "vqgan_transformer_prior")
        prior_config = prior_checkpoint.get("model_config")
        if not isinstance(prior_config, dict):
            raise TypeError("VQGAN prior model_config must be a mapping")
        prior = CausalTransformerPrior(**prior_config)
        prior.load_state_dict(prior_checkpoint["state_dict"], strict=True)
        if prior_checkpoint.get("tokenizer_interface_id") != interface_id:
            raise ValueError(f"{prior_path} was trained for another tokenizer")

    return EvaluatedSystem(
        name=_vqgan_label(checkpoint, tokenizer_path),
        family="vqgan",
        tokenizer=tokenizer.to(device).eval().requires_grad_(False),
        prior=(
            prior.to(device).eval().requires_grad_(False) if prior is not None else None
        ),
        discriminator=discriminator.to(device).eval().requires_grad_(False),
        tokenizer_checkpoint=tokenizer_path,
        prior_checkpoint=prior_path if prior is not None else None,
        training_config=_training_config(checkpoint),
    )


def load_vq_vae_baseline(
    tokenizer_path: Path,
    prior_path: Path,
    device: torch.device,
) -> EvaluatedSystem | None:
    if not tokenizer_path.is_file():
        return None
    checkpoint = _load_checkpoint(tokenizer_path, "vq_vae_tokenizer")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("VQ-VAE model_config must be a mapping")
    tokenizer = VQVAE32(**model_config)
    tokenizer.load_state_dict(checkpoint["state_dict"], strict=True)
    interface_id = model_state_fingerprint(tokenizer)
    if checkpoint.get("interface_id") != interface_id:
        raise ValueError(f"{tokenizer_path} has no valid tokenizer interface identity")

    prior: PixelCNNPrior | None = None
    if prior_path.is_file():
        prior_checkpoint = _load_checkpoint(prior_path, "vq_vae_pixelcnn_prior")
        prior_config = prior_checkpoint.get("model_config")
        if not isinstance(prior_config, dict):
            raise TypeError("VQ-VAE prior model_config must be a mapping")
        prior = PixelCNNPrior(**prior_config)
        prior.load_state_dict(prior_checkpoint["state_dict"], strict=True)
        if prior_checkpoint.get("tokenizer_interface_id") != interface_id:
            raise ValueError(f"{prior_path} was trained for another tokenizer")

    return EvaluatedSystem(
        name="vq_vae_baseline",
        family="vq_vae",
        tokenizer=tokenizer.to(device).eval().requires_grad_(False),
        prior=(
            prior.to(device).eval().requires_grad_(False) if prior is not None else None
        ),
        discriminator=None,
        tokenizer_checkpoint=tokenizer_path,
        prior_checkpoint=prior_path if prior is not None else None,
        training_config={"direct_baseline": True},
    )


def discover_vqgan_tokenizers(args: argparse.Namespace) -> list[Path]:
    if args.vqgan_tokenizer:
        candidates = list(args.vqgan_tokenizer)
    else:
        root = OUTPUT_ROOT / "vqgan"
        candidates = []
        flat = root / "tokenizer.pth"
        if flat.is_file():
            candidates.append(flat)
        candidates.extend(sorted(root.glob("*/tokenizer.pth")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def load_systems(
    args: argparse.Namespace, device: torch.device
) -> list[EvaluatedSystem]:
    tokenizer_paths = discover_vqgan_tokenizers(args)
    if not tokenizer_paths:
        raise FileNotFoundError(
            "no VQGAN tokenizer exists; train 7.0_vqgan.py first or pass "
            "--vqgan-tokenizer"
        )
    systems = [load_vqgan_system(path, device) for path in tokenizer_paths]
    if not args.skip_vq_vae_baseline:
        baseline = load_vq_vae_baseline(
            args.vq_vae_tokenizer, args.vq_vae_prior, device
        )
        if baseline is not None:
            systems.append(baseline)

    counts: dict[str, int] = {}
    for system in systems:
        count = counts.get(system.name, 0) + 1
        counts[system.name] = count
        if count > 1:
            system.name = f"{system.name}_{count}"
    return systems


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
    reconstruction_reference = FeatureMoments(feature_dim)
    generation_reference = FeatureMoments(feature_dim)
    examples = 0
    generation_count = 0
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        features = feature_extractor(x)
        reconstruction_reference.update(features)
        if generation_count < generation_examples:
            count = min(x.shape[0], generation_examples - generation_count)
            generation_reference.update(features[:count])
            generation_count += count
        examples += x.shape[0]
    return reconstruction_reference, generation_reference


@torch.inference_mode()
def evaluate_reconstruction(
    system: EvaluatedSystem,
    loader: DataLoader,
    feature_extractor: nn.Module,
    perceptual: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    reconstruction_moments = FeatureMoments(feature_dim)
    usage = TokenUsageAccumulator(system.vocabulary_size)
    paired_totals = torch.zeros(4, device=device)
    discriminator_totals = torch.zeros(2, device=device)
    squared_error = 0.0
    element_count = 0
    prior_nll = 0.0
    prior_examples = 0
    examples = 0
    positions = 0
    comparison: Tensor | None = None
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        reconstruction, indices, diagnostics = system.reconstruct_and_tokens(x)
        if comparison is None:
            comparison = torch.cat((x[:16], reconstruction[:16])).cpu()
        positions = indices.shape[1] * indices.shape[2]
        reconstruction_moments.update(feature_extractor(reconstruction))
        squared_error += float((reconstruction - x).square().sum())
        element_count += x.numel()
        paired_totals += (
            torch.stack(
                [
                    F.l1_loss(reconstruction, x),
                    perceptual(reconstruction, x),
                    structural_similarity_index(reconstruction, x),
                    diagnostics["quantization_mse"],
                ]
            )
            * x.shape[0]
        )
        usage.update(indices)
        loss = system.prior_loss(indices)
        if loss is not None:
            prior_nll += float(loss) * x.shape[0]
            prior_examples += x.shape[0]
        if system.discriminator is not None:
            discriminator_totals += (
                torch.stack(
                    [
                        system.discriminator(x).mean(),
                        system.discriminator(reconstruction).mean(),
                    ]
                )
                * x.shape[0]
            )
        examples += x.shape[0]

    if comparison is None or examples < 2:
        raise ValueError("evaluation needs at least two held-out examples")
    paired = (paired_totals / examples).tolist()
    mse = squared_error / element_count
    token_statistics = usage.statistics()
    entropy_bits = float(token_statistics["token_entropy_nats"]) / math.log(2)
    prior_metrics: dict[str, object] | None = None
    if prior_examples:
        assert system.prior is not None
        nll = prior_nll / prior_examples
        prior_metrics = {
            "nll_nats_per_token": nll,
            "bits_per_token": nll / math.log(2),
            "bits_per_image": positions * nll / math.log(2),
            "parameter_count": sum(
                parameter.numel() for parameter in system.prior.parameters()
            ),
        }
    discriminator_metrics: dict[str, float] | None = None
    if system.discriminator is not None:
        logits = (discriminator_totals / examples).tolist()
        discriminator_metrics = {
            "real_logit": logits[0],
            "reconstruction_logit": logits[1],
            "real_minus_reconstruction_logit": logits[0] - logits[1],
        }
    return {
        "examples": examples,
        "paired_fidelity": {
            "pixel_l1": paired[0],
            "mse": mse,
            "psnr_for_minus_one_to_one_range": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
            "ssim": paired[2],
            "lpips_v0_1_vgg": paired[1],
        },
        "reconstruction_distribution": {
            "projected_inception_frechet": frechet_distance(
                real_moments, reconstruction_moments
            )
        },
        "quantization": {
            "mse": paired[3],
            "vocabulary_size": system.vocabulary_size,
            "positions": positions,
            "active_codes": int(token_statistics["active_codes"]),
            "usage_fraction": float(token_statistics["usage_fraction"]),
            "perplexity": float(token_statistics["perplexity"]),
            "marginal_entropy_bits_per_token": entropy_bits,
            "marginal_entropy_bits_per_image": positions * entropy_bits,
            "fixed_length_bits_per_image": positions
            * math.ceil(math.log2(system.vocabulary_size)),
        },
        "patch_discriminator": discriminator_metrics,
        "prior": prior_metrics,
    }, comparison


@torch.inference_mode()
def evaluate_generation(
    system: EvaluatedSystem,
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
    sample_batches: list[Tensor] = []
    saved_examples = 0
    elapsed = 0.0
    generated = 0
    while generated < examples:
        count = min(batch_size, examples - generated)
        start = time.perf_counter()
        images = system.sample(count, device=device, temperature=temperature)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        generated_moments.update(feature_extractor(images))
        if saved_examples < 64:
            save_count = min(64 - saved_examples, images.shape[0])
            sample_batches.append(images[:save_count].cpu())
            saved_examples += save_count
        generated += count
    return {
        "projected_inception_frechet": frechet_distance(
            real_moments, generated_moments
        ),
        "seconds": elapsed,
        "images_per_second": examples / max(elapsed, 1e-12),
        "temperature": temperature,
    }, torch.cat(sample_batches)


def _artifact_stem(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in name
    )


def _comparison_summary(
    model_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name, result in model_results.items():
        paired = result["paired_fidelity"]
        distribution = result["reconstruction_distribution"]
        quantization = result["quantization"]
        prior = result["prior"]
        generation = result["generation"]
        assert isinstance(paired, dict)
        assert isinstance(distribution, dict)
        assert isinstance(quantization, dict)
        summary[name] = {
            "family": result["family"],
            "training_config": result["training_config"],
            "pixel_l1": paired["pixel_l1"],
            "lpips_v0_1_vgg": paired["lpips_v0_1_vgg"],
            "reconstruction_frechet_proxy": distribution["projected_inception_frechet"],
            "token_entropy_bits_per_image": quantization[
                "marginal_entropy_bits_per_image"
            ],
            "prior_bits_per_token": (
                prior["bits_per_token"] if isinstance(prior, dict) else None
            ),
            "generation_frechet_proxy": (
                generation["projected_inception_frechet"]
                if isinstance(generation, dict)
                else None
            ),
        }
    return summary


def evaluate(args: argparse.Namespace) -> None:
    if args.max_examples < 2 or args.generation_examples < 2:
        raise ValueError("reconstruction and generation evaluation need >=2 examples")
    if args.generation_batch_size < 1 or args.batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_test_loader(args, device)
    systems = load_systems(args, device)
    projection_dim = (
        None if args.inception_projection_dim == 0 else args.inception_projection_dim
    )
    feature_extractor = TorchvisionInceptionFeatures(
        projection_dim=projection_dim,
        projection_seed=args.feature_seed,
    ).to(device)
    perceptual = build_perceptual_loss("lpips", device)
    feature_dim = feature_extractor.feature_dim
    real_reconstruction, real_generation = real_feature_moments(
        loader,
        feature_extractor,
        feature_dim=feature_dim,
        max_examples=args.max_examples,
        generation_examples=args.generation_examples,
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_results: dict[str, dict[str, object]] = {}
    results: dict[str, object] = {
        "protocol": {
            "dataset": "CIFAR-10 test",
            "max_reconstruction_examples": args.max_examples,
            "generation_examples": args.generation_examples,
            "generation_batch_size": args.generation_batch_size,
            "sampling_temperature": args.temperature,
            "seed": args.seed,
            "feature_projection_seed": args.feature_seed,
            "paired_perceptual_metric": "LPIPS v0.1, VGG backbone",
            "feature_extractor": "torchvision Inception-v3 pool features",
            "fixed_random_projection_dimension": projection_dim,
            "fid_warning": (
                "These values use torchvision preprocessing and are not "
                "canonical TensorFlow FID values."
            ),
            "baseline_warning": (
                "VQ-VAE and VQGAN use different tokenizer architectures; "
                "cross-script differences are system comparisons, not causal "
                "loss ablations."
            ),
        },
        "models": model_results,
    }
    for system in systems:
        metrics, comparison = evaluate_reconstruction(
            system,
            loader,
            feature_extractor,
            perceptual,
            real_reconstruction,
            feature_dim=feature_dim,
            max_examples=args.max_examples,
            device=device,
        )
        stem = _artifact_stem(system.name)
        save_image(
            comparison.mul(0.5).add(0.5),
            args.output_dir / f"{stem}_real_and_reconstruction.png",
            nrow=min(16, comparison.shape[0] // 2),
        )
        generation: dict[str, float] | None = None
        if system.has_complete_prior:
            generation, images = evaluate_generation(
                system,
                feature_extractor,
                real_generation,
                feature_dim=feature_dim,
                examples=args.generation_examples,
                batch_size=args.generation_batch_size,
                temperature=args.temperature,
                device=device,
            )
            save_image(
                images.mul(0.5).add(0.5),
                args.output_dir / f"{stem}_prior_samples.png",
                nrow=8,
            )
        model_results[system.name] = {
            "family": system.family,
            "artifacts": {
                "tokenizer": str(system.tokenizer_checkpoint),
                "prior": (
                    str(system.prior_checkpoint)
                    if system.prior_checkpoint is not None
                    else None
                ),
            },
            "training_config": system.training_config,
            **metrics,
            "generation": generation,
        }
        paired = metrics["paired_fidelity"]
        distribution = metrics["reconstruction_distribution"]
        assert isinstance(paired, dict)
        assert isinstance(distribution, dict)
        print(
            f"{system.name}: L1={paired['pixel_l1']:.4f}, "
            f"LPIPS={paired['lpips_v0_1_vgg']:.4f}, "
            "rFID-proxy="
            f"{distribution['projected_inception_frechet']:.2f}, "
            f"prior={'yes' if system.has_complete_prior else 'no'}"
        )
    results["comparison_summary"] = _comparison_summary(model_results)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved evaluation to {args.output_dir}")


class TinyFeatures(nn.Module):
    feature_dim = 3

    def forward(self, images: Tensor) -> Tensor:
        return images.mean(dim=(2, 3))


def smoke_test() -> None:
    set_seed(7)
    device = torch.device("cpu")
    tokenizer = VQPerceptualAutoencoder32(
        latent_channels=4, codebook_size=16, hidden_channels=16
    )
    discriminator = PatchDiscriminator32(base_channels=8)
    prior_config = {
        "vocabulary_size": 16,
        "sequence_length": 64,
        "model_dim": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.0,
    }
    prior = CausalTransformerPrior(**prior_config)
    interface_id = model_state_fingerprint(tokenizer)
    baseline_system: EvaluatedSystem | None = None
    with TemporaryDirectory(prefix="vqgan-evaluation-") as directory:
        tokenizer_path = Path(directory) / "tokenizer.pth"
        atomic_torch_save(
            {
                "model_name": "vqgan_tokenizer",
                "interface_id": interface_id,
                "state_dict": tokenizer.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "discriminator_config": {"base_channels": 8},
                "perceptual_loss": "lpips-v0.1-vgg",
                "model_config": {
                    "latent_channels": 4,
                    "codebook_size": 16,
                    "hidden_channels": 16,
                },
                "training_config": {
                    "run_name": "smoke",
                    "adversarial_weighting": "fixed",
                },
            },
            tokenizer_path,
        )
        atomic_torch_save(
            {
                "model_name": "vqgan_transformer_prior",
                "state_dict": prior.state_dict(),
                "model_config": prior_config,
                "tokenizer_interface_id": interface_id,
            },
            tokenizer_path.with_name("transformer_prior.pth"),
        )
        system = load_vqgan_system(tokenizer_path, device)
        baseline_tokenizer = VQVAE32(
            hidden_channels=16, embedding_dim=4, codebook_size=16
        )
        baseline_prior = PixelCNNPrior(16, hidden_channels=8, layers=1)
        baseline_interface_id = model_state_fingerprint(baseline_tokenizer)
        baseline_directory = Path(directory) / "vq-vae"
        baseline_tokenizer_path = baseline_directory / "tokenizer.pth"
        baseline_prior_path = baseline_directory / "pixelcnn_prior.pth"
        atomic_torch_save(
            {
                "model_name": "vq_vae_tokenizer",
                "interface_id": baseline_interface_id,
                "state_dict": baseline_tokenizer.state_dict(),
                "model_config": {
                    "hidden_channels": 16,
                    "embedding_dim": 4,
                    "codebook_size": 16,
                },
            },
            baseline_tokenizer_path,
        )
        atomic_torch_save(
            {
                "model_name": "vq_vae_pixelcnn_prior",
                "state_dict": baseline_prior.state_dict(),
                "model_config": {
                    "vocabulary_size": 16,
                    "hidden_channels": 8,
                    "layers": 1,
                },
                "tokenizer_interface_id": baseline_interface_id,
            },
            baseline_prior_path,
        )
        baseline_system = load_vq_vae_baseline(
            baseline_tokenizer_path, baseline_prior_path, device
        )
        loader = DataLoader(
            TensorDataset(
                torch.rand(8, 3, 32, 32).mul(2).sub(1),
                torch.arange(8) % 10,
            ),
            batch_size=4,
        )
        features = TinyFeatures()
        perceptual = RandomFeaturePerceptualLoss(channels=4)
        real, generated_reference = real_feature_moments(
            loader,
            features,
            feature_dim=features.feature_dim,
            max_examples=8,
            generation_examples=2,
            device=device,
        )
        metrics, comparison = evaluate_reconstruction(
            system,
            loader,
            features,
            perceptual,
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
            examples=2,
            batch_size=2,
            temperature=1.0,
            device=device,
        )
    assert comparison.shape == (8, 3, 32, 32)
    assert images.shape == (2, 3, 32, 32)
    assert baseline_system is not None and baseline_system.has_complete_prior
    assert metrics["patch_discriminator"] is not None
    assert metrics["prior"] is not None
    quantization = metrics["quantization"]
    assert isinstance(quantization, dict)
    assert quantization["fixed_length_bits_per_image"] == 256
    assert math.isfinite(generation["projected_inception_frechet"])
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=2_048)
    parser.add_argument("--generation-examples", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--inception-projection-dim", type=int, default=256)
    parser.add_argument("--feature-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--vqgan-tokenizer",
        type=Path,
        action="append",
        help=(
            "repeatable tokenizer checkpoint path; its Transformer prior is "
            "loaded from the same directory"
        ),
    )
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
    parser.add_argument("--skip-vq-vae-baseline", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT / "vqgan_evaluation",
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
