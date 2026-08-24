"""Validate the KL-AE first stage before latent diffusion.

An optional VQGAN checkpoint provides a discrete-tokenizer coordinate. Both
models use the same paired pixel, local SSIM, frozen VGG-feature, and
projected Inception reconstruction-distribution protocol. Model-specific
evidence stays separate: VQGAN reports token usage and optional Transformer
prior cross-entropy, while KL-AE reports posterior moments, KL, sample/mean
gap, and the frozen scaling interface.

The attached VQ token prior and the missing KL continuous-latent model are not
treated as a matched generation comparison. Nominal discrete bits, float
storage, and variational KL are reported in different fields because they are
not interchangeable rate definitions.
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
    structural_similarity_index,
)
from dl_utils.vae.perceptual_autoencoder import (
    KLPerceptualAutoencoder32,
    RandomFeaturePerceptualLoss,
    VGGPerceptualLoss,
    VQPerceptualAutoencoder32,
)
from dl_utils.vae.quantization import TokenUsageAccumulator
from dl_utils.vae.token_prior import CausalTransformerPrior
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output"


@dataclass
class VQGANSystem:
    tokenizer: VQPerceptualAutoencoder32
    checkpoint: dict[str, object]
    prior: CausalTransformerPrior | None
    prior_checkpoint: dict[str, object] | None


@dataclass
class KLAutoencoderSystem:
    model: KLPerceptualAutoencoder32
    checkpoint: dict[str, object]
    latent_scale: float


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


def load_vqgan(
    tokenizer_path: Path,
    prior_path: Path,
    device: torch.device,
) -> VQGANSystem | None:
    if not tokenizer_path.is_file():
        return None
    checkpoint = torch.load(
        tokenizer_path, map_location=device, weights_only=True
    )
    if checkpoint.get("model_name") != "vqgan_tokenizer":
        raise ValueError(f"{tokenizer_path} is not a VQGAN tokenizer")
    tokenizer = VQPerceptualAutoencoder32(
        **checkpoint["model_config"]
    ).to(device)
    tokenizer.load_state_dict(checkpoint["state_dict"], strict=True)
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    if checkpoint.get("interface_id") != tokenizer_interface_id:
        raise ValueError("VQGAN tokenizer has no valid interface identity")
    prior = None
    prior_checkpoint = None
    if prior_path.is_file():
        prior_checkpoint = torch.load(
            prior_path, map_location=device, weights_only=True
        )
        if prior_checkpoint.get("model_name") != "vqgan_transformer_prior":
            raise ValueError(f"{prior_path} is not a VQGAN Transformer prior")
        prior = CausalTransformerPrior(
            **prior_checkpoint["model_config"]
        ).to(device)
        prior.load_state_dict(prior_checkpoint["state_dict"], strict=True)
        if (
            prior_checkpoint.get("tokenizer_interface_id")
            != tokenizer_interface_id
        ):
            raise ValueError("VQGAN prior was trained for a different tokenizer")
        if prior.vocabulary_size != tokenizer.quantizer.codebook_size:
            raise ValueError("VQGAN tokenizer and prior vocabularies do not match")
    return VQGANSystem(
        tokenizer.eval().requires_grad_(False),
        checkpoint,
        None if prior is None else prior.eval().requires_grad_(False),
        prior_checkpoint,
    )


def load_kl_autoencoder(
    path: Path, device: torch.device
) -> KLAutoencoderSystem | None:
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "kl_perceptual_autoencoder":
        raise ValueError(f"{path} is not a KL perceptual autoencoder")
    model = KLPerceptualAutoencoder32(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("interface_id") != model_state_fingerprint(model):
        raise ValueError("KL-AE checkpoint has no valid interface identity")
    latent_interface = checkpoint.get("latent_interface")
    if not isinstance(latent_interface, dict) or "latent_scale" not in latent_interface:
        raise ValueError("KL-AE checkpoint has no frozen latent-scale interface")
    latent_scale = float(latent_interface["latent_scale"])
    if latent_scale <= 0:
        raise ValueError("KL-AE latent scale must be positive")
    return KLAutoencoderSystem(
        model.eval().requires_grad_(False), checkpoint, latent_scale
    )


def paired_values(
    reconstruction: Tensor,
    target: Tensor,
    perceptual: nn.Module,
) -> Tensor:
    return torch.stack(
        [
            F.l1_loss(reconstruction, target),
            F.mse_loss(reconstruction, target),
            structural_similarity_index(reconstruction, target),
            perceptual(reconstruction, target),
        ]
    ).double()


def finish_paired_metrics(values: Tensor) -> dict[str, float]:
    l1, mse, ssim, feature = values.tolist()
    return {
        "pixel_l1": l1,
        "pixel_mse": mse,
        "psnr_for_minus_one_to_one_range": 10.0
        * math.log10(4.0 / max(mse, 1e-12)),
        "ssim_uniform_window": ssim,
        "frozen_feature_l1": feature,
    }


@torch.inference_mode()
def collect_real_moments(
    loader: DataLoader,
    feature_extractor: nn.Module,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
) -> FeatureMoments:
    moments = FeatureMoments(feature_dim)
    examples = 0
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        moments.update(feature_extractor(x))
        examples += x.shape[0]
    if examples < 2:
        raise ValueError("at least two examples are required for feature moments")
    return moments


@torch.inference_mode()
def evaluate_vqgan(
    system: VQGANSystem,
    loader: DataLoader,
    perceptual: nn.Module,
    feature_extractor: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    model = system.tokenizer
    vocabulary_size = model.quantizer.codebook_size
    reconstruction_moments = FeatureMoments(feature_dim)
    usage = TokenUsageAccumulator(vocabulary_size)
    paired_total = torch.zeros(4, device=device, dtype=torch.float64)
    quantization_total = 0.0
    prior_nll_total = 0.0
    examples = 0
    comparison = None
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(x)
        paired_total += paired_values(reconstruction, x, perceptual) * x.shape[0]
        reconstruction_moments.update(feature_extractor(reconstruction))
        usage.update(indices)
        quantization_total += float(diagnostics["quantization_mse"]) * x.shape[0]
        if system.prior is not None:
            logits, targets = system.prior.teacher_forcing(indices)
            prior_nll_total += float(
                F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            ) * x.shape[0]
        examples += x.shape[0]
        if comparison is None:
            comparison = torch.cat((x[:16], reconstruction[:16])).cpu()
    if examples == 0 or comparison is None:
        raise ValueError("cannot evaluate VQGAN on an empty loader")
    token_statistics = usage.statistics()
    entropy_bits = float(token_statistics["token_entropy_nats"]) / math.log(2)
    positions = 64
    prior_metrics = None
    if system.prior is not None:
        nll = prior_nll_total / examples
        prior_metrics = {
            "held_out_nll_nats_per_token": nll,
            "held_out_bits_per_token": nll / math.log(2),
            "held_out_bits_per_image": positions * nll / math.log(2),
            "parameter_count": sum(
                parameter.numel() for parameter in system.prior.parameters()
            ),
        }
    return {
        "examples": examples,
        "reconstruction": finish_paired_metrics(paired_total / examples),
        "projected_inception_reconstruction_frechet": frechet_distance(
            real_moments, reconstruction_moments
        ),
        "latent_interface": {
            "type": "discrete_token_grid",
            "shape": [8, 8],
            "vocabulary_size": vocabulary_size,
            "active_codes": int(token_statistics["active_codes"]),
            "usage_fraction": float(token_statistics["usage_fraction"]),
            "perplexity": float(token_statistics["perplexity"]),
            "quantization_mse": quantization_total / examples,
            "marginal_entropy_bits_per_token": entropy_bits,
            "marginal_entropy_bits_per_image": positions * entropy_bits,
            "fixed_length_bits_per_image": positions
            * math.ceil(math.log2(vocabulary_size)),
        },
        "attached_transformer_prior": prior_metrics,
        "unconditional_generation_available": system.prior is not None,
    }, comparison


@torch.inference_mode()
def evaluate_kl_autoencoder(
    system: KLAutoencoderSystem,
    loader: DataLoader,
    perceptual: nn.Module,
    feature_extractor: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
    active_variance_threshold: float,
) -> tuple[dict[str, object], Tensor]:
    model = system.model
    channels = model.latent_channels
    mean_moments = FeatureMoments(feature_dim)
    sample_moments = FeatureMoments(feature_dim)
    mean_paired_total = torch.zeros(4, device=device, dtype=torch.float64)
    sample_paired_total = torch.zeros_like(mean_paired_total)
    mu_total = torch.zeros(channels, device=device, dtype=torch.float64)
    mu_square_total = torch.zeros_like(mu_total)
    posterior_variance_total = torch.zeros_like(mu_total)
    kl_total_by_channel = torch.zeros_like(mu_total)
    sample_mean_mse_total = 0.0
    interface_error = 0.0
    values_per_channel = 0
    examples = 0
    comparison = None
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        z = reparameterize_logvar(mu, logvar)
        mean_reconstruction = model.decoder(mu)
        sample_reconstruction = model.decoder(z)
        mean_paired_total += paired_values(
            mean_reconstruction, x, perceptual
        ) * x.shape[0]
        sample_paired_total += paired_values(
            sample_reconstruction, x, perceptual
        ) * x.shape[0]
        mean_moments.update(feature_extractor(mean_reconstruction))
        sample_moments.update(feature_extractor(sample_reconstruction))
        sample_mean_mse_total += float(
            F.mse_loss(sample_reconstruction, mean_reconstruction)
        ) * x.shape[0]

        mu64 = mu.double()
        logvar64 = logvar.double()
        variance64 = logvar64.exp()
        reduce_dimensions = (0, 2, 3)
        mu_total += mu64.sum(dim=reduce_dimensions)
        mu_square_total += mu64.square().sum(dim=reduce_dimensions)
        posterior_variance_total += variance64.sum(dim=reduce_dimensions)
        kl_total_by_channel += (
            0.5 * (mu64.square() + variance64 - 1.0 - logvar64)
        ).sum(dim=reduce_dimensions)
        values_per_channel += mu.shape[0] * mu.shape[2] * mu.shape[3]

        scaled_mean = model.encode_latent(
            x, sample=False, latent_scale=system.latent_scale
        )
        interface_reconstruction = model.decode_latent(
            scaled_mean, latent_scale=system.latent_scale
        )
        interface_error = max(
            interface_error,
            float((interface_reconstruction - mean_reconstruction).abs().max()),
        )
        examples += x.shape[0]
        if comparison is None:
            comparison = torch.cat(
                (x[:8], mean_reconstruction[:8], sample_reconstruction[:8])
            ).cpu()
    if examples == 0 or comparison is None:
        raise ValueError("cannot evaluate KL-AE on an empty loader")
    posterior_mean = mu_total / values_per_channel
    posterior_mean_square = mu_square_total / values_per_channel
    posterior_variance = posterior_variance_total / values_per_channel
    variance_of_mu = (
        posterior_mean_square - posterior_mean.square()
    ).clamp_min(0.0)
    expected_second_moment = posterior_mean_square + posterior_variance
    kl_per_channel_per_image = kl_total_by_channel / examples
    return {
        "examples": examples,
        "posterior_mean_reconstruction": finish_paired_metrics(
            mean_paired_total / examples
        ),
        "posterior_sample_reconstruction": finish_paired_metrics(
            sample_paired_total / examples
        ),
        "projected_inception_mean_reconstruction_frechet": frechet_distance(
            real_moments, mean_moments
        ),
        "projected_inception_sample_reconstruction_frechet": frechet_distance(
            real_moments, sample_moments
        ),
        "sample_mean_reconstruction_mse": sample_mean_mse_total / examples,
        "latent_interface": {
            "type": "spatial_diagonal_gaussian",
            "shape": [channels, 8, 8],
            "latent_scale": system.latent_scale,
            "posterior_mean_by_channel": posterior_mean.cpu().tolist(),
            "posterior_mean_std_by_channel": variance_of_mu.sqrt().cpu().tolist(),
            "posterior_std_by_channel": posterior_variance.sqrt().cpu().tolist(),
            "unscaled_posterior_rms_by_channel": (
                expected_second_moment.sqrt().cpu().tolist()
            ),
            "scaled_expected_second_moment": float(
                expected_second_moment.mean() * system.latent_scale**2
            ),
            "kl_nats_per_channel_per_image": (
                kl_per_channel_per_image.cpu().tolist()
            ),
            "kl_nats_per_image": float(kl_per_channel_per_image.sum()),
            "active_variance_threshold": active_variance_threshold,
            "active_channel_definition": (
                "variance of posterior means over examples and spatial positions"
            ),
            "active_channels": int(
                (variance_of_mu > active_variance_threshold).sum()
            ),
            "scaled_mean_encode_decode_max_abs_error": interface_error,
            "float32_storage_bits_per_image": channels * 8 * 8 * 32,
            "storage_warning": (
                "Float storage is not an information-rate estimate and is not "
                "comparable to VQ nominal token bits."
            ),
        },
        "attached_continuous_latent_model": None,
        "unconditional_generation_available": False,
    }, comparison


class TinyFeatures(nn.Module):
    """Small deterministic distribution feature map for offline smoke tests."""

    feature_dim = 12

    def forward(self, images: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(images, (2, 2))
        return pooled.flatten(1)


def smoke_test() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    images = torch.randn(8, 3, 32, 32).clamp(-1, 1)
    loader = DataLoader(TensorDataset(images, torch.zeros(8)), batch_size=4)
    perceptual = RandomFeaturePerceptualLoss(channels=4)
    feature_extractor = TinyFeatures()
    real_moments = collect_real_moments(
        loader,
        feature_extractor,
        feature_dim=feature_extractor.feature_dim,
        max_examples=8,
        device=device,
    )
    tokenizer = VQPerceptualAutoencoder32(
        latent_channels=8, codebook_size=16, hidden_channels=32
    )
    prior = CausalTransformerPrior(
        16, 64, model_dim=32, heads=4, layers=1
    )
    vq_metrics, _ = evaluate_vqgan(
        VQGANSystem(tokenizer, {}, prior, {}),
        loader,
        perceptual,
        feature_extractor,
        real_moments,
        feature_dim=feature_extractor.feature_dim,
        max_examples=8,
        device=device,
    )
    kl_metrics, _ = evaluate_kl_autoencoder(
        KLAutoencoderSystem(
            KLPerceptualAutoencoder32(
                latent_channels=4, hidden_channels=32
            ),
            {},
            0.5,
        ),
        loader,
        perceptual,
        feature_extractor,
        real_moments,
        feature_dim=feature_extractor.feature_dim,
        max_examples=8,
        device=device,
        active_variance_threshold=1e-2,
    )
    assert math.isfinite(
        float(vq_metrics["projected_inception_reconstruction_frechet"])
    )
    latent = kl_metrics["latent_interface"]
    assert isinstance(latent, dict)
    assert float(latent["scaled_mean_encode_decode_max_abs_error"]) < 1e-6
    assert kl_metrics["unconditional_generation_available"] is False
    print("smoke test passed: shared protocol and latent boundaries are finite")


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.max_examples < 2:
        raise ValueError("--max-examples must be at least two")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_test_loader(args, device)
    vqgan = load_vqgan(
        args.vqgan_tokenizer, args.vqgan_prior, device
    )
    kl_autoencoder = load_kl_autoencoder(args.kl_autoencoder, device)
    if kl_autoencoder is None:
        raise FileNotFoundError(
            "no KL first-stage checkpoint exists; run "
            "3.0_kl_autoencoder.py first"
        )
    if args.perceptual == "vgg":
        perceptual: nn.Module = VGGPerceptualLoss().to(device)
        perceptual_name = "frozen torchvision VGG16 feature L1"
    else:
        print("warning: random features are a debug distance, not a perceptual metric")
        perceptual = RandomFeaturePerceptualLoss().to(device)
        perceptual_name = "random frozen feature debug distance"
    projection_dim = (
        None if args.inception_projection_dim == 0
        else args.inception_projection_dim
    )
    feature_extractor = TorchvisionInceptionFeatures(
        projection_dim=projection_dim,
        projection_seed=args.feature_seed,
    ).to(device)
    real_moments = collect_real_moments(
        loader,
        feature_extractor,
        feature_dim=feature_extractor.feature_dim,
        max_examples=args.max_examples,
        device=device,
    )
    out_dir = OUTPUT_ROOT / "diffusion" / "first_stage_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "protocol": {
            "dataset": "CIFAR-10 test",
            "examples": args.max_examples,
            "input_value_range": [-1.0, 1.0],
            "paired_feature_metric": perceptual_name,
            "distribution_feature_extractor": (
                "torchvision Inception-v3 pool features"
            ),
            "fixed_random_projection_dimension": projection_dim,
            "frechet_warning": (
                "This uses torchvision preprocessing and is not canonical "
                "TensorFlow FID. Compare only runs from this exact protocol."
            ),
            "generation_comparison": (
                "not run: this lesson has no matched continuous latent model "
                "for KL-AE"
            ),
            "rate_warning": (
                "VQ nominal bits, empirical token entropy, KL, and float "
                "storage are distinct quantities."
            ),
        },
        "models": {},
    }
    models = results["models"]
    assert isinstance(models, dict)
    if vqgan is not None:
        metrics, comparison = evaluate_vqgan(
            vqgan,
            loader,
            perceptual,
            feature_extractor,
            real_moments,
            feature_dim=feature_extractor.feature_dim,
            max_examples=args.max_examples,
            device=device,
        )
        metrics["checkpoint"] = str(args.vqgan_tokenizer)
        models["vqgan"] = metrics
        save_image(
            comparison.mul(0.5).add(0.5),
            out_dir / "vqgan_real_and_reconstruction.png",
            nrow=16,
        )
    if kl_autoencoder is not None:
        metrics, comparison = evaluate_kl_autoencoder(
            kl_autoencoder,
            loader,
            perceptual,
            feature_extractor,
            real_moments,
            feature_dim=feature_extractor.feature_dim,
            max_examples=args.max_examples,
            device=device,
            active_variance_threshold=args.active_variance_threshold,
        )
        metrics["checkpoint"] = str(args.kl_autoencoder)
        metrics["checkpoint_interface_id"] = kl_autoencoder.checkpoint.get(
            "interface_id"
        )
        models["kl_autoencoder"] = metrics
        save_image(
            comparison.mul(0.5).add(0.5),
            out_dir / "kl_autoencoder_real_mean_and_sample.png",
            nrow=8,
        )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--vqgan-tokenizer",
        type=Path,
        default=OUTPUT_ROOT / "vqgan" / "tokenizer.pth",
    )
    parser.add_argument(
        "--vqgan-prior",
        type=Path,
        default=OUTPUT_ROOT / "vqgan" / "transformer_prior.pth",
    )
    parser.add_argument(
        "--kl-autoencoder",
        type=Path,
        default=OUTPUT_ROOT / "kl_autoencoder" / "kl_autoencoder.pth",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--perceptual", choices=("vgg", "random"), default="vgg")
    parser.add_argument("--inception-projection-dim", type=int, default=256)
    parser.add_argument("--feature-seed", type=int, default=2026)
    parser.add_argument("--active-variance-threshold", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
