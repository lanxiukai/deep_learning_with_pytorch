"""Analyze the frozen standard-VAE baseline without retraining it.

The script keeps three image paths separate:

* posterior-mean reconstruction is deterministic and reads a real image;
* posterior-sample reconstruction also reads a real image;
* prior generation decodes an independent ``N(0, I)`` draw.

Latent interpolation is retained as a local decoder diagnostic, not evidence
of independent semantic factors. The training script and its 256x256 model
remain unchanged as the introductory baseline for later compact lessons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.vae import VAE, device, diagonal_gaussian_kl, reparameterize


PROJECT_ROOT = infer_project_root()


class PosteriorAccumulator:
    """Accumulate standard-VAE posterior diagnostics without storing samples."""

    def __init__(self, latent_dims: int) -> None:
        self.mu_total = torch.zeros(latent_dims, dtype=torch.float64)
        self.mu_square_total = torch.zeros_like(self.mu_total)
        self.std_square_total = torch.zeros_like(self.mu_total)
        self.kl_total = torch.zeros_like(self.mu_total)
        self.examples = 0

    def update(self, mu: Tensor, std: Tensor) -> None:
        self.mu_total += mu.detach().double().sum(dim=0).cpu()
        self.mu_square_total += mu.detach().double().square().sum(dim=0).cpu()
        self.std_square_total += std.detach().double().square().sum(dim=0).cpu()
        self.kl_total += diagonal_gaussian_kl(mu, std).detach().double().sum(dim=0).cpu()
        self.examples += mu.shape[0]

    def metrics(self, *, active_variance_threshold: float) -> dict[str, object]:
        if self.examples == 0:
            raise ValueError("no posterior values were accumulated")
        mean = self.mu_total / self.examples
        variance_of_mu = (
            self.mu_square_total / self.examples - mean.square()
        ).clamp_min(0.0)
        posterior_variance = self.std_square_total / self.examples
        kl_per_dimension = self.kl_total / self.examples
        return {
            "examples": self.examples,
            "posterior_mean_by_dimension": mean.tolist(),
            "posterior_mean_std_by_dimension": variance_of_mu.sqrt().tolist(),
            "posterior_std_by_dimension": posterior_variance.sqrt().tolist(),
            "kl_nats_per_dimension": kl_per_dimension.tolist(),
            "kl_nats_per_image": float(kl_per_dimension.sum()),
            "active_variance_threshold": active_variance_threshold,
            "active_dimensions": int(
                (variance_of_mu > active_variance_threshold).sum()
            ),
        }


def infer_latent_dims(state_dict: dict[str, Tensor]) -> int:
    weight = state_dict.get("encoder.linear2.weight")
    if weight is None or weight.ndim != 2:
        raise ValueError("checkpoint does not contain the baseline posterior head")
    return weight.shape[0]


@torch.inference_mode()
def evaluate(
    model: VAE,
    loader,
    *,
    latent_dims: int,
    maximum_batches: int,
    active_variance_threshold: float,
) -> tuple[dict[str, object], Tensor, Tensor, Tensor]:
    model.eval()
    accumulator = PosteriorAccumulator(latent_dims)
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    comparison = None
    interpolation = None
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        images = images.to(device)
        mu, std = model.encoder.statistics(images)
        mean_reconstruction = model.decoder(mu)
        sample_reconstruction = model.decoder(reparameterize(mu, std))
        squared_error += float((mean_reconstruction - images).square().sum())
        absolute_error += float((mean_reconstruction - images).abs().sum())
        elements += images.numel()
        accumulator.update(mu, std)
        if comparison is None:
            count = min(8, images.shape[0])
            comparison = torch.cat(
                (
                    images[:count],
                    mean_reconstruction[:count],
                    sample_reconstruction[:count],
                )
            ).cpu()
            if count >= 2:
                weights = torch.linspace(0, 1, 7, device=device)[:, None]
                path = (1.0 - weights) * mu[0] + weights * mu[1]
                interpolation = model.decoder(path).cpu()
    if elements == 0 or comparison is None:
        raise ValueError("cannot analyze an empty loader")
    if interpolation is None:
        raise ValueError("at least two images are required for interpolation")
    mse = squared_error / elements
    metrics = {
        "posterior_mean_reconstruction": {
            "pixel_l1": absolute_error / elements,
            "pixel_mse": mse,
            "psnr_for_zero_to_one_range": 10.0
            * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item(),
        },
        "posterior": accumulator.metrics(
            active_variance_threshold=active_variance_threshold
        ),
        "path_boundaries": {
            "posterior_mean_reconstruction": "reads a real image",
            "posterior_sample_reconstruction": "reads a real image",
            "prior_generation": "decodes an independent standard-normal draw",
            "interpolation": "local decoder diagnostic, not unconditional generation",
        },
    }
    prior_samples = model.decoder(
        torch.randn(18, latent_dims, device=device)
    ).cpu()
    return metrics, comparison, prior_samples, interpolation


def smoke_test() -> None:
    torch.manual_seed(7)
    model = VAE(latent_dims=2).eval()
    images = torch.rand(2, 3, 256, 256)
    with torch.inference_mode():
        mu, std = model.encoder.statistics(images)
        mean_reconstruction = model.decoder(mu)
        sample_reconstruction = model.decoder(reparameterize(mu, std))
        prior = model.decoder(torch.randn(2, 2))
    accumulator = PosteriorAccumulator(2)
    accumulator.update(mu, std)
    metrics = accumulator.metrics(active_variance_threshold=1e-2)
    assert mean_reconstruction.shape == sample_reconstruction.shape == images.shape
    assert prior.shape == images.shape
    assert float(metrics["kl_nats_per_image"]) >= 0
    print("smoke test passed: reconstruction and prior paths are distinct")


def analyze(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {args.checkpoint}; run 1.0_vae_train.py first"
        )
    state_dict = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    latent_dims = infer_latent_dims(state_dict)
    model = VAE(latent_dims=latent_dims).to(device)
    model.load_state_dict(state_dict, strict=True)
    loader = image_folder_loader(
        args.data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics, comparison, prior_samples, interpolation = evaluate(
        model,
        loader,
        latent_dims=latent_dims,
        maximum_batches=args.maximum_batches,
        active_variance_threshold=args.active_variance_threshold,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_image(
        comparison,
        args.output / "real_mean_and_sample_reconstruction.png",
        nrow=comparison.shape[0] // 3,
    )
    save_image(
        prior_samples,
        args.output / "standard_normal_prior_samples.png",
        nrow=6,
    )
    save_image(
        interpolation,
        args.output / "posterior_mean_interpolation_not_generation.png",
        nrow=len(interpolation),
    )
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "vae" / "VAEglasses.pth",
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "glasses-256"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "vae" / "vae_analysis",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-batches", type=int, default=100)
    parser.add_argument("--active-variance-threshold", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        analyze(args)


if __name__ == "__main__":
    main()
