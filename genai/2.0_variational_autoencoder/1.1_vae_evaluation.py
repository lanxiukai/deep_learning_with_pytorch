"""Evaluate the frozen standard-VAE checkpoint without retraining it.

The script keeps three image paths separate:

* posterior-mean reconstruction is deterministic and reads a real image;
* posterior-sample reconstruction also reads a real image;
* prior generation decodes an independent ``N(0, I)`` draw.

Latent interpolation is a local decoder diagnostic, not evidence of
independent semantic factors or unconditional generation.
"""

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights
from dl_utils.vae.vae import VAE, diagonal_gaussian_kl, reparameterize

PROJECT_ROOT = infer_project_root()
DEFAULT_CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "vae" / "vae.pth"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vae" / "evaluation"

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAXIMUM_BATCHES = 100
DEFAULT_ACTIVE_VARIANCE_THRESHOLD = 1e-2
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
NUM_COMPARISON_IMAGES = 8
NUM_PRIOR_SAMPLES = 18
PRIOR_GRID_COLUMNS = 6
NUM_INTERPOLATION_STEPS = 7
MINIMUM_PSNR_ERROR = 1e-12


class PosteriorAccumulator:
    """Accumulate standard-VAE posterior diagnostics without storing samples."""

    def __init__(self, z_dim):
        self.mu_total = torch.zeros(z_dim, dtype=torch.float64)
        self.mu_square_total = torch.zeros_like(self.mu_total)
        self.std_square_total = torch.zeros_like(self.mu_total)
        self.kl_total = torch.zeros_like(self.mu_total)
        self.examples = 0

    def update(self, mu, std):
        self.mu_total += mu.detach().double().sum(dim=0).cpu()
        self.mu_square_total += mu.detach().double().square().sum(dim=0).cpu()
        self.std_square_total += std.detach().double().square().sum(dim=0).cpu()
        self.kl_total += (
            diagonal_gaussian_kl(mu, std).detach().double().sum(dim=0).cpu()
        )
        self.examples += mu.shape[0]

    def metrics(self, *, active_variance_threshold):
        if self.examples == 0:
            raise ValueError("no posterior values were accumulated")
        posterior_mean = self.mu_total / self.examples
        variance_of_mu = (
            self.mu_square_total / self.examples - posterior_mean.square()
        ).clamp_min(0.0)
        posterior_variance = self.std_square_total / self.examples
        kl_per_dimension = self.kl_total / self.examples
        return {
            "examples": self.examples,
            "posterior_mean_by_dimension": posterior_mean.tolist(),
            "posterior_mean_std_by_dimension": variance_of_mu.sqrt().tolist(),
            "posterior_std_by_dimension": posterior_variance.sqrt().tolist(),
            "kl_nats_per_dimension": kl_per_dimension.tolist(),
            "kl_nats_per_image": float(kl_per_dimension.sum()),
            "active_variance_threshold": active_variance_threshold,
            "active_dimensions": int(
                (variance_of_mu > active_variance_threshold).sum()
            ),
        }


@torch.inference_mode()
def evaluate(
    model,
    loader,
    *,
    z_dim,
    maximum_batches,
    active_variance_threshold,
    device,
):
    """Evaluate posterior paths, prior samples, and a local interpolation."""
    if maximum_batches < 1:
        raise ValueError("maximum_batches must be positive")

    model.eval()
    accumulator = PosteriorAccumulator(z_dim)
    squared_error_total = 0.0
    absolute_error_total = 0.0
    evaluated_elements = 0
    comparison = None
    interpolation = None
    limited_loader = islice(loader, maximum_batches)
    progress = tqdm(
        limited_loader,
        total=min(len(loader), maximum_batches),
        desc="Analyze VAE",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        mu, std = model.encoder.statistics(images)
        mean_reconstructions = model.decoder(mu)
        sample_reconstructions = model.decoder(reparameterize(mu, std))
        squared_error_total += float((mean_reconstructions - images).square().sum())
        absolute_error_total += float((mean_reconstructions - images).abs().sum())
        evaluated_elements += images.numel()
        accumulator.update(mu, std)

        if comparison is None:
            comparison_count = min(NUM_COMPARISON_IMAGES, images.shape[0])
            comparison = torch.cat(
                (
                    images[:comparison_count],
                    mean_reconstructions[:comparison_count],
                    sample_reconstructions[:comparison_count],
                )
            ).cpu()
            if comparison_count >= 2:
                interpolation_weights = torch.linspace(
                    0,
                    1,
                    NUM_INTERPOLATION_STEPS,
                    device=device,
                )[:, None]
                latent_path = (1.0 - interpolation_weights) * mu[
                    0
                ] + interpolation_weights * mu[1]
                interpolation = model.decoder(latent_path).cpu()

    if evaluated_elements == 0 or comparison is None:
        raise ValueError("cannot analyze an empty loader")
    if interpolation is None:
        raise ValueError("at least two images are required for interpolation")

    mean_squared_error = squared_error_total / evaluated_elements
    metrics = {
        "posterior_mean_reconstruction": {
            "pixel_l1": absolute_error_total / evaluated_elements,
            "pixel_mse": mean_squared_error,
            "psnr_for_zero_to_one_range": -10.0
            * math.log10(max(mean_squared_error, MINIMUM_PSNR_ERROR)),
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
        torch.randn(NUM_PRIOR_SAMPLES, z_dim, device=device)
    ).cpu()
    return metrics, comparison, prior_samples, interpolation


def smoke_test(device):
    """Check the distinct reconstruction and prior paths on a tiny batch."""
    torch.manual_seed(7)
    z_dim = 2
    model = VAE(z_dim=z_dim).to(device).eval()
    images = torch.rand(2, 3, 256, 256, device=device)
    with torch.inference_mode():
        mu, std = model.encoder.statistics(images)
        mean_reconstructions = model.decoder(mu)
        sample_reconstructions = model.decoder(reparameterize(mu, std))
        prior_samples = model.decoder(torch.randn(2, z_dim, device=device))
    accumulator = PosteriorAccumulator(z_dim)
    accumulator.update(mu, std)
    metrics = accumulator.metrics(active_variance_threshold=1e-2)
    kl_nats_per_image = metrics["kl_nats_per_image"]
    if not isinstance(kl_nats_per_image, int | float):
        raise TypeError("KL metric must be numeric")
    assert mean_reconstructions.shape == sample_reconstructions.shape == images.shape
    assert prior_samples.shape == images.shape
    assert kl_nats_per_image >= 0
    print("smoke test passed: reconstruction and prior paths are distinct")


def analyze(args, device):
    """Load the final checkpoint and write all standard-VAE diagnostics."""
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {args.checkpoint}; run 1.0_vae.py first"
        )
    model, model_config = load_model_weights(
        args.checkpoint,
        VAE,
        device=device,
        expected_metadata={"model_name": "vae"},
    )
    z_dim = model_config.get("z_dim")
    if isinstance(z_dim, bool) or not isinstance(z_dim, int) or z_dim < 1:
        raise ValueError("checkpoint model_config has an invalid z_dim")

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
        z_dim=z_dim,
        maximum_batches=args.maximum_batches,
        active_variance_threshold=args.active_variance_threshold,
        device=device,
    )
    reset_dir(str(args.output))
    save_image(
        comparison,
        args.output / "real_mean_and_sample_reconstruction.png",
        nrow=comparison.shape[0] // 3,
    )
    save_image(
        prior_samples,
        args.output / "standard_normal_prior_samples.png",
        nrow=PRIOR_GRID_COLUMNS,
    )
    save_image(
        interpolation,
        args.output / "posterior_mean_interpolation_not_generation.png",
        nrow=len(interpolation),
    )
    with (args.output / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    console_metrics = {
        "posterior_mean_reconstruction": metrics["posterior_mean_reconstruction"],
        "posterior": {
            "examples": metrics["posterior"]["examples"],
            "kl_nats_per_image": metrics["posterior"]["kl_nats_per_image"],
            "active_variance_threshold": metrics["posterior"][
                "active_variance_threshold"
            ],
            "active_dimensions": metrics["posterior"]["active_dimensions"],
        },
        "path_boundaries": metrics["path_boundaries"],
    }
    print(json.dumps(console_metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--maximum-batches",
        type=int,
        default=DEFAULT_MAXIMUM_BATCHES,
    )
    parser.add_argument(
        "--active-variance-threshold",
        type=float,
        default=DEFAULT_ACTIVE_VARIANCE_THRESHOLD,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = try_gpu()
    if args.smoke_test:
        smoke_test(device)
    else:
        analyze(args, device)


if __name__ == "__main__":
    main()
