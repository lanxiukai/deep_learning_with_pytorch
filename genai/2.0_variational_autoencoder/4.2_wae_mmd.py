"""WAE-MMD: match the aggregate posterior instead of every posterior.

This focused branch keeps the same 32x32 Gaussian encoder/decoder and
Bernoulli reconstruction used by the rate-control lessons, but removes the
per-example KL from the training objective:

    loss = distortion + lambda_mmd * MMD^2(q(z), N(0, I)).

The distinction is the lesson.  MMD compares one minibatch of aggregate
encoded samples with prior samples; it is not a calibrated likelihood, a
Wasserstein-distance estimate, or a replacement name for VAE rate.  The
script therefore compares WAE runs at matched reconstruction, capacity, and
training budget in
``4.3_representation_evaluation.py`` instead of placing them on the
beta-VAE matched-rate axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.factor_shapes import (
    FACTOR_NAMES,
    FACTOR_SIZES,
    FactorShapes32,
    render_factor_shapes,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import reproducibility_metadata
from dl_utils.vae.aggregate_matching import (
    imq_mmd2,
    latent_moment_diagnostics,
)
from dl_utils.vae.inference import GaussianVAE32, model_config
from dl_utils.vae.vae_common import reparameterize_logvar


PROJECT_ROOT = infer_project_root()


def parse_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(dict.fromkeys(float(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def number_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def wae_mmd_loss(
    reconstruction: Tensor,
    target: Tensor,
    encoded: Tensor,
    prior: Tensor,
    *,
    mmd_weight: float,
    kernel_scales: tuple[float, ...],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return reconstruction plus a biased, minibatch aggregate MMD."""
    if mmd_weight <= 0.0:
        raise ValueError("mmd_weight must be positive")
    distortion = F.binary_cross_entropy(
        reconstruction, target, reduction="none"
    ).flatten(1).sum(dim=1).mean()
    mmd2 = imq_mmd2(
        encoded, prior, scales=kernel_scales, biased=True
    )
    return distortion + float(mmd_weight) * mmd2, {
        "distortion": distortion.detach(),
        "mmd2": mmd2.detach(),
        "weighted_mmd": (float(mmd_weight) * mmd2).detach(),
    }


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    train_set = FactorShapes32(
        split="train", split_seed=args.split_seed
    )
    validation_set = FactorShapes32(
        split="test", split_seed=args.split_seed
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(
            validation_set, shuffle=False, drop_last=False, **common
        ),
    )


@torch.inference_mode()
def evaluate(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    kernel_scales: tuple[float, ...],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    distortion = 0.0
    examples = 0
    posterior_samples = []
    posterior_means = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar, _ = model.encode(x)
        encoded = reparameterize_logvar(mu, logvar)
        reconstruction = model.decode(mu)
        distortion += float(
            F.binary_cross_entropy(reconstruction, x, reduction="sum")
        )
        posterior_samples.append(encoded.cpu())
        posterior_means.append(mu.cpu())
        examples += x.shape[0]

    encoded = torch.cat(posterior_samples)
    means = torch.cat(posterior_means)
    generator = torch.Generator().manual_seed(17)
    prior = torch.randn(
        encoded.shape, generator=generator, dtype=encoded.dtype
    )
    moments = latent_moment_diagnostics(encoded)
    mean_variance = means.var(dim=0, unbiased=False)
    return {
        "distortion": distortion / examples,
        "aggregate_mmd2_biased": float(
            imq_mmd2(encoded, prior, scales=kernel_scales, biased=True)
        ),
        "aggregate_mmd2_unbiased": float(
            imq_mmd2(encoded, prior, scales=kernel_scales, biased=False)
        ),
        "latent_mean_norm": float(moments["latent_mean_norm"]),
        "latent_variance_mean": float(moments["latent_variance_mean"]),
        "latent_variance_error": float(
            moments["latent_variance_error"]
        ),
        "active_units": float((mean_variance > 1e-2).sum()),
        "posterior_mean_variance": float(mean_variance.mean()),
    }


def train_one(
    args: argparse.Namespace,
    *,
    mmd_weight: float,
    seed: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    set_seed(seed)
    model = GaussianVAE32(
        latent_dim=args.latent_dim,
        hidden_channels=args.hidden_channels,
        context_dim=args.context_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = torch.zeros(4, device=device)
        examples = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            mu, logvar, _ = model.encode(x)
            encoded = reparameterize_logvar(mu, logvar)
            reconstruction = model.decode(encoded)
            prior = torch.randn_like(encoded)
            loss, terms = wae_mmd_loss(
                reconstruction,
                x,
                encoded,
                prior,
                mmd_weight=mmd_weight,
                kernel_scales=args.kernel_scales,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += torch.stack(
                (
                    loss.detach(),
                    terms["distortion"],
                    terms["mmd2"],
                    terms["weighted_mmd"],
                )
            ) * x.shape[0]
            examples += x.shape[0]
        values = (totals / examples).tolist()
        print(
            f"lambda={mmd_weight:g}, seed={seed}, epoch {epoch:03d}: "
            f"loss={values[0]:.3f}, D={values[1]:.3f}, "
            f"MMD^2={values[2]:.5f}, "
            f"lambda*MMD^2={values[3]:.3f}"
        )

    validation = evaluate(
        model,
        validation_loader,
        kernel_scales=args.kernel_scales,
        device=device,
    )
    out_dir = (
        PROJECT_ROOT
        / "output"
        / "wae_mmd"
        / f"lambda_{number_slug(mmd_weight)}"
        / f"seed_{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": "wae_mmd",
            "roadmap_role": "focused_branch",
            "roadmap_step": "3X",
            "direct_baseline": "standard VAE encoder/decoder backbone",
            "visible_increment": (
                "replace per-sample KL with aggregate-posterior IMQ-MMD"
            ),
            "state_dict": model.state_dict(),
            "model_config": model_config(model),
            "mmd_weight": mmd_weight,
            "kernel": {
                "name": "multiscale inverse multiquadratic",
                "scales": args.kernel_scales,
                "training_estimator": "biased minibatch V-statistic",
            },
            "seed": seed,
            "dataset": "FactorShapes32",
            "factor_names": FACTOR_NAMES,
            "factor_sizes": FACTOR_SIZES,
            "split_seed": args.split_seed,
            "image_size": 32,
            "observation": "independent Bernoulli mean",
            "loss_reduction": "sum pixels per sample, then mean batch",
            "rate_semantics": (
                "no per-sample KL is optimized or reported as WAE rate"
            ),
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"wae_mmd": model},
                seed=seed,
                training_budget={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.epochs * len(train_loader),
                },
                data_preprocessing={
                    "renderer": "FactorShapes32 deterministic renderer",
                    "value_range": [0.0, 1.0],
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "model.pth",
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
        comparison_batch = next(iter(validation_loader))[0][:8].to(device)
        comparison = torch.cat(
            (
                comparison_batch,
                model.reconstruct(comparison_batch),
                samples[:8],
            )
        )
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    save_image(
        comparison,
        out_dir / "real_reconstruction_prior.png",
        nrow=8,
    )
    print(
        f"lambda={mmd_weight:g}, seed={seed}: validation "
        f"D={validation['distortion']:.3f}, "
        f"MMD^2={validation['aggregate_mmd2_biased']:.5f}; "
        f"saved {out_dir}"
    )


def smoke_test() -> None:
    set_seed(7)
    factors = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 2, 3, 4],
            [2, 3, 5, 6, 6],
            [0, 2, 3, 1, 5],
        ]
    )
    x = render_factor_shapes(factors)
    model = GaussianVAE32(
        latent_dim=6, hidden_channels=32, context_dim=16
    )
    mu, logvar, _ = model.encode(x)
    encoded = reparameterize_logvar(mu, logvar)
    prior = torch.randn_like(encoded)
    reconstruction = model.decode(encoded)
    loss, terms = wae_mmd_loss(
        reconstruction,
        x,
        encoded,
        prior,
        mmd_weight=10.0,
        kernel_scales=(0.5, 1.0, 2.0),
    )
    loss.backward()
    identical = imq_mmd2(encoded.detach(), encoded.detach())
    shifted = imq_mmd2(encoded.detach() + 5.0, encoded.detach())
    unbiased = imq_mmd2(
        encoded.detach(), prior.detach(), biased=False
    )
    assert model.base_posterior.weight.grad is not None
    assert abs(float(identical)) < 1e-6
    assert shifted > identical
    assert torch.isfinite(unbiased)
    assert all(torch.isfinite(value) for value in terms.values())
    print(
        "smoke test passed: "
        f"D={terms['distortion'].item():.3f}, "
        f"MMD^2={terms['mmd2'].item():.5f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--mmd-weights", type=parse_floats, default=(10.0, 100.0)
    )
    parser.add_argument(
        "--kernel-scales",
        type=parse_floats,
        default=(0.5, 1.0, 2.0),
    )
    parser.add_argument("--seeds", type=parse_ints, default=(11, 22, 33))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(args, device)
    for mmd_weight in args.mmd_weights:
        for seed in args.seeds:
            train_one(
                args,
                mmd_weight=mmd_weight,
                seed=seed,
                train_loader=train_loader,
                validation_loader=validation_loader,
                device=device,
            )


if __name__ == "__main__":
    main()
