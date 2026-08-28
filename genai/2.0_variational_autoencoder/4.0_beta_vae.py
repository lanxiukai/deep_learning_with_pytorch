"""Beta-VAE: scan rate-distortion working points on known image factors.

The model and Bernoulli observation distribution stay fixed while beta scales
the complete per-sample Gaussian KL:

    loss = distortion + beta * rate.

The 32x32 procedural dataset independently varies shape, scale, rotation, and
two positions.  Defaults train several beta values and random seeds so the
later evaluation can separate method effects from seed variation.  A final
weight artifact keeps only model construction and comparison controls; this
short lesson has no checkpoint resumption machinery.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.factor_shapes import FactorShapes32, render_factor_shapes
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.vae.inference import GaussianVAE32, model_config
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)

PROJECT_ROOT = infer_project_root()


def parse_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(dict.fromkeys(float(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("beta values must be non-negative")
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


def beta_vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return D + beta R while keeping the unweighted terms observable."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    distortion = F.binary_cross_entropy(
        reconstruction, target, reduction="none"
    ).flatten(1).sum(dim=1).mean()
    per_dimension_rate = diagonal_gaussian_kl_from_logvar(mu, logvar)
    rate = per_dimension_rate.sum(dim=1).mean()
    return distortion + float(beta) * rate, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "weighted_rate": (float(beta) * rate).detach(),
        "active_kl_dimensions": (
            per_dimension_rate.mean(dim=0) > 0.05
        ).sum().detach(),
    }


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    train_set = FactorShapes32(
        split="train", split_seed=args.split_seed
    )
    test_set = FactorShapes32(
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
        DataLoader(test_set, shuffle=False, drop_last=False, **common),
    )


@torch.inference_mode()
def evaluate(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    distortion = 0.0
    rate = 0.0
    examples = 0
    means = []
    per_dimension_rates = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        reconstruction = model.decode(mu)
        distortion += float(
            F.binary_cross_entropy(reconstruction, x, reduction="sum")
        )
        batch_rates = diagonal_gaussian_kl_from_logvar(mu, logvar)
        rate += float(batch_rates.sum())
        means.append(mu.cpu())
        per_dimension_rates.append(batch_rates.cpu())
        examples += x.shape[0]
    all_means = torch.cat(means)
    mean_rates = torch.cat(per_dimension_rates).mean(dim=0)
    distortion /= examples
    rate /= examples
    return {
        "distortion": distortion,
        "rate": rate,
        "objective": distortion + beta * rate,
        "active_units": float(
            (all_means.var(dim=0, unbiased=False) > 1e-2).sum()
        ),
        "active_kl_dimensions": float((mean_rates > 0.05).sum()),
    }


def train_one(
    args: argparse.Namespace,
    *,
    beta: float,
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
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
            mu, logvar = model.encode(x)
            z = reparameterize_logvar(mu, logvar)
            reconstruction = model.decode(z)
            loss, terms = beta_vae_loss(
                reconstruction, x, mu, logvar, beta=beta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += torch.stack(
                [
                    loss.detach(),
                    terms["distortion"],
                    terms["rate"],
                    terms["weighted_rate"],
                ]
            ) * x.shape[0]
            examples += x.shape[0]
        means = (totals / examples).tolist()
        print(
            f"beta={beta:g}, seed={seed}, epoch {epoch:03d}: "
            f"loss={means[0]:.3f}, D={means[1]:.3f}, "
            f"R={means[2]:.3f}, beta*R={means[3]:.3f}"
        )

    validation = evaluate(
        model, test_loader, beta=beta, device=device
    )
    out_dir = (
        PROJECT_ROOT
        / "output"
        / "vae"
        / "beta_vae"
        / f"beta_{number_slug(beta)}"
        / f"seed_{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model_weights(
        model,
        out_dir / "model.pth",
        metadata={
            "model_name": "beta_vae",
            "model_config": model_config(model),
            "beta": beta,
            "seed": seed,
            "split_seed": args.split_seed,
        },
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    print(
        f"beta={beta:g}, seed={seed}: validation "
        f"D={validation['distortion']:.3f}, R={validation['rate']:.3f}; "
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
    mu, logvar = model.encode(x)
    reconstruction = model.decode(reparameterize_logvar(mu, logvar))
    loss, terms = beta_vae_loss(
        reconstruction, x, mu, logvar, beta=4.0
    )
    loss.backward()
    assert x.shape == (4, 1, 32, 32)
    assert model.base_posterior.weight.grad is not None
    assert all(torch.isfinite(value) for value in terms.values())
    print(
        "smoke test passed: "
        f"D={terms['distortion'].item():.3f}, "
        f"R={terms['rate'].item():.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--betas", type=parse_floats, default=(1.0, 4.0, 10.0)
    )
    parser.add_argument(
        "--seeds", type=parse_ints, default=(11, 22, 33)
    )
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
    train_loader, test_loader = make_loaders(args, device)
    for beta in args.betas:
        for seed in args.seeds:
            train_one(
                args,
                beta=beta,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
            )


if __name__ == "__main__":
    main()
