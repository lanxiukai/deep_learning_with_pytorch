"""Importance-Weighted Autoencoder: tighten the bound without changing networks.

For each image, K samples from the same diagonal-Gaussian posterior form

    log_mean_exp(log p(x | z_k) + log p(z_k) - log q(z_k | x)).

The particle reduction happens per image and entirely in log space.  K=1 is
exactly the Monte Carlo ELBO.  Larger K changes the objective and compute
budget, not the VAE model family.

By default every K receives the same number of optimizer steps.  Add
`--match-particle-budget` to divide the update count by K, making the total
number of decoded particles approximately equal instead.

Each run saves only final model weights and constructor metadata. These short
runs intentionally have no checkpoint resumption machinery.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.vae.inference import (
    GaussianVAE32,
    importance_diagnostics,
    importance_log_weights,
    log_mean_exp,
    model_config,
)

PROJECT_ROOT = infer_project_root()


def parse_particles(text: str) -> tuple[int, ...]:
    try:
        particles = tuple(dict.fromkeys(int(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "particles must be comma-separated integers"
        ) from error
    if not particles or any(value < 1 for value in particles):
        raise argparse.ArgumentTypeError("every particle count must be positive")
    return particles


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    train_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transform,
    )
    validation_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=False,
        download=True,
        transform=transform,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(validation_set, shuffle=False, drop_last=False, **common),
    )


@torch.inference_mode()
def evaluate_bound(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    particles: int,
    particle_chunk_size: int,
    max_examples: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for x, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        log_weights, terms = importance_log_weights(
            model,
            x,
            particles=particles,
            particle_chunk_size=particle_chunk_size,
        )
        diagnostics = importance_diagnostics(log_weights, terms)
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * x.shape[0]
        examples += x.shape[0]
    return {name: value / examples for name, value in totals.items()}


def _next_batch(
    iterator: object, loader: DataLoader
) -> tuple[object, tuple[Tensor, Tensor]]:
    try:
        batch = next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return iterator, batch


def train_one(
    args: argparse.Namespace,
    *,
    particles: int,
    minimum_particles: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    set_seed(args.seed)
    model = GaussianVAE32(
        latent_dim=args.latent_dim,
        hidden_channels=args.hidden_channels,
        context_dim=args.context_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    base_updates = args.epochs * len(train_loader)
    updates = (
        max(1, base_updates * minimum_particles // particles)
        if args.match_particle_budget
        else base_updates
    )
    run_name = (
        f"matched_k{particles}"
        if args.match_particle_budget
        else f"k{particles}"
    )
    out_dir = PROJECT_ROOT / "output" / "vae" / "iwae" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    iterator = iter(train_loader)
    report_interval = max(1, len(train_loader))
    running: dict[str, float] = {}
    running_examples = 0

    model.train()
    for update in range(1, updates + 1):
        iterator, (x, _) = _next_batch(iterator, train_loader)
        x = x.to(device, non_blocking=True)
        log_weights, terms = importance_log_weights(
            model, x, particles=particles
        )
        loss = -log_mean_exp(log_weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        diagnostics = importance_diagnostics(log_weights, terms)
        diagnostics["loss"] = loss.detach()
        for name, value in diagnostics.items():
            running[name] = running.get(name, 0.0) + float(value) * x.shape[0]
        running_examples += x.shape[0]
        if update % report_interval == 0 or update == updates:
            means = {
                name: value / running_examples
                for name, value in running.items()
            }
            print(
                f"K={particles} update {update:05d}/{updates:05d}: "
                f"loss={means['loss']:.3f}, "
                f"bound={means['bound']:.3f}, "
                f"ESS/K={means['ess_fraction']:.3f}, "
                f"log-w range={means['log_weight_range']:.3f}"
            )
            running.clear()
            running_examples = 0

    validation = evaluate_bound(
        model,
        validation_loader,
        particles=args.validation_particles,
        particle_chunk_size=args.particle_chunk_size,
        max_examples=args.validation_examples,
        device=device,
    )
    save_model_weights(
        model,
        out_dir / "model.pth",
        metadata={
            "model_name": "iwae",
            "model_config": model_config(model),
        },
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    print(
        f"K={particles}: validation bound={validation['bound']:.3f}, "
        f"ESS/K={validation['ess_fraction']:.3f}; saved {out_dir}"
    )


def smoke_test() -> None:
    set_seed(7)
    model = GaussianVAE32(
        latent_dim=5, hidden_channels=32, context_dim=16
    )
    x = torch.rand(4, 1, 32, 32)
    log_weights_1, terms_1 = importance_log_weights(
        model, x, particles=1
    )
    bound_1 = log_mean_exp(log_weights_1)
    assert torch.allclose(bound_1, log_weights_1[:, 0])
    expected = (
        terms_1["log_px"] + terms_1["log_pz"] - terms_1["log_q"]
    )
    assert torch.allclose(log_weights_1, expected)

    log_weights_5, terms_5 = importance_log_weights(
        model, x, particles=5, particle_chunk_size=2
    )
    loss = -log_mean_exp(log_weights_5).mean()
    loss.backward()
    diagnostics = importance_diagnostics(log_weights_5, terms_5)
    assert log_weights_5.shape == (4, 5)
    assert model.base_posterior.weight.grad is not None
    assert model.decoder[-2].weight.grad is not None
    assert all(torch.isfinite(value) for value in diagnostics.values())
    print(
        "smoke test passed: K=1 identity and chunked K=5 "
        f"bound={diagnostics['bound'].item():.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--particles", type=parse_particles, default=(1, 5)
    )
    parser.add_argument("--match-particle-budget", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-particles", type=int, default=64)
    parser.add_argument("--particle-chunk-size", type=int, default=8)
    parser.add_argument("--validation-examples", type=int, default=2_048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(args, device)
    minimum_particles = min(args.particles)
    for particles in args.particles:
        train_one(
            args,
            particles=particles,
            minimum_particles=minimum_particles,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
        )


if __name__ == "__main__":
    main()
