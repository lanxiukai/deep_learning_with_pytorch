"""Conditional VAE: make the training and generation information paths explicit.

The main standard-prior model factorizes

    p(x, z | c) = p(z) p(x | z, c)

and trains the recognition posterior q(z | x, c) with

    distortion + KL[q(z | x, c) || p(z)].

The target image enters q during training, but generation calls only
`model.generate(labels)`, which cannot observe the target image.  A learned
conditional prior p(z | c) is a separate ablation after the posterior and
decoder condition interfaces work.  A deterministic condition-to-image
baseline is also separate, not a mislabelled zero-KL VAE.

Examples:
    python genai/2.0_variational_autoencoder/2.0_conditional_vae.py
    python genai/2.0_variational_autoencoder/2.0_conditional_vae.py \
        --variant all
    python genai/2.0_variational_autoencoder/2.0_conditional_vae.py \
        --smoke-test
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import reproducibility_metadata
from dl_utils.vae.conditional import (
    ConditionalMeanDecoder32,
    ConditionalVAE32,
)
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar


PROJECT_ROOT = infer_project_root()
VARIANTS = ("standard-prior", "learned-prior", "deterministic")


def conditional_vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    statistics: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return negative conditional ELBO with explicit per-sample reductions."""
    distortion = F.binary_cross_entropy(
        reconstruction, target, reduction="none"
    ).flatten(1).sum(dim=1).mean()
    rate_per_dimension = diagonal_gaussian_kl_from_logvar(
        statistics["q_mu"],
        statistics["q_logvar"],
        statistics["p_mu"],
        statistics["p_logvar"],
    )
    rate = rate_per_dimension.sum(dim=1).mean()
    return distortion + rate, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "active_units": (
            rate_per_dimension.mean(dim=0) > 0.05
        ).sum().detach(),
    }


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


def _accumulate(
    totals: dict[str, float], metrics: dict[str, Tensor], batch_size: int
) -> None:
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + float(value) * batch_size


@torch.inference_mode()
def evaluate_cvae(
    model: ConditionalVAE32,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction, statistics = model(x, labels)
        loss, terms = conditional_vae_loss(reconstruction, x, statistics)
        _accumulate(totals, {"loss": loss.detach(), **terms}, x.shape[0])
        examples += x.shape[0]
    return {name: value / examples for name, value in totals.items()}


@torch.inference_mode()
def evaluate_deterministic(
    model: ConditionalMeanDecoder32,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    distortion = 0.0
    examples = 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction = model(labels)
        batch_loss = F.binary_cross_entropy(
            reconstruction, x, reduction="none"
        ).flatten(1).sum(dim=1)
        distortion += float(batch_loss.sum())
        examples += x.shape[0]
    return {"distortion": distortion / examples}


def _save_conditional_samples(
    model: nn.Module,
    path: Path,
    *,
    device: torch.device,
    samples_per_class: int = 8,
) -> None:
    labels = torch.arange(10, device=device).repeat_interleave(
        samples_per_class
    )
    model.eval()
    with torch.inference_mode():
        if isinstance(model, ConditionalVAE32):
            images = model.generate(labels)
        elif isinstance(model, ConditionalMeanDecoder32):
            images = model.generate(labels)
        else:
            raise TypeError("unsupported conditional model")
    save_image(images, path, nrow=samples_per_class)


def train_cvae(
    args: argparse.Namespace,
    *,
    learned_prior: bool,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    variant = "learned-prior" if learned_prior else "standard-prior"
    out_dir = PROJECT_ROOT / "output" / "vae" / "conditional_vae" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ConditionalVAE32(
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        hidden_channels=args.hidden_channels,
        learned_prior=learned_prior,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            reconstruction, statistics = model(x, labels)
            loss, terms = conditional_vae_loss(reconstruction, x, statistics)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _accumulate(
                totals,
                {"loss": loss.detach(), **terms},
                x.shape[0],
            )
            examples += x.shape[0]

        train_metrics = {
            name: value / examples for name, value in totals.items()
        }
        validation_metrics = evaluate_cvae(
            model, validation_loader, device
        )
        print(
            f"{variant} epoch {epoch:03d}: "
            f"train loss={train_metrics['loss']:.3f}, "
            f"D={train_metrics['distortion']:.3f}, "
            f"R={train_metrics['rate']:.3f}; "
            f"validation loss={validation_metrics['loss']:.3f}, "
            f"D={validation_metrics['distortion']:.3f}, "
            f"R={validation_metrics['rate']:.3f}"
        )

    checkpoint = {
        "format_version": 2,
        "model_name": "conditional_vae",
        "roadmap_role": "historical_anchor",
        "roadmap_step": 1,
        "direct_baseline": "standard VAE",
        "visible_increment": (
            "conditional-prior ablation after conditioning the posterior "
            "and decoder"
            if learned_prior
            else "condition the posterior and decoder"
        ),
        "conditional_prior_ablation": learned_prior,
        "variant": variant,
        "state_dict": model.state_dict(),
        "model_config": {
            "num_classes": 10,
            "latent_dim": args.latent_dim,
            "condition_dim": args.condition_dim,
            "hidden_channels": args.hidden_channels,
            "learned_prior": learned_prior,
        },
        "dataset": "MNIST",
        "image_size": 32,
        "observation": "independent Bernoulli mean",
        "loss_reduction": "sum pixels per sample, then mean batch",
        "validation_metrics": validation_metrics,
        **reproducibility_metadata(
            models={"conditional_vae": model},
            seed=args.seed,
            training_budget={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer_updates": args.epochs * len(train_loader),
            },
            data_preprocessing={
                "resize": [32, 32],
                "value_range": [0.0, 1.0],
                "augmentation": "none",
            },
        ),
    }
    torch.save(checkpoint, out_dir / "model.pth")
    _save_conditional_samples(
        model, out_dir / "conditional_samples.png", device=device
    )


def train_deterministic(
    args: argparse.Namespace,
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    out_dir = PROJECT_ROOT / "output" / "vae" / "conditional_vae" / "deterministic"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ConditionalMeanDecoder32(
        condition_dim=args.condition_dim,
        hidden_channels=args.hidden_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        distortion = 0.0
        examples = 0
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            reconstruction = model(labels)
            loss = F.binary_cross_entropy(
                reconstruction, x, reduction="none"
            ).flatten(1).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            distortion += float(loss.detach()) * x.shape[0]
            examples += x.shape[0]
        validation = evaluate_deterministic(
            model, validation_loader, device
        )
        print(
            f"deterministic epoch {epoch:03d}: "
            f"train D={distortion / examples:.3f}; "
            f"validation D={validation['distortion']:.3f}"
        )

    torch.save(
        {
            "format_version": 2,
            "model_name": "conditional_mean_decoder",
            "roadmap_role": "controlled_baseline",
            "direct_baseline": "deterministic conditional decoder",
            "variant": "deterministic",
            "state_dict": model.state_dict(),
            "model_config": {
                "num_classes": 10,
                "condition_dim": args.condition_dim,
                "hidden_channels": args.hidden_channels,
            },
            "dataset": "MNIST",
            "image_size": 32,
            "observation": "independent Bernoulli mean",
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"conditional_mean_decoder": model},
                seed=args.seed,
                training_budget={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.epochs * len(train_loader),
                },
                data_preprocessing={
                    "resize": [32, 32],
                    "value_range": [0.0, 1.0],
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "model.pth",
    )
    _save_conditional_samples(
        model, out_dir / "conditional_samples.png", device=device
    )


def selected_variants(name: str) -> Iterable[str]:
    return VARIANTS if name == "all" else (name,)


def smoke_test() -> None:
    set_seed(7)
    x = torch.rand(6, 1, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    model = ConditionalVAE32(
        latent_dim=6,
        condition_dim=8,
        hidden_channels=32,
        learned_prior=True,
    )
    reconstruction, statistics = model(x, labels)
    loss, terms = conditional_vae_loss(reconstruction, x, statistics)
    loss.backward()
    assert reconstruction.shape == x.shape
    assert model.prior_network is not None
    assert model.prior_network[-1].weight.grad is not None
    assert model.posterior[-1].weight.grad is not None
    assert model.generate(labels[:3]).shape == (3, 1, 32, 32)

    fixed = ConditionalVAE32(
        latent_dim=6,
        condition_dim=8,
        hidden_channels=32,
        learned_prior=False,
    )
    p_mu, p_logvar = fixed.prior(labels)
    assert torch.count_nonzero(p_mu) == 0
    assert torch.count_nonzero(p_logvar) == 0

    deterministic = ConditionalMeanDecoder32(
        condition_dim=8, hidden_channels=32
    )
    deterministic_loss = F.binary_cross_entropy(
        deterministic(labels), x
    )
    deterministic_loss.backward()
    assert deterministic.decoder.net[-2].weight.grad is not None
    assert all(torch.isfinite(value) for value in terms.values())
    print(
        "smoke test passed: "
        f"loss={loss.item():.3f}, D={terms['distortion'].item():.3f}, "
        f"R={terms['rate'].item():.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, "all"),
        default="standard-prior",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--condition-dim", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    for variant in selected_variants(args.variant):
        # Keep common initializations independent of loop order.
        set_seed(args.seed)
        if variant == "deterministic":
            train_deterministic(
                args,
                train_loader=train_loader,
                validation_loader=validation_loader,
                device=device,
            )
        else:
            train_cvae(
                args,
                learned_prior=variant == "learned-prior",
                train_loader=train_loader,
                validation_loader=validation_loader,
                device=device,
            )


if __name__ == "__main__":
    main()
