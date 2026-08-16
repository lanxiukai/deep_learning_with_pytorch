"""FactorVAE: estimate total correlation with a latent discriminator.

This is the adversarial counterpart to ``1.4_beta_tc_vae.py``.  A discriminator
learns to distinguish samples from the aggregated posterior ``q(z)`` from
samples whose dimensions were independently permuted across the minibatch,
an approximation to ``product_j q(z_j)``.  Its logit ratio supplies the TC
penalty while the image model keeps the ordinary VAE objective.

The discriminator estimate is only as good as its capacity and optimization;
FactorVAE does not escape the identifiability limits of unsupervised
disentanglement.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae_common import (
    ConvGaussianVAE28,
    diagonal_gaussian_kl_from_logvar,
)


PROJECT_ROOT = infer_project_root()


class TotalCorrelationDiscriminator(nn.Module):
    """Class 0 is joint q(z); class 1 is dimension-wise permuted q(z)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(4):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2)])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def permute_latent_dimensions(z: Tensor) -> Tensor:
    """Independently shuffle every latent coordinate across the batch."""
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("expected a [B, D] latent batch with B >= 2")
    columns = [z[torch.randperm(z.shape[0], device=z.device), dim] for dim in range(z.shape[1])]
    return torch.stack(columns, dim=1)


def factor_vae_step(
    model: ConvGaussianVAE28,
    discriminator: TotalCorrelationDiscriminator,
    x: Tensor,
    vae_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    *,
    gamma: float,
) -> dict[str, Tensor]:
    """Perform the FactorVAE model step followed by its density-ratio step."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    discriminator.requires_grad_(False)
    reconstruction, mu, logvar, z = model(x)
    distortion = F.binary_cross_entropy(reconstruction, x, reduction="none")
    distortion = distortion.flatten(1).sum(dim=1).mean()
    rate = diagonal_gaussian_kl_from_logvar(mu, logvar).sum(dim=1).mean()
    joint_logits = discriminator(z)
    tc = (joint_logits[:, 0] - joint_logits[:, 1]).mean()
    vae_loss = distortion + rate + float(gamma) * tc
    vae_optimizer.zero_grad(set_to_none=True)
    vae_loss.backward()
    vae_optimizer.step()

    discriminator.requires_grad_(True)
    # Re-encode after the VAE update.  The detach is the explicit boundary:
    # the discriminator learns a density ratio but cannot move the encoder.
    with torch.no_grad():
        _, _, _, joint_z = model(x)
        product_z = permute_latent_dimensions(joint_z)
    joint_logits = discriminator(joint_z.detach())
    product_logits = discriminator(product_z.detach())
    joint_targets = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    product_targets = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
    discriminator_loss = 0.5 * (
        F.cross_entropy(joint_logits, joint_targets)
        + F.cross_entropy(product_logits, product_targets)
    )
    discriminator_optimizer.zero_grad(set_to_none=True)
    discriminator_loss.backward()
    discriminator_optimizer.step()
    return {
        "vae_loss": vae_loss.detach(),
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "tc_ratio": tc.detach(),
        "discriminator": discriminator_loss.detach(),
    }


def smoke_test() -> None:
    torch.manual_seed(7)
    model = ConvGaussianVAE28(latent_dim=6)
    discriminator = TotalCorrelationDiscriminator(6, hidden_dim=32)
    vae_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    metrics = factor_vae_step(
        model,
        discriminator,
        torch.rand(8, 1, 28, 28),
        vae_optimizer,
        discriminator_optimizer,
        gamma=6.0,
    )
    assert model.posterior.weight.grad is not None
    assert discriminator.net[-1].weight.grad is not None
    assert all(torch.isfinite(value) for value in metrics.values())
    print(
        "smoke test passed: "
        + ", ".join(f"{name}={value.item():.3f}" for name, value in metrics.items())
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "factor_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = ConvGaussianVAE28(args.latent_dim).to(device)
    discriminator = TotalCorrelationDiscriminator(args.latent_dim).to(device)
    vae_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9)
    )
    fixed_z = torch.randn(64, args.latent_dim, device=device)

    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(5, device=device)
        examples = 0
        model.train()
        discriminator.train()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            metrics = factor_vae_step(
                model,
                discriminator,
                x,
                vae_optimizer,
                discriminator_optimizer,
                gamma=args.gamma,
            )
            sums += torch.stack(list(metrics.values())) * x.shape[0]
            examples += x.shape[0]
        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: VAE={means[0]:.3f}, distortion={means[1]:.3f}, "
            f"R={means[2]:.3f}, TC_ratio={means[3]:.3f}, D_loss={means[4]:.3f}"
        )
        model.eval()
        with torch.inference_mode():
            samples = model.decode(fixed_z)
        save_image(samples, out_dir / f"epoch_{epoch:03d}.png", nrow=8)

    torch.save(
        {
            "format_version": 1,
            "model_name": "factor_vae",
            "vae": model.state_dict(),
            "tc_discriminator": discriminator.state_dict(),
            "model_config": {"latent_dim": args.latent_dim},
            "gamma": args.gamma,
        },
        out_dir / "factor_vae.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
