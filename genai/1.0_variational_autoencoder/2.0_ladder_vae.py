"""
Two-level Ladder VAE with precision-weighted posterior fusion.

The generative path is

    z2 ~ N(0, I)
    z1 ~ p(z1 | z2)
    x  ~ p(x | z1)

The inference path first extracts bottom-up Gaussian evidence from x.  During
the top-down pass, each approximate posterior is a product of two Gaussians:
the generative prior and the bottom-up evidence.  The product is implemented
by adding precisions, which is the central Ladder VAE mechanism.

This is a deliberately small MNIST implementation.  It also exposes KL
warm-up and free bits so layer-wise posterior collapse can be inspected.

Run:
    python genai/1.0_variational_autoencoder/2.0_ladder_vae.py --smoke-test
    python genai/1.0_variational_autoencoder/2.0_ladder_vae.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fuse_gaussians(
    prior_mu: Tensor,
    prior_logvar: Tensor,
    evidence_mu: Tensor,
    evidence_logvar: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return parameters of N_prior * N_evidence after normalization."""
    prior_precision = torch.exp(-prior_logvar)
    evidence_precision = torch.exp(-evidence_logvar)
    fused_precision = prior_precision + evidence_precision
    fused_var = fused_precision.reciprocal()
    fused_mu = fused_var * (
        prior_precision * prior_mu + evidence_precision * evidence_mu
    )
    return fused_mu, torch.log(fused_var)


def gaussian_kl(
    q_mu: Tensor, q_logvar: Tensor, p_mu: Tensor, p_logvar: Tensor
) -> Tensor:
    """KL(q || p), summed over latent dimensions and kept per sample."""
    return 0.5 * (
        p_logvar
        - q_logvar
        + torch.exp(q_logvar - p_logvar)
        + (q_mu - p_mu).square() * torch.exp(-p_logvar)
        - 1.0
    ).sum(dim=1)


class LadderVAE(nn.Module):
    def __init__(self, z1_dim: int = 32, z2_dim: int = 16) -> None:
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim

        self.bottom_up_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ELU(inplace=True),
        )
        self.bottom_up_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
        )
        self.z1_evidence = nn.Linear(512, 2 * z1_dim)
        self.z2_evidence = nn.Linear(256, 2 * z2_dim)

        self.z1_prior = nn.Sequential(
            nn.Linear(z2_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 2 * z1_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z1_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),
        )

    @staticmethod
    def _split_gaussian_parameters(raw: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = raw.chunk(2, dim=1)
        return mu, logvar.clamp(-12.0, 12.0)

    @staticmethod
    def _sample(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def infer(self, x: Tensor) -> dict[str, Tensor]:
        h1 = self.bottom_up_1(x)
        h2 = self.bottom_up_2(h1)
        evidence1_mu, evidence1_logvar = self._split_gaussian_parameters(
            self.z1_evidence(h1)
        )
        evidence2_mu, evidence2_logvar = self._split_gaussian_parameters(
            self.z2_evidence(h2)
        )

        prior2_mu = torch.zeros_like(evidence2_mu)
        prior2_logvar = torch.zeros_like(evidence2_logvar)
        q2_mu, q2_logvar = fuse_gaussians(
            prior2_mu, prior2_logvar, evidence2_mu, evidence2_logvar
        )
        z2 = self._sample(q2_mu, q2_logvar)

        prior1_mu, prior1_logvar = self._split_gaussian_parameters(
            self.z1_prior(z2)
        )
        q1_mu, q1_logvar = fuse_gaussians(
            prior1_mu, prior1_logvar, evidence1_mu, evidence1_logvar
        )
        z1 = self._sample(q1_mu, q1_logvar)
        return {
            "z1": z1,
            "z2": z2,
            "q1_mu": q1_mu,
            "q1_logvar": q1_logvar,
            "p1_mu": prior1_mu,
            "p1_logvar": prior1_logvar,
            "q2_mu": q2_mu,
            "q2_logvar": q2_logvar,
            "p2_mu": prior2_mu,
            "p2_logvar": prior2_logvar,
        }

    def decode(self, z1: Tensor) -> Tensor:
        return self.decoder(z1).reshape(-1, 1, 28, 28)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        latents = self.infer(x)
        return self.decode(latents["z1"]), latents

    def sample(self, count: int, device: torch.device) -> Tensor:
        z2 = torch.randn(count, self.z2_dim, device=device)
        p1_mu, p1_logvar = self._split_gaussian_parameters(self.z1_prior(z2))
        z1 = self._sample(p1_mu, p1_logvar)
        return self.decode(z1)


def ladder_loss(
    x_hat: Tensor,
    x: Tensor,
    latents: dict[str, Tensor],
    kl_weight: float,
    free_bits: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    reconstruction = F.binary_cross_entropy(x_hat, x, reduction="none")
    reconstruction = reconstruction.flatten(1).sum(dim=1).mean()
    kl1_raw = gaussian_kl(
        latents["q1_mu"],
        latents["q1_logvar"],
        latents["p1_mu"],
        latents["p1_logvar"],
    )
    kl2_raw = gaussian_kl(
        latents["q2_mu"],
        latents["q2_logvar"],
        latents["p2_mu"],
        latents["p2_logvar"],
    )
    # Clamping the objective removes the incentive to push a layer's KL below
    # the threshold; it does not guarantee that the layer carries that much.
    kl1_objective = kl1_raw.clamp_min(free_bits).mean()
    kl2_objective = kl2_raw.clamp_min(free_bits).mean()
    loss = reconstruction + kl_weight * (kl1_objective + kl2_objective)
    terms = {
        "reconstruction": reconstruction.detach(),
        "kl_z1": kl1_raw.mean().detach(),
        "kl_z2": kl2_raw.mean().detach(),
    }
    return loss, terms


def smoke_test() -> None:
    torch.manual_seed(7)
    model = LadderVAE(z1_dim=12, z2_dim=6)
    x = torch.rand(8, 1, 28, 28)
    x_hat, latents = model(x)
    loss, terms = ladder_loss(x_hat, x, latents, kl_weight=0.5, free_bits=0.2)
    loss.backward()
    assert x_hat.shape == x.shape
    assert latents["q1_mu"].shape == (8, 12)
    assert latents["q2_mu"].shape == (8, 6)
    assert model.z1_prior[-1].weight.grad is not None
    print(
        "smoke test passed: "
        + ", ".join(f"{key}={value.item():.3f}" for key, value in terms.items())
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "ladder_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        PROJECT_ROOT / "data",
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
    )
    model = LadderVAE(args.z1_dim, args.z2_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        kl_weight = min(1.0, epoch / max(args.warmup_epochs, 1))
        sums = torch.zeros(4, device=device)
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_hat, latents = model(x)
            loss, terms = ladder_loss(
                x_hat, x, latents, kl_weight, args.free_bits
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = x.shape[0]
            sums += torch.stack(
                [
                    loss.detach(),
                    terms["reconstruction"],
                    terms["kl_z1"],
                    terms["kl_z2"],
                ]
            ) * batch_size
            examples += batch_size

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: beta={kl_weight:.2f}, loss={means[0]:.3f}, "
            f"recon={means[1]:.3f}, KL(z1)={means[2]:.3f}, KL(z2)={means[3]:.3f}"
        )
        model.eval()
        with torch.inference_mode():
            samples = model.sample(64, device)
        save_image(samples, out_dir / f"epoch_{epoch:03d}.png", nrow=8)

    torch.save(
        {
            "model": model.state_dict(),
            "z1_dim": args.z1_dim,
            "z2_dim": args.z2_dim,
            "free_bits": args.free_bits,
        },
        out_dir / "ladder_vae.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--free-bits", type=float, default=0.5)
    parser.add_argument("--z1-dim", type=int, default=32)
    parser.add_argument("--z2-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
