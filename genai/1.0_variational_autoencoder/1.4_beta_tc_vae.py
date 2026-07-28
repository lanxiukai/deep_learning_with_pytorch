"""
Beta-TCVAE: penalize dependence in the aggregated posterior directly.

The usual beta-VAE multiplies the whole KL term.  Beta-TCVAE instead uses

    reconstruction + alpha * mutual_information
                   + beta  * total_correlation
                   + gamma * dimension_wise_KL

This file intentionally uses the simple minibatch-mixture estimator: every
q(z_i) is estimated from the other posteriors in the current batch.  It makes
the density-ratio calculation visible, but paper-level dataset comparisons
should use the minibatch-weighted estimator from Chen et al. (2018).

Run a one-batch gradient/shape check:
    python genai/1.0_variational_autoencoder/1.4_beta_tc_vae.py --smoke-test

Train on MNIST:
    python genai/1.0_variational_autoencoder/1.4_beta_tc_vae.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_2PI = math.log(2.0 * math.pi)


class BetaTCVAE(nn.Module):
    """Small convolutional VAE for 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
        )
        self.to_mu = nn.Linear(256, latent_dim)
        self.to_logvar = nn.Linear(256, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-12.0, 12.0)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.decoder_input(z))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def gaussian_log_density(samples: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Elementwise log N(samples; mu, exp(logvar)), with broadcasting."""
    return -0.5 * (
        LOG_2PI + logvar + (samples - mu).square() * torch.exp(-logvar)
    )


def decompose_kl(
    z: Tensor, mu: Tensor, logvar: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate MI, total correlation and dimension-wise KL from one batch.

    Shapes:
      z[:, None, :]   -> samples i
      mu[None, :, :]  -> posterior mixture components q(z | x_j)
      pair_log_qz[i, j, d] = log q(z_i[d] | x_j)
    """
    batch_size = z.shape[0]
    if batch_size < 2:
        raise ValueError("Beta-TCVAE's minibatch estimator needs batch_size >= 2")

    log_q_z_given_x = gaussian_log_density(z, mu, logvar).sum(dim=1)
    pair_log_qz = gaussian_log_density(
        z[:, None, :], mu[None, :, :], logvar[None, :, :]
    )
    normalizer = math.log(batch_size)
    log_q_z = torch.logsumexp(pair_log_qz.sum(dim=2), dim=1) - normalizer
    log_prod_q_z = (
        torch.logsumexp(pair_log_qz, dim=1) - normalizer
    ).sum(dim=1)
    log_p_z = gaussian_log_density(
        z, torch.zeros_like(z), torch.zeros_like(z)
    ).sum(dim=1)

    mutual_information = (log_q_z_given_x - log_q_z).mean()
    total_correlation = (log_q_z - log_prod_q_z).mean()
    dimension_wise_kl = (log_prod_q_z - log_p_z).mean()
    return mutual_information, total_correlation, dimension_wise_kl


def beta_tc_vae_loss(
    x_hat: Tensor,
    x: Tensor,
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    alpha: float = 1.0,
    beta: float = 6.0,
    gamma: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    reconstruction = F.binary_cross_entropy(x_hat, x, reduction="none")
    reconstruction = reconstruction.flatten(1).sum(dim=1).mean()
    mi, tc, dimension_kl = decompose_kl(z, mu, logvar)
    loss = reconstruction + alpha * mi + beta * tc + gamma * dimension_kl
    terms = {
        "reconstruction": reconstruction.detach(),
        "mutual_information": mi.detach(),
        "total_correlation": tc.detach(),
        "dimension_kl": dimension_kl.detach(),
    }
    return loss, terms


def smoke_test() -> None:
    torch.manual_seed(7)
    model = BetaTCVAE(latent_dim=6)
    x = torch.rand(8, 1, 28, 28)
    x_hat, mu, logvar, z = model(x)
    loss, terms = beta_tc_vae_loss(x_hat, x, z, mu, logvar)
    loss.backward()
    assert x_hat.shape == x.shape
    assert model.to_mu.weight.grad is not None
    values = ", ".join(f"{name}={value.item():.3f}" for name, value in terms.items())
    print(f"smoke test passed: loss={loss.item():.3f}, {values}")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "output" / "beta_tc_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = BetaTCVAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fixed_z = torch.randn(64, args.latent_dim, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = torch.zeros(5, device=device)
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar, z = model(x)
            loss, terms = beta_tc_vae_loss(
                x_hat,
                x,
                z,
                mu,
                logvar,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            values = [
                loss.detach(),
                terms["reconstruction"],
                terms["mutual_information"],
                terms["total_correlation"],
                terms["dimension_kl"],
            ]
            sums += torch.stack(values) * batch_size
            examples += batch_size

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: loss={means[0]:.3f}, recon={means[1]:.3f}, "
            f"mi={means[2]:.3f}, tc={means[3]:.3f}, dim_kl={means[4]:.3f}"
        )
        model.eval()
        with torch.inference_mode():
            samples = model.decode(fixed_z)
        save_image(samples, out_dir / f"epoch_{epoch:03d}.png", nrow=8)

    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": args.latent_dim,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        },
        out_dir / "beta_tc_vae.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--gamma", type=float, default=1.0)
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
