"""Small Gaussian-VAE primitives shared by the post-beta-VAE lessons.

The older :mod:`dl_utils.vae.vae` module intentionally keeps the 256x256
face model used by the first VAE lessons.  This module contains only the
distribution algebra and the compact 28x28 model reused by the later
algorithm-comparison scripts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

LOG_2PI = math.log(2.0 * math.pi)


def split_gaussian_parameters(
    raw: Tensor, *, dimension: int = 1, minimum: float = -12.0, maximum: float = 12.0
) -> tuple[Tensor, Tensor]:
    """Split a tensor into mean/log-variance and bound the exponential range."""
    mu, logvar = raw.chunk(2, dim=dimension)
    return mu, logvar.clamp(minimum, maximum)


def reparameterize_logvar(mu: Tensor, logvar: Tensor) -> Tensor:
    """Draw ``N(mu, exp(logvar))`` with the reparameterization trick."""
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have matching shapes")
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def diagonal_gaussian_kl_from_logvar(
    q_mu: Tensor,
    q_logvar: Tensor,
    p_mu: Tensor | None = None,
    p_logvar: Tensor | None = None,
) -> Tensor:
    """Return elementwise ``KL(q || p)`` for diagonal Gaussian distributions.

    When ``p_mu`` and ``p_logvar`` are omitted, ``p`` is ``N(0, I)``.  No
    dimensions are reduced so each lesson can make its own per-sample and
    per-layer reduction explicit.
    """
    if q_mu.shape != q_logvar.shape:
        raise ValueError("q_mu and q_logvar must have matching shapes")
    if (p_mu is None) != (p_logvar is None):
        raise ValueError("p_mu and p_logvar must be supplied together")
    if p_mu is None:
        return 0.5 * (q_mu.square() + q_logvar.exp() - 1.0 - q_logvar)
    if p_mu.shape != q_mu.shape or p_logvar.shape != q_mu.shape:
        raise ValueError("q and p parameters must have matching shapes")
    return 0.5 * (
        p_logvar
        - q_logvar
        + torch.exp(q_logvar - p_logvar)
        + (q_mu - p_mu).square() * torch.exp(-p_logvar)
        - 1.0
    )


def diagonal_gaussian_log_density(
    sample: Tensor, mu: Tensor, logvar: Tensor
) -> Tensor:
    """Return elementwise ``log N(sample; mu, exp(logvar))`` with broadcasting."""
    return -0.5 * (
        LOG_2PI + logvar + (sample - mu).square() * torch.exp(-logvar)
    )


def fuse_diagonal_gaussians(
    prior_mu: Tensor,
    prior_logvar: Tensor,
    evidence_mu: Tensor,
    evidence_logvar: Tensor,
) -> tuple[Tensor, Tensor]:
    """Normalize the product of two diagonal Gaussians.

    This is the precision-weighted update used by the Ladder VAE posterior.
    """
    if not (
        prior_mu.shape
        == prior_logvar.shape
        == evidence_mu.shape
        == evidence_logvar.shape
    ):
        raise ValueError("all Gaussian parameters must have matching shapes")
    prior_precision = torch.exp(-prior_logvar)
    evidence_precision = torch.exp(-evidence_logvar)
    fused_precision = prior_precision + evidence_precision
    fused_variance = fused_precision.reciprocal()
    fused_mu = fused_variance * (
        prior_precision * prior_mu + evidence_precision * evidence_mu
    )
    return fused_mu, fused_variance.log()


class ConvGaussianVAE28(nn.Module):
    """Compact convolutional Gaussian VAE for 28x28 grayscale lessons."""

    def __init__(self, latent_dim: int = 10, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.posterior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return split_gaussian_parameters(self.posterior(self.encoder(x)))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.decoder_input(z))

    def reconstruct(self, x: Tensor, *, sample: bool = False) -> Tensor:
        mu, logvar = self.encode(x)
        z = reparameterize_logvar(mu, logvar) if sample else mu
        return self.decode(z)

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        return self.decode(torch.randn(count, self.latent_dim, device=device))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize_logvar(mu, logvar)
        return self.decode(z), mu, logvar, z


__all__ = [
    "LOG_2PI",
    "ConvGaussianVAE28",
    "diagonal_gaussian_kl_from_logvar",
    "diagonal_gaussian_log_density",
    "fuse_diagonal_gaussians",
    "reparameterize_logvar",
    "split_gaussian_parameters",
]
