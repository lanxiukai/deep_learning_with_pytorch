"""Reusable two-level Gaussian hierarchy for the HVAE/Ladder lessons."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
    split_gaussian_parameters,
)


class CompactHierarchicalVAE(nn.Module):
    """Two-level HVAE with a top-down conditional posterior.

    The posterior factorization is ``q(z2 | x) q(z1 | z2, x)`` and therefore
    matches the direction of ``p(z2) p(z1 | z2) p(x | z1)``.  The Ladder
    lesson subclasses this model and changes only how the lower posterior is
    constructed.
    """

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
        self.top_posterior = nn.Linear(256, 2 * z2_dim)
        self.lower_prior = nn.Sequential(
            nn.Linear(z2_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 2 * z1_dim),
        )
        self.lower_posterior = nn.Sequential(
            nn.Linear(512 + z2_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 2 * z1_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z1_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),
        )

    def infer(self, x: Tensor, *, sample: bool = True) -> dict[str, Tensor]:
        h1 = self.bottom_up_1(x)
        h2 = self.bottom_up_2(h1)

        # The top posterior is parameterized by bottom-up evidence alone.  It
        # is compared with N(0, I), but is not multiplied by that prior.
        q2_mu, q2_logvar = split_gaussian_parameters(self.top_posterior(h2))
        z2 = reparameterize_logvar(q2_mu, q2_logvar) if sample else q2_mu

        p1_mu, p1_logvar = split_gaussian_parameters(self.lower_prior(z2))
        q1_mu, q1_logvar = split_gaussian_parameters(
            self.lower_posterior(torch.cat((h1, z2), dim=1))
        )
        z1 = reparameterize_logvar(q1_mu, q1_logvar) if sample else q1_mu
        return {
            "z1": z1,
            "z2": z2,
            "q1_mu": q1_mu,
            "q1_logvar": q1_logvar,
            "p1_mu": p1_mu,
            "p1_logvar": p1_logvar,
            "q2_mu": q2_mu,
            "q2_logvar": q2_logvar,
            "p2_mu": torch.zeros_like(q2_mu),
            "p2_logvar": torch.zeros_like(q2_logvar),
            "bottom_up_h1": h1,
        }

    def decode(self, z1: Tensor) -> Tensor:
        return self.decoder(z1).reshape(-1, 1, 28, 28)

    def reconstruct(self, x: Tensor, *, sample: bool = False) -> Tensor:
        return self.decode(self.infer(x, sample=sample)["z1"])

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        latents = self.infer(x, sample=True)
        return self.decode(latents["z1"]), latents

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        z2 = torch.randn(count, self.z2_dim, device=device)
        p1_mu, p1_logvar = split_gaussian_parameters(self.lower_prior(z2))
        z1 = reparameterize_logvar(p1_mu, p1_logvar)
        return self.decode(z1)


class ActiveUnitAccumulator:
    """Dataset-window variance of top means and lower posterior corrections."""

    def __init__(self) -> None:
        self.count = 0
        self.lower_sum: Tensor | None = None
        self.lower_square_sum: Tensor | None = None
        self.top_sum: Tensor | None = None
        self.top_square_sum: Tensor | None = None

    def update(self, latents: dict[str, Tensor]) -> None:
        lower = (latents["q1_mu"] - latents["p1_mu"]).detach().double()
        top = latents["q2_mu"].detach().double()
        if self.lower_sum is None:
            self.lower_sum = torch.zeros_like(lower[0])
            self.lower_square_sum = torch.zeros_like(lower[0])
            self.top_sum = torch.zeros_like(top[0])
            self.top_square_sum = torch.zeros_like(top[0])
        assert self.lower_square_sum is not None
        assert self.top_sum is not None
        assert self.top_square_sum is not None
        self.lower_sum += lower.sum(dim=0)
        self.lower_square_sum += lower.square().sum(dim=0)
        self.top_sum += top.sum(dim=0)
        self.top_square_sum += top.square().sum(dim=0)
        self.count += lower.shape[0]

    def counts(self, *, variance_threshold: float = 1e-2) -> tuple[int, int]:
        if self.count < 2 or self.lower_sum is None:
            return 0, 0
        assert self.lower_square_sum is not None
        assert self.top_sum is not None
        assert self.top_square_sum is not None
        lower_variance = (
            self.lower_square_sum / self.count
            - (self.lower_sum / self.count).square()
        )
        top_variance = (
            self.top_square_sum / self.count - (self.top_sum / self.count).square()
        )
        return (
            int((lower_variance > variance_threshold).sum().item()),
            int((top_variance > variance_threshold).sum().item()),
        )


def hierarchical_vae_loss(
    reconstruction: Tensor,
    x: Tensor,
    latents: dict[str, Tensor],
    *,
    kl_weight: float,
    free_bits: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Negative hierarchical ELBO with group-wise free bits.

    Each layer's KL is first summed per sample and averaged over the batch.
    The free-bits threshold is applied *after* that batch expectation.
    """
    if kl_weight < 0 or free_bits < 0:
        raise ValueError("kl_weight and free_bits must be non-negative")
    distortion = F.binary_cross_entropy(reconstruction, x, reduction="none")
    distortion = distortion.flatten(1).sum(dim=1).mean()
    kl_z1 = diagonal_gaussian_kl_from_logvar(
        latents["q1_mu"],
        latents["q1_logvar"],
        latents["p1_mu"],
        latents["p1_logvar"],
    ).sum(dim=1).mean()
    kl_z2 = diagonal_gaussian_kl_from_logvar(
        latents["q2_mu"], latents["q2_logvar"]
    ).sum(dim=1).mean()

    threshold = distortion.new_tensor(float(free_bits))
    kl_objective = torch.maximum(kl_z1, threshold) + torch.maximum(
        kl_z2, threshold
    )
    loss = distortion + float(kl_weight) * kl_objective
    return loss, {
        "distortion": distortion.detach(),
        "kl_z1": kl_z1.detach(),
        "kl_z2": kl_z2.detach(),
        "kl_objective": kl_objective.detach(),
    }


def active_units(
    latents: dict[str, Tensor], *, variance_threshold: float = 1e-2
) -> tuple[Tensor, Tensor]:
    """Count batch-active top units and lower posterior corrections."""
    if latents["q2_mu"].shape[0] < 2:
        zero = latents["q2_mu"].new_zeros((), dtype=torch.long)
        return zero, zero
    top = latents["q2_mu"].var(dim=0, unbiased=False)
    lower_correction = latents["q1_mu"] - latents["p1_mu"]
    lower = lower_correction.var(dim=0, unbiased=False)
    return (
        (lower > variance_threshold).sum().detach(),
        (top > variance_threshold).sum().detach(),
    )


__all__ = [
    "ActiveUnitAccumulator",
    "CompactHierarchicalVAE",
    "active_units",
    "hierarchical_vae_loss",
]
