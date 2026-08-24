"""Aggregate-posterior matching primitives for the focused WAE lesson."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _validate_samples(encoded: Tensor, prior: Tensor) -> None:
    if encoded.ndim != 2 or prior.ndim != 2:
        raise ValueError("encoded and prior samples must be [batch, latent]")
    if encoded.shape != prior.shape:
        raise ValueError("encoded and prior samples must have matching shapes")
    if encoded.shape[0] < 2:
        raise ValueError("MMD needs at least two samples")


def _imq_kernel(
    left: Tensor,
    right: Tensor,
    *,
    scales: Sequence[float],
) -> Tensor:
    """Return a multiscale inverse-multiquadratic kernel matrix."""
    if not scales or any(scale <= 0.0 for scale in scales):
        raise ValueError("kernel scales must be positive")
    squared_distance = torch.cdist(left, right).square()
    latent_dim = left.shape[1]
    kernels = []
    for scale in scales:
        bandwidth = 2.0 * latent_dim * float(scale)
        kernels.append(bandwidth / (bandwidth + squared_distance))
    return torch.stack(kernels).mean(dim=0)


def imq_mmd2(
    encoded: Tensor,
    prior: Tensor,
    *,
    scales: Sequence[float] = (0.5, 1.0, 2.0),
    biased: bool = True,
) -> Tensor:
    """Estimate MMD^2 between aggregate encoded samples and the prior.

    The biased V-statistic is non-negative up to floating-point error and is
    the stable training default.  The unbiased U-statistic omits the two
    self-kernel diagonals and can be slightly negative for a finite batch.
    """
    _validate_samples(encoded, prior)
    encoded_kernel = _imq_kernel(encoded, encoded, scales=scales)
    prior_kernel = _imq_kernel(prior, prior, scales=scales)
    cross_kernel = _imq_kernel(encoded, prior, scales=scales)
    if biased:
        return (
            encoded_kernel.mean()
            + prior_kernel.mean()
            - 2.0 * cross_kernel.mean()
        )

    batch_size = encoded.shape[0]
    denominator = batch_size * (batch_size - 1)
    encoded_term = (
        encoded_kernel.sum() - encoded_kernel.diagonal().sum()
    ) / denominator
    prior_term = (
        prior_kernel.sum() - prior_kernel.diagonal().sum()
    ) / denominator
    return encoded_term + prior_term - 2.0 * cross_kernel.mean()


def latent_moment_diagnostics(encoded: Tensor) -> dict[str, Tensor]:
    """Measure simple aggregate-posterior moment mismatch to N(0, I)."""
    if encoded.ndim != 2 or encoded.shape[0] < 2:
        raise ValueError("encoded must be [samples, latent] with samples >= 2")
    mean = encoded.mean(dim=0)
    variance = encoded.var(dim=0, unbiased=False)
    return {
        "latent_mean_norm": mean.norm(),
        "latent_variance_mean": variance.mean(),
        "latent_variance_error": (variance - 1.0).abs().mean(),
        "active_units": (variance > 1e-2).sum().to(encoded.dtype),
    }


__all__ = ["imq_mmd2", "latent_moment_diagnostics"]
