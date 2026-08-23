"""Shared 32x32 VAE backbone and posterior-flow primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    LOG_2PI,
    diagonal_gaussian_log_density,
    split_gaussian_parameters,
)


class GaussianVAE32(nn.Module):
    """Compact Bernoulli VAE with an explicit particle-shaped posterior API."""

    posterior_family = "diagonal_gaussian"

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        hidden_channels: int = 128,
        context_dim: int = 128,
    ) -> None:
        super().__init__()
        if hidden_channels % 8:
            raise ValueError("hidden_channels must be divisible by 8")
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4, context_dim),
            nn.SiLU(),
        )
        self.base_posterior = nn.Linear(context_dim, 2 * latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 4 * 4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (hidden_channels, 4, 4)),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels // 2, 4, 2, 1
            ),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_channels // 2, hidden_channels // 4, 4, 2, 1
            ),
            nn.GroupNorm(4, hidden_channels // 4),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1:] != (1, 32, 32):
            raise ValueError("x must have shape [batch, 1, 32, 32]")
        context = self.context(self.encoder(x))
        mu, logvar = split_gaussian_parameters(
            self.base_posterior(context)
        )
        return mu, logvar, context

    def decode(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError("the final z dimension must equal latent_dim")
        leading_shape = z.shape[:-1]
        flat_z = z.reshape(-1, self.latent_dim)
        images = self.decoder(self.decoder_input(flat_z))
        return images.reshape(*leading_shape, 1, 32, 32)

    def transform_posterior(
        self, z0: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        del context
        return (
            z0,
            z0.new_zeros(z0.shape[:-1]),
            {
                "log_scale_min": z0.new_zeros(()),
                "log_scale_max": z0.new_zeros(()),
            },
        )

    def sample_from_statistics(
        self,
        mu: Tensor,
        logvar: Tensor,
        context: Tensor,
        *,
        particles: int,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if particles < 1:
            raise ValueError("particles must be positive")
        epsilon = torch.randn(
            mu.shape[0],
            particles,
            self.latent_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        z0 = mu[:, None, :] + torch.exp(0.5 * logvar[:, None, :]) * epsilon
        log_q0 = diagonal_gaussian_log_density(
            z0, mu[:, None, :], logvar[:, None, :]
        ).sum(dim=-1)
        z, log_determinant, flow_statistics = self.transform_posterior(
            z0, context
        )
        return z, log_q0 - log_determinant, {
            "z0": z0,
            "log_determinant": log_determinant,
            **flow_statistics,
        }

    def sample_posterior(
        self, x: Tensor, *, particles: int
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        mu, logvar, context = self.encode(x)
        z, log_q, statistics = self.sample_from_statistics(
            mu, logvar, context, particles=particles
        )
        return z, log_q, {
            "base_mu": mu,
            "base_logvar": logvar,
            "context": context,
            **statistics,
        }

    def posterior_representative(self, x: Tensor) -> Tensor:
        """Transform the base mean into one deterministic latent.

        This equals the posterior mean for the identity Gaussian family. For
        nonlinear flows it is a reproducible representative, not generally
        the expectation of the transformed distribution.
        """
        mu, _, context = self.encode(x)
        z, _, _ = self.transform_posterior(mu[:, None, :], context)
        return z[:, 0]

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.decode(self.posterior_representative(x))

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        return self.decode(
            torch.randn(count, self.latent_dim, device=device)
        )


class MaskedLinear(nn.Linear):
    """Linear layer with a fixed MADE connectivity mask."""

    def __init__(
        self, in_features: int, out_features: int, mask: Tensor
    ) -> None:
        super().__init__(in_features, out_features)
        if mask.shape != (out_features, in_features):
            raise ValueError("mask shape must match the linear weight")
        self.register_buffer("mask", mask.to(dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


def _hidden_degrees(width: int, latent_dim: int) -> Tensor:
    if latent_dim < 2:
        raise ValueError("IAF requires latent_dim >= 2")
    return torch.arange(width) % (latent_dim - 1) + 1


class ConditionalMADE(nn.Module):
    """Produce affine IAF parameters using strictly autoregressive masks."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        input_degrees = torch.arange(1, latent_dim + 1)
        hidden_degrees_1 = _hidden_degrees(hidden_dim, latent_dim)
        hidden_degrees_2 = _hidden_degrees(hidden_dim, latent_dim)
        output_degrees = torch.arange(1, latent_dim + 1).repeat(2)

        input_mask = (
            hidden_degrees_1[:, None] >= input_degrees[None, :]
        )
        hidden_mask = (
            hidden_degrees_2[:, None] >= hidden_degrees_1[None, :]
        )
        output_mask = (
            output_degrees[:, None] > hidden_degrees_2[None, :]
        )
        self.input = MaskedLinear(
            latent_dim, hidden_dim, input_mask
        )
        self.context = nn.Linear(context_dim, hidden_dim, bias=False)
        self.hidden = MaskedLinear(
            hidden_dim, hidden_dim, hidden_mask
        )
        self.output = MaskedLinear(
            hidden_dim, 2 * latent_dim, output_mask
        )
        # An identity initialization lets the base Gaussian train before the
        # flow learns useful shifts and scales.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        h = F.silu(self.input(z) + self.context(context))
        h = F.silu(self.hidden(h))
        return self.output(h).chunk(2, dim=-1)


class AffineIAFLayer(nn.Module):
    """One parallel-sampling affine inverse autoregressive flow layer."""

    def __init__(
        self,
        *,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        max_abs_log_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if max_abs_log_scale <= 0:
            raise ValueError("max_abs_log_scale must be positive")
        self.max_abs_log_scale = float(max_abs_log_scale)
        self.autoregressive = ConditionalMADE(
            latent_dim, context_dim, hidden_dim
        )

    def forward(
        self, z: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        shift, raw_log_scale = self.autoregressive(z, context)
        log_scale = self.max_abs_log_scale * torch.tanh(raw_log_scale)
        transformed = shift + torch.exp(log_scale) * z
        return transformed, log_scale.sum(dim=-1), log_scale


class FlowGaussianVAE32(GaussianVAE32):
    """Gaussian-base VAE whose variational posterior ends in short IAF."""

    posterior_family = "iaf"

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        hidden_channels: int = 128,
        context_dim: int = 128,
        flow_layers: int = 2,
        flow_hidden_dim: int = 128,
        max_abs_log_scale: float = 2.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            context_dim=context_dim,
        )
        if flow_layers < 1:
            raise ValueError("flow_layers must be positive")
        self.flow_layers_count = flow_layers
        self.flow_hidden_dim = flow_hidden_dim
        self.max_abs_log_scale = max_abs_log_scale
        self.flows = nn.ModuleList(
            [
                AffineIAFLayer(
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    hidden_dim=flow_hidden_dim,
                    max_abs_log_scale=max_abs_log_scale,
                )
                for _ in range(flow_layers)
            ]
        )
        for index in range(flow_layers):
            permutation = (
                torch.arange(latent_dim)
                if index == 0
                else torch.arange(latent_dim - 1, -1, -1)
            )
            self.register_buffer(f"permutation_{index}", permutation)

    def transform_posterior(
        self, z0: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if z0.ndim != 3:
            raise ValueError("flow posterior expects [batch, particles, latent]")
        context_particles = context[:, None, :].expand(
            -1, z0.shape[1], -1
        )
        z = z0
        total_log_determinant = z0.new_zeros(z0.shape[:-1])
        all_log_scales = []
        for index, flow in enumerate(self.flows):
            permutation = getattr(self, f"permutation_{index}")
            z = z[..., permutation]
            z, log_determinant, log_scale = flow(
                z, context_particles
            )
            total_log_determinant += log_determinant
            all_log_scales.append(log_scale)
        stacked_scales = torch.stack(all_log_scales, dim=0)
        return z, total_log_determinant, {
            "log_scale_min": stacked_scales.min(),
            "log_scale_max": stacked_scales.max(),
        }


def standard_normal_log_density(z: Tensor) -> Tensor:
    """Return log N(z; 0, I), reduced only over the final event dimension."""
    return -0.5 * (LOG_2PI + z.square()).sum(dim=-1)


def bernoulli_log_density(mean: Tensor, target: Tensor) -> Tensor:
    """Return log p(target | mean) for particle-shaped Bernoulli means."""
    if mean.ndim != 5 or target.ndim != 4:
        raise ValueError(
            "mean must be [B, K, C, H, W] and target [B, C, H, W]"
        )
    expanded_target = target[:, None, ...].expand_as(mean)
    return -F.binary_cross_entropy(
        mean, expanded_target, reduction="none"
    ).flatten(2).sum(dim=-1)


def importance_log_weights(
    model: GaussianVAE32,
    x: Tensor,
    *,
    particles: int,
    particle_chunk_size: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute per-example importance log weights entirely in log space."""
    if particles < 1:
        raise ValueError("particles must be positive")
    chunk_size = particles if particle_chunk_size is None else particle_chunk_size
    if chunk_size < 1:
        raise ValueError("particle_chunk_size must be positive")
    mu, logvar, context = model.encode(x)
    log_weight_chunks = []
    log_px_chunks = []
    log_pz_chunks = []
    log_q_chunks = []
    log_determinant_chunks = []
    scale_mins = []
    scale_maxes = []
    remaining = particles
    while remaining:
        count = min(remaining, chunk_size)
        z, log_q, posterior = model.sample_from_statistics(
            mu, logvar, context, particles=count
        )
        reconstruction = model.decode(z)
        log_px = bernoulli_log_density(reconstruction, x)
        log_pz = standard_normal_log_density(z)
        log_weight_chunks.append(log_px + log_pz - log_q)
        log_px_chunks.append(log_px)
        log_pz_chunks.append(log_pz)
        log_q_chunks.append(log_q)
        log_determinant_chunks.append(posterior["log_determinant"])
        scale_mins.append(posterior["log_scale_min"])
        scale_maxes.append(posterior["log_scale_max"])
        remaining -= count

    log_weights = torch.cat(log_weight_chunks, dim=1)
    return log_weights, {
        "log_px": torch.cat(log_px_chunks, dim=1),
        "log_pz": torch.cat(log_pz_chunks, dim=1),
        "log_q": torch.cat(log_q_chunks, dim=1),
        "base_mu": mu,
        "base_logvar": logvar,
        "log_determinant": torch.cat(log_determinant_chunks, dim=1),
        "log_scale_min": torch.stack(scale_mins).min(),
        "log_scale_max": torch.stack(scale_maxes).max(),
    }


def log_mean_exp(log_weights: Tensor) -> Tensor:
    """Reduce only the particle axis of [batch, particles] log weights."""
    if log_weights.ndim != 2 or log_weights.shape[1] < 1:
        raise ValueError("log_weights must have shape [batch, particles]")
    return torch.logsumexp(log_weights, dim=1) - math.log(
        log_weights.shape[1]
    )


def importance_diagnostics(
    log_weights: Tensor, terms: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Return detached diagnostics without changing the IWAE gradient."""
    normalized = torch.softmax(log_weights, dim=1)
    ess = normalized.square().sum(dim=1).reciprocal()
    return {
        "bound": log_mean_exp(log_weights).mean().detach(),
        "log_px": terms["log_px"].mean().detach(),
        "log_pz": terms["log_pz"].mean().detach(),
        "log_q": terms["log_q"].mean().detach(),
        "ess_fraction": (ess / log_weights.shape[1]).mean().detach(),
        "log_weight_range": (
            log_weights.max(dim=1).values
            - log_weights.min(dim=1).values
        ).mean().detach(),
        "log_determinant": terms["log_determinant"].mean().detach(),
        "log_scale_min": terms["log_scale_min"].detach(),
        "log_scale_max": terms["log_scale_max"].detach(),
    }


def model_config(model: GaussianVAE32) -> dict[str, int | float]:
    config: dict[str, int | float] = {
        "latent_dim": model.latent_dim,
        "hidden_channels": model.hidden_channels,
        "context_dim": model.context_dim,
    }
    if isinstance(model, FlowGaussianVAE32):
        config.update(
            {
                "flow_layers": model.flow_layers_count,
                "flow_hidden_dim": model.flow_hidden_dim,
                "max_abs_log_scale": model.max_abs_log_scale,
            }
        )
    return config


__all__ = [
    "AffineIAFLayer",
    "FlowGaussianVAE32",
    "GaussianVAE32",
    "bernoulli_log_density",
    "importance_diagnostics",
    "importance_log_weights",
    "log_mean_exp",
    "model_config",
    "standard_normal_log_density",
]
