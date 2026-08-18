"""Training primitives shared by the compact StyleGAN lessons."""

from __future__ import annotations

import math
import random

import torch


def sample_mixing_latents(
    batch_size: int,
    z_dim: int,
    device: torch.device,
    mixing_probability: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample a primary Z batch and an optional style-mixing batch."""
    if batch_size < 1 or z_dim < 1:
        raise ValueError("batch_size and z_dim must be positive.")
    if not 0.0 <= mixing_probability <= 1.0:
        raise ValueError("mixing_probability must be within [0, 1].")
    z = torch.randn(batch_size, z_dim, device=device)
    mixing_z = (
        torch.randn_like(z)
        if random.random() < mixing_probability
        else None
    )
    return z, mixing_z


def r1_penalty(
    real_scores: torch.Tensor,
    real_images: torch.Tensor,
) -> torch.Tensor:
    """Measure squared discriminator gradients at real images."""
    gradients = torch.autograd.grad(
        real_scores.sum(),
        real_images,
        create_graph=True,
    )[0]
    return gradients.square().flatten(1).sum(dim=1).mean()


def path_length_penalty(
    images: torch.Tensor,
    ws: torch.Tensor,
    running_mean: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularize image-space change per unit movement in W."""
    if images.ndim != 4:
        raise ValueError("images must have BCHW shape.")
    if ws.ndim != 3 or ws.shape[0] != images.shape[0]:
        raise ValueError("ws must have shape [batch, num_ws, style_dim].")
    if running_mean.ndim != 0:
        raise ValueError("running_mean must be a scalar tensor.")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be within [0, 1].")

    noise = torch.randn_like(images) / math.sqrt(
        images.shape[2] * images.shape[3]
    )
    gradients = torch.autograd.grad(
        (images * noise).sum(),
        ws,
        create_graph=True,
    )[0]
    lengths = torch.sqrt(
        gradients.square().sum(dim=2).mean(dim=1) + 1e-8
    )
    updated_mean = running_mean.lerp(lengths.mean().detach(), decay)
    return (lengths - updated_mean).square().mean(), updated_mean


__all__ = [
    "path_length_penalty",
    "r1_penalty",
    "sample_mixing_latents",
]
