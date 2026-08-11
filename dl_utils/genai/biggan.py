"""Compact BigGAN additions built on the SAGAN discriminator.

The generator demonstrates shared class embeddings, conditional BatchNorm,
hierarchical latent inputs, and modified orthogonal regularization. It remains
a small CIFAR-10 teaching model rather than a large-scale BigGAN reproduction.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from dl_utils.genai.sagan import SAGANDiscriminator, SelfAttention


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm whose affine scale and bias depend on a condition vector."""

    def __init__(self, channels, condition_dim):
        super().__init__()
        self.normalization = nn.BatchNorm2d(channels, affine=False)
        self.affine = spectral_norm(nn.Linear(condition_dim, 2 * channels))
        nn.init.zeros_(self.affine.bias)

    def forward(self, inputs, condition):
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return self.normalization(inputs) * (1 + gamma) + beta


class BigGANGeneratorResidualBlock(nn.Module):
    """Conditional residual block that upsamples by a factor of two."""

    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.norm1 = ConditionalBatchNorm2d(in_channels, condition_dim)
        self.norm2 = ConditionalBatchNorm2d(out_channels, condition_dim)
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs, condition):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs, condition), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(self.norm2(hidden, condition), inplace=True))
        return hidden + residual


class CompactBigGANGenerator(nn.Module):
    """32x32 or 64x64 BigGAN-style generator with latent conditioning."""

    def __init__(
        self,
        z_dim=128,
        num_classes=10,
        base_channels=32,
        class_embedding_dim=128,
        image_size=32,
    ):
        super().__init__()
        if image_size not in (32, 64):
            raise ValueError("image_size must be 32 or 64.")
        self.image_size = image_size
        self.num_generator_blocks = 3 if image_size == 32 else 4
        num_latent_chunks = self.num_generator_blocks + 1
        if z_dim % num_latent_chunks != 0:
            raise ValueError(
                "z_dim must be divisible by the number of hierarchical "
                f"latent chunks ({num_latent_chunks}), got {z_dim}."
            )

        self.z_dim = z_dim
        self.latent_chunk_dim = z_dim // num_latent_chunks
        condition_dim = class_embedding_dim + self.latent_chunk_dim

        self.class_embedding = nn.Embedding(
            num_classes, class_embedding_dim
        )
        self.input = spectral_norm(
            nn.Linear(
                self.latent_chunk_dim,
                base_channels * 8 * 4 * 4,
            )
        )
        self.block1 = BigGANGeneratorResidualBlock(
            base_channels * 8,
            base_channels * 4,
            condition_dim,
        )
        self.block2 = BigGANGeneratorResidualBlock(
            base_channels * 4,
            base_channels * 2,
            condition_dim,
        )
        self.attention = SelfAttention(base_channels * 2)
        self.block3 = BigGANGeneratorResidualBlock(
            base_channels * 2,
            base_channels,
            condition_dim,
        )
        self.block4 = (
            BigGANGeneratorResidualBlock(
                base_channels,
                base_channels,
                condition_dim,
            )
            if image_size == 64
            else nn.Identity()
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        latent_chunks = noise.split(self.latent_chunk_dim, dim=1)

        class_condition = self.class_embedding(labels)
        block_conditions = [
            torch.cat([class_condition, chunk], dim=1)
            for chunk in latent_chunks[1:]
        ]

        hidden = self.input(latent_chunks[0])
        hidden = hidden.view(noise.shape[0], -1, 4, 4)
        hidden = self.block1(hidden, block_conditions[0])
        hidden = self.block2(hidden, block_conditions[1])
        hidden = self.attention(hidden)
        hidden = self.block3(hidden, block_conditions[2])
        if self.image_size == 64:
            hidden = self.block4(hidden, block_conditions[3])
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class BigGANDiscriminator(SAGANDiscriminator):
    """SAGAN projection discriminator reused by the compact BigGAN lesson."""


def initialize_orthogonal_weights(module):
    """Apply BigGAN's orthogonal initialization to learned weight tensors."""
    if not isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        return
    weight = getattr(module, "weight_orig", None)
    if weight is None:
        weight = module.weight
    nn.init.orthogonal_(weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def modified_orthogonal_regularization(module, strength=1e-4):
    """Return BigGAN's filter-wise modified orthogonal regularizer.

    For tall matrices, the equivalent column-Gram identity avoids allocating
    a potentially large filter-by-filter matrix without changing the loss or
    its gradient. The factor of one half matches the gradient update in the
    reference BigGAN implementation.
    """
    if strength < 0:
        raise ValueError("strength must be non-negative.")
    first_parameter = next(module.parameters())
    penalty = first_parameter.new_zeros(())

    for name, parameter in module.named_parameters():
        if parameter.ndim < 2 or "class_embedding" in name:
            continue
        matrix = parameter.flatten(start_dim=1)
        if matrix.shape[0] <= matrix.shape[1]:
            gram = matrix @ matrix.transpose(0, 1)
            off_diagonal = gram - torch.diag_embed(
                torch.diagonal(gram)
            )
            penalty = penalty + off_diagonal.square().sum()
        else:
            column_gram = matrix.transpose(0, 1) @ matrix
            row_norm_fourth = matrix.square().sum(dim=1).square().sum()
            penalty = penalty + column_gram.square().sum() - row_norm_fourth

    return 0.5 * strength * penalty


def truncated_normal(shape, truncation, device):
    """Sample a standard normal and reject coordinates outside the threshold."""
    if truncation <= 0:
        raise ValueError("truncation must be positive.")

    samples = torch.randn(shape, device=device)
    invalid = samples.abs() > truncation
    while invalid.any():
        samples[invalid] = torch.randn_like(samples[invalid])
        invalid = samples.abs() > truncation
    return samples


__all__ = [
    "BigGANDiscriminator",
    "BigGANGeneratorResidualBlock",
    "CompactBigGANGenerator",
    "ConditionalBatchNorm2d",
    "initialize_orthogonal_weights",
    "modified_orthogonal_regularization",
    "truncated_normal",
]
