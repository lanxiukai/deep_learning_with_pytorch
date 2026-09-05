"""BigGAN components for a paper-oriented 128x128 teaching model.

The 64-channel default keeps the original 128x128 architecture, hierarchical
120-dimensional latent input, shared 128-dimensional class embedding,
64x64 attention, spectral normalization, and projection discriminator. It
deliberately avoids the paper's costly 96-channel and 2,048-image scaling.
"""

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from dl_utils.gan.sagan import SAGANDiscriminator, SelfAttention
from dl_utils.gan.sn_gan import (
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    validate_class_labels,
)


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
    """Vector-conditioned residual block that upsamples by two."""

    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.norm1 = ConditionalBatchNorm2d(in_channels, condition_dim)
        self.norm2 = ConditionalBatchNorm2d(out_channels, condition_dim)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
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


class BigGANGenerator(nn.Module):
    """128x128 BigGAN generator with hierarchical latent conditioning."""

    def __init__(
        self,
        z_dim=120,
        num_classes=IMAGENETTE_NUM_CLASSES,
        base_channels=64,
        class_embedding_dim=128,
        image_size=IMAGENETTE_IMAGE_SIZE,
        attention_resolution=64,
    ):
        super().__init__()
        if base_channels < 1 or num_classes < 1 or class_embedding_dim < 1:
            raise ValueError(
                "base_channels, num_classes, and class_embedding_dim must be positive."
            )
        if image_size != IMAGENETTE_IMAGE_SIZE:
            raise ValueError(
                "BigGANGenerator currently implements only 128x128 output."
            )
        block_resolutions = (8, 16, 32, 64, 128)
        if attention_resolution not in block_resolutions:
            raise ValueError(
                f"attention_resolution must be one of {block_resolutions}."
            )

        self.num_generator_blocks = len(block_resolutions)
        num_latent_chunks = self.num_generator_blocks + 1
        if z_dim < 1 or z_dim % num_latent_chunks != 0:
            raise ValueError(
                "z_dim must be positive and divisible by the six hierarchical "
                f"latent chunks, got {z_dim}."
            )

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.latent_chunk_dim = z_dim // num_latent_chunks
        self.attention_after_block = block_resolutions.index(attention_resolution)
        condition_dim = class_embedding_dim + self.latent_chunk_dim
        channels = [
            base_channels * 16,
            base_channels * 16,
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
        ]

        self.class_embedding = nn.Embedding(
            num_classes,
            class_embedding_dim,
        )
        self.input = spectral_norm(
            nn.Linear(
                self.latent_chunk_dim,
                channels[0] * 4 * 4,
            )
        )
        self.blocks = nn.ModuleList(
            BigGANGeneratorResidualBlock(
                in_channels,
                out_channels,
                condition_dim,
            )
            for in_channels, out_channels in pairwise(channels)
        )
        attention_channels = channels[self.attention_after_block + 1]
        self.attention = SelfAttention(attention_channels)
        self.output_norm = nn.BatchNorm2d(channels[-1])
        self.output = spectral_norm(nn.Conv2d(channels[-1], 3, 3, padding=1))

    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        validate_class_labels(labels, noise.shape[0])

        latent_chunks = noise.split(self.latent_chunk_dim, dim=1)
        class_condition = self.class_embedding(labels)
        block_conditions = [
            torch.cat([class_condition, chunk], dim=1) for chunk in latent_chunks[1:]
        ]

        hidden = self.input(latent_chunks[0]).view(
            noise.shape[0],
            -1,
            4,
            4,
        )
        for index, (block, condition) in enumerate(zip(self.blocks, block_conditions)):
            hidden = block(hidden, condition)
            if index == self.attention_after_block:
                hidden = self.attention(hidden)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class BigGANDiscriminator(SAGANDiscriminator):
    """BigGAN's wide projection discriminator with 64x64 attention."""

    def __init__(
        self,
        num_classes=IMAGENETTE_NUM_CLASSES,
        base_channels=64,
        image_size=IMAGENETTE_IMAGE_SIZE,
        attention_resolution=64,
    ):
        super().__init__(
            num_classes=num_classes,
            base_channels=base_channels,
            image_size=image_size,
            attention_resolution=attention_resolution,
        )


# Keep old imports working while the lesson now presents the model as BigGAN.
CompactBigGANGenerator = BigGANGenerator


def initialize_orthogonal_weights(module):
    """Apply BigGAN's orthogonal initialization to learned weight tensors."""
    if not isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        return
    weight = getattr(module, "weight_orig", None)
    if weight is None:
        weight = module.weight
    nn.init.orthogonal_(weight)
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        nn.init.zeros_(bias)


def modified_orthogonal_regularization(module, strength=1e-4):
    """Return BigGAN's filter-wise modified orthogonal regularizer."""
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
            off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
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


def orthogonal_scratch_minimal(weight, gain=1.0):
    """Fill a weight tensor with a scaled orthogonal matrix using QR."""
    if weight.ndim < 2:
        raise ValueError(
            "orthogonal_scratch_minimal requires at least two-dimensional "
            f"weight, but got {weight.ndim} dimensions"
        )

    with torch.no_grad():
        rows = weight.size(0)
        columns = weight[0].numel()
        matrix = weight.new_empty(rows, columns).normal_()
        if rows < columns:
            matrix = matrix.t()

        q, r = torch.linalg.qr(matrix)
        q *= torch.diagonal(r).sign()
        if rows < columns:
            q = q.t()

        weight.copy_(q.reshape_as(weight)).mul_(gain)
    return weight


__all__ = [
    "BigGANDiscriminator",
    "BigGANGenerator",
    "BigGANGeneratorResidualBlock",
    "CompactBigGANGenerator",
    "ConditionalBatchNorm2d",
    "initialize_orthogonal_weights",
    "modified_orthogonal_regularization",
    "orthogonal_scratch_minimal",
    "truncated_normal",
]
