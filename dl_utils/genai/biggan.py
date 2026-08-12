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
        # Map each condition from (B, condition_dim) to (B, 2C):
        # first C values for gamma, remaining C values for beta.
        self.affine = spectral_norm(nn.Linear(condition_dim, 2 * channels))
        nn.init.zeros_(self.affine.bias)

    def forward(self, inputs, condition):
        # inputs shape: (B, C, H, W), condition shape: (B, condition_dim)
        gamma, beta = self.affine(condition).chunk(2, dim=1)  # gamma: (B, C), beta: (B, C)
        gamma = gamma[:, :, None, None]  # (B, C, 1, 1)
        beta = beta[:, :, None, None]    # (B, C, 1, 1)
        return self.normalization(inputs) * (1 + gamma) + beta  # (B, C, H, W)


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
        # inputs shape: (B, C_in, H, W), condition shape: (B, condition_dim)
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")  # (B, C_in, 2H, 2W)
        residual = self.skip(residual)  # (B, C_out, 2H, 2W)

        hidden = F.relu(self.norm1(inputs, condition), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")  # (B, C_in, 2H, 2W)
        hidden = self.conv1(hidden)  # (B, C_out, 2H, 2W)
        hidden = self.conv2(F.relu(self.norm2(hidden, condition), inplace=True))
        return hidden + residual  # (B, C_out, 2H, 2W)


class CompactBigGANGenerator(nn.Module):
    """BigGAN-style 32x32 generator with a compact 64-channel base width."""

    def __init__(
        self,
        z_dim=128,
        num_classes=10,
        base_channels=64,
        class_embedding_dim=128,
    ):
        super().__init__()
        self.num_generator_blocks = 3
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
        )  # labels: (B,) -> e_y: (B, class_embedding_dim)
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
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        # noise shape: (B, z_dim), labels shape: (B,)
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        # Split noise into z_0 for the input projection and z_1 through z_3
        # for the three generator blocks.
        latent_chunks = noise.split(self.latent_chunk_dim, dim=1)  # (B, z_dim/4) * 4

        class_condition = self.class_embedding(labels)  # (B, class_embedding_dim)
        block_conditions = [
            torch.cat([class_condition, chunk], dim=1)
            for chunk in latent_chunks[1:]
        ]  # (B, class_embedding_dim + z_dim/4) * 3

        hidden = self.input(latent_chunks[0])  # (B, 8C * 4 * 4)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)     # (B, 8C, 4, 4)
        hidden = self.block1(hidden, block_conditions[0])  # (B, 4C, 8, 8)
        hidden = self.block2(hidden, block_conditions[1])  # (B, 2C, 16, 16)
        hidden = self.attention(hidden)                    # (B, 2C, 16, 16)
        hidden = self.block3(hidden, block_conditions[2])  # (B, C, 32, 32)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))  # (B, 3, 32, 32)


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
        # Flatten each output filter/neuron into one row:
        # Conv2d: (C_out, C_in, k_h, k_w) -> (C_out, C_in * k_h * k_w)
        # Linear: (C_out, C_in) remains unchanged.
        matrix = parameter.flatten(start_dim=1)
        if matrix.shape[0] <= matrix.shape[1]:
            # Row Gram matrix: G = W @ W^T, where G[i, j] = w_i · w_j.
            gram = matrix @ matrix.transpose(0, 1)  # (C_out, C_out)
            # Remove the diagonal, which contains each row's squared L2 norm.
            off_diagonal = gram - torch.diag_embed(
                torch.diagonal(gram)
            )
            # Penalize squared inner products between distinct rows:
            # sum_{i != j} (w_i · w_j)^2.
            penalty = penalty + off_diagonal.square().sum()
        else:
            # For a tall matrix, use the smaller column Gram matrix: W^T @ W.
            column_gram = matrix.transpose(0, 1) @ matrix  # (D, D)
            # Remove the row-Gram diagonal contribution: sum_i ||w_i||_2^4.
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
    return samples  # x ~ N(0, 1), |x| <= t


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
    "BigGANGeneratorResidualBlock",
    "CompactBigGANGenerator",
    "ConditionalBatchNorm2d",
    "initialize_orthogonal_weights",
    "modified_orthogonal_regularization",
    "orthogonal_scratch_minimal",
    "truncated_normal",
]
