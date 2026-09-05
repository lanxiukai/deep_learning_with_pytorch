"""Self-Attention GAN models for class-conditional 128x128 images.

SAGAN extends the preceding projection SN-GAN with spectral normalization in
the generator and a non-local attention block at 32x32. The 64-channel model
and five-stage 4-to-128 hierarchy follow the public paper implementation.
"""

from collections.abc import Sequence
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from dl_utils.gan.sn_gan import (
    DISCRIMINATOR_128_BLOCK_RESOLUTIONS,
    GENERATOR_128_BLOCK_RESOLUTIONS,
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    CategoricalConditionalBatchNorm2d,
    SNDiscriminatorResidualBlock,
    SNGeneratorResidualBlock,
    discriminator_128_channels,
    generator_128_channels,
    validate_class_labels,
)


def _initialize_sagan_weights(module):
    """Apply the public implementation's Xavier/zero-bias initialization."""
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.xavier_uniform_(weight)
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            nn.init.zeros_(bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_fixed_class_latent_grid(
    class_indices,
    samples_per_class,
    z_dim,
    device,
    *,
    base_noise=None,
):
    """Repeat fixed latent columns for each requested class index.

    Passing an integer preserves the original convenience API and selects
    ``range(class_indices)``. A sequence can select a small, spread-out subset
    of dataset classes for readable sample grids.
    """
    if isinstance(class_indices, int):
        if class_indices < 1:
            raise ValueError("class_indices must be positive.")
        selected_classes = tuple(range(class_indices))
    elif isinstance(class_indices, Sequence):
        selected_classes = tuple(class_indices)
        if not selected_classes:
            raise ValueError("class_indices must not be empty.")
    else:
        raise TypeError("class_indices must be an integer or a sequence.")

    if base_noise is None:
        base_noise = torch.randn(samples_per_class, z_dim, device=device)
    elif tuple(base_noise.shape) != (samples_per_class, z_dim):
        raise ValueError(
            "base_noise must have shape "
            f"({samples_per_class}, {z_dim}), got {tuple(base_noise.shape)}."
        )
    else:
        base_noise = base_noise.to(device)

    classes = torch.tensor(selected_classes, dtype=torch.long, device=device)
    labels = classes.repeat_interleave(samples_per_class)
    noise = base_noise.repeat(len(selected_classes), 1)
    return noise, labels


class SelfAttention(nn.Module):
    """SAGAN non-local attention with pooled keys and values."""

    def __init__(self, channels):
        super().__init__()
        key_channels = max(1, channels // 8)
        value_channels = max(1, channels // 2)
        self.query = spectral_norm(nn.Conv2d(channels, key_channels, 1))
        self.key = spectral_norm(nn.Conv2d(channels, key_channels, 1))
        self.value = spectral_norm(nn.Conv2d(channels, value_channels, 1))
        self.output = spectral_norm(nn.Conv2d(value_channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("SelfAttention expects a BCHW tensor.")
        batch, _, height, width = inputs.shape
        if min(height, width) < 2:
            raise ValueError("SelfAttention needs spatial dimensions >= 2.")

        query = self.query(inputs).flatten(2).transpose(1, 2)
        key = F.max_pool2d(self.key(inputs), 2).flatten(2)
        attention = torch.softmax(query @ key, dim=-1)
        value = F.max_pool2d(self.value(inputs), 2).flatten(2)
        attended = value @ attention.transpose(1, 2)
        attended = attended.view(batch, -1, height, width)
        return inputs + self.gamma * self.output(attended)


ConditionalBatchNorm2d = CategoricalConditionalBatchNorm2d


class SAGANGeneratorResidualBlock(SNGeneratorResidualBlock):
    """Spectral-normalized conditional residual block with 2x upsampling."""

    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__(
            in_channels,
            out_channels,
            num_classes,
            spectral_normalization=True,
        )


class SAGANGenerator(nn.Module):
    """Paper-oriented 128x128 SAGAN generator with 32x32 attention."""

    def __init__(
        self,
        z_dim=128,
        num_classes=IMAGENETTE_NUM_CLASSES,
        base_channels=64,
        image_size=IMAGENETTE_IMAGE_SIZE,
        attention_resolution=32,
    ):
        super().__init__()
        if z_dim < 1 or base_channels < 1 or num_classes < 1:
            raise ValueError("z_dim, base_channels, and num_classes must be positive.")
        if image_size != IMAGENETTE_IMAGE_SIZE:
            raise ValueError("SAGANGenerator currently implements only 128x128 output.")
        if attention_resolution not in GENERATOR_128_BLOCK_RESOLUTIONS:
            raise ValueError(
                "attention_resolution must be one of "
                f"{GENERATOR_128_BLOCK_RESOLUTIONS}."
            )

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.image_size = image_size
        self.attention_after_block = GENERATOR_128_BLOCK_RESOLUTIONS.index(
            attention_resolution
        )
        channels = generator_128_channels(base_channels)
        self.input = spectral_norm(nn.Linear(z_dim, channels[0] * 4 * 4))
        blocks = [
            SAGANGeneratorResidualBlock(
                in_channels,
                out_channels,
                num_classes,
            )
            for in_channels, out_channels in pairwise(channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        attention_channels = channels[self.attention_after_block + 1]
        self.attention = SelfAttention(attention_channels)
        self.output_norm = nn.BatchNorm2d(channels[-1])
        self.output = spectral_norm(nn.Conv2d(channels[-1], 3, 3, padding=1))
        self.apply(_initialize_sagan_weights)
        for block in blocks:
            nn.init.ones_(block.norm1.gain.weight)
            nn.init.zeros_(block.norm1.bias.weight)
            nn.init.ones_(block.norm2.gain.weight)
            nn.init.zeros_(block.norm2.bias.weight)

    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        validate_class_labels(labels, noise.shape[0])

        hidden = self.input(noise).view(noise.shape[0], -1, 4, 4)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, labels)
            if index == self.attention_after_block:
                hidden = self.attention(hidden)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class SAGANDiscriminator(nn.Module):
    """Wide projection discriminator with configurable attention resolution."""

    def __init__(
        self,
        num_classes=IMAGENETTE_NUM_CLASSES,
        base_channels=64,
        image_size=IMAGENETTE_IMAGE_SIZE,
        attention_resolution=32,
    ):
        super().__init__()
        if base_channels < 1 or num_classes < 1:
            raise ValueError("base_channels and num_classes must be positive.")
        if image_size != IMAGENETTE_IMAGE_SIZE:
            raise ValueError(
                "SAGANDiscriminator currently implements only 128x128 input."
            )
        if attention_resolution not in DISCRIMINATOR_128_BLOCK_RESOLUTIONS:
            raise ValueError("attention_resolution must be one of (4, 8, 16, 32, 64).")

        self.num_classes = num_classes
        self.image_size = image_size
        self.attention_after_block = DISCRIMINATOR_128_BLOCK_RESOLUTIONS.index(
            attention_resolution
        )
        channels = discriminator_128_channels(base_channels)
        self.blocks = nn.ModuleList(
            [
                SNDiscriminatorResidualBlock(
                    3,
                    channels[0],
                    first=True,
                    wide=True,
                )
            ]
            + [
                SNDiscriminatorResidualBlock(
                    in_channels,
                    out_channels,
                    wide=True,
                )
                for in_channels, out_channels in pairwise(channels[:-1])
            ]
            + [
                SNDiscriminatorResidualBlock(
                    channels[-2],
                    channels[-1],
                    downsample=False,
                    wide=True,
                )
            ]
        )
        attention_channels = channels[self.attention_after_block]
        self.attention = SelfAttention(attention_channels)
        self.feature_channels = channels[-1]
        self.output = spectral_norm(nn.Linear(self.feature_channels, 1))
        self.class_embedding = spectral_norm(
            nn.Embedding(num_classes, self.feature_channels)
        )
        self.apply(_initialize_sagan_weights)

    def extract_features(self, images):
        if (
            images.ndim != 4
            or images.shape[1] != 3
            or tuple(images.shape[-2:]) != (self.image_size, self.image_size)
        ):
            raise ValueError(
                "Expected images with shape "
                f"(batch, 3, {self.image_size}, {self.image_size}), "
                f"got {tuple(images.shape)}."
            )
        hidden = images
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index == self.attention_after_block:
                hidden = self.attention(hidden)
        return F.relu(hidden, inplace=True).sum(dim=(2, 3))

    def forward(self, images, labels):
        validate_class_labels(labels, images.shape[0])
        features = self.extract_features(images)
        unconditional = self.output(features).squeeze(1)
        projection = (self.class_embedding(labels) * features).sum(dim=1)
        return unconditional + projection


__all__ = [
    "ConditionalBatchNorm2d",
    "SAGANDiscriminator",
    "SAGANGenerator",
    "SAGANGeneratorResidualBlock",
    "SelfAttention",
    "make_fixed_class_latent_grid",
]
