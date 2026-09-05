"""128x128 SN-GAN models and spectral-normalization teaching tools.

The default network adapts the public projection SN-GAN architecture to the
10-class Imagenette teaching dataset: a 128-dimensional latent vector, a
64-channel ResNet stem, categorical conditional BatchNorm in the generator,
and a projection discriminator. The generator is intentionally not spectrally
normalized; that extension is introduced by the following SAGAN lesson.
"""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from dl_utils.data.imagenette import (
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    uniform_dequantize_uint8,
)


def _maybe_spectral_norm(module, enabled):
    return spectral_norm(module) if enabled else module


def _initialize_affine_layer(module, gain=1.0):
    """Initialize one affine layer, including a spectral-norm wrapped layer."""
    weight = getattr(module, "weight_orig", None)
    if weight is None:
        weight = module.weight
    nn.init.xavier_uniform_(weight, gain=gain)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _initialize_resnet_weights(module):
    """Initialize non-residual maps and normalization affine parameters."""
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        _initialize_affine_layer(module)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _initialize_residual_branches(*blocks):
    """Use the reference implementation's sqrt(2) residual-branch gain."""
    for block in blocks:
        _initialize_affine_layer(block.conv1, gain=math.sqrt(2))
        _initialize_affine_layer(block.conv2, gain=math.sqrt(2))


def validate_class_labels(labels, batch_size):
    """Validate the shape and dtype shared by conditional GAN forwards."""
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError("labels must have shape (batch,).")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long.")


class CategoricalConditionalBatchNorm2d(nn.Module):
    """BatchNorm with one learned gain and bias vector per class."""

    def __init__(self, channels, num_classes):
        super().__init__()
        self.normalization = nn.BatchNorm2d(channels, affine=False)
        self.gain = nn.Embedding(num_classes, channels)
        self.bias = nn.Embedding(num_classes, channels)
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, inputs, labels):
        gain = self.gain(labels)[:, :, None, None]
        bias = self.bias(labels)[:, :, None, None]
        return self.normalization(inputs) * gain + bias


class SNGeneratorResidualBlock(nn.Module):
    """Conditional pre-activation residual block with 2x upsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        *,
        spectral_normalization=False,
    ):
        super().__init__()
        self.norm1 = CategoricalConditionalBatchNorm2d(
            in_channels,
            num_classes,
        )
        self.norm2 = CategoricalConditionalBatchNorm2d(
            out_channels,
            num_classes,
        )
        self.conv1 = _maybe_spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            spectral_normalization,
        )
        self.conv2 = _maybe_spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            spectral_normalization,
        )
        self.skip = _maybe_spectral_norm(
            nn.Conv2d(in_channels, out_channels, 1),
            spectral_normalization,
        )

    def forward(self, inputs, labels):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs, labels), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(self.norm2(hidden, labels), inplace=True))
        return hidden + residual


class SNGenerator(nn.Module):
    """Projection SN-GAN's class-conditional 128x128 generator."""

    def __init__(
        self,
        z_dim=128,
        num_classes=IMAGENETTE_NUM_CLASSES,
        base_channels=64,
        image_size=IMAGENETTE_IMAGE_SIZE,
    ):
        super().__init__()
        if z_dim < 1 or base_channels < 1 or num_classes < 1:
            raise ValueError("z_dim, base_channels, and num_classes must be positive.")
        if image_size != IMAGENETTE_IMAGE_SIZE:
            raise ValueError("SNGenerator currently implements only 128x128 output.")
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.image_size = image_size
        channels = [
            base_channels * 16,
            base_channels * 16,
            base_channels * 8,
            base_channels * 4,
            base_channels * 2,
            base_channels,
        ]
        self.input = nn.Linear(z_dim, channels[0] * 4 * 4)
        blocks = [
            SNGeneratorResidualBlock(
                in_channels,
                out_channels,
                num_classes,
            )
            for in_channels, out_channels in pairwise(channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.output_norm = nn.BatchNorm2d(channels[-1])
        self.output = nn.Conv2d(channels[-1], 3, 3, padding=1)
        self.apply(_initialize_resnet_weights)
        _initialize_residual_branches(*self.blocks)
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
        for block in self.blocks:
            hidden = block(hidden, labels)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class SNDiscriminatorResidualBlock(nn.Module):
    """Spectral-normalized residual block with optional downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        first=False,
        downsample=True,
        wide=False,
    ):
        super().__init__()
        self.first = first
        self.downsample = downsample
        hidden_channels = out_channels if first or wide else in_channels
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )
        needs_projection = in_channels != out_channels or downsample
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if needs_projection
            else nn.Identity()
        )

    def forward(self, inputs):
        hidden = inputs if self.first else F.relu(inputs, inplace=False)
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(hidden, inplace=True))
        if self.downsample:
            hidden = F.avg_pool2d(hidden, 2)

        residual = self.skip(inputs)
        if self.downsample:
            residual = F.avg_pool2d(residual, 2)
        return hidden + residual


class SNDiscriminator(nn.Module):
    """128x128 spectral-normalized projection discriminator."""

    def __init__(
        self,
        num_classes=IMAGENETTE_NUM_CLASSES,
        base_channels=64,
        image_size=IMAGENETTE_IMAGE_SIZE,
    ):
        super().__init__()
        if base_channels < 1 or num_classes < 1:
            raise ValueError("base_channels and num_classes must be positive.")
        if image_size != IMAGENETTE_IMAGE_SIZE:
            raise ValueError("SNDiscriminator currently implements only 128x128 input.")
        self.num_classes = num_classes
        self.image_size = image_size
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 16,
        ]
        self.blocks = nn.ModuleList(
            [SNDiscriminatorResidualBlock(3, channels[0], first=True)]
            + [
                SNDiscriminatorResidualBlock(in_channels, out_channels)
                for in_channels, out_channels in pairwise(channels[:-1])
            ]
            + [
                SNDiscriminatorResidualBlock(
                    channels[-2],
                    channels[-1],
                    downsample=False,
                )
            ]
        )
        self.feature_channels = channels[-1]
        self.output = spectral_norm(nn.Linear(self.feature_channels, 1))
        self.class_embedding = spectral_norm(
            nn.Embedding(num_classes, self.feature_channels)
        )
        self.apply(_initialize_resnet_weights)
        _initialize_residual_branches(*self.blocks)

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
        for block in self.blocks:
            hidden = block(hidden)
        return F.relu(hidden, inplace=True).sum(dim=(2, 3))

    def forward(self, images, labels):
        validate_class_labels(labels, images.shape[0])
        features = self.extract_features(images)
        unconditional = self.output(features).squeeze(1)
        projection = (self.class_embedding(labels) * features).sum(dim=1)
        return unconditional + projection


def discriminator_hinge_loss(real_scores, fake_scores):
    """Return the discriminator hinge loss."""
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_hinge_loss(fake_scores):
    """Return the generator hinge objective."""
    return -fake_scores.mean()


def init_spectral_norm_state(weight_orig, eps=1e-12):
    """Initialize unit vectors for the minimal spectral-normalization lesson.

    Accepts 2D ``Linear`` weights and higher-dimensional tensors. Tensors with
    more than two dimensions are flattened to ``[dim0, prod(other dims)]``.
    """
    if weight_orig.ndim < 2:
        raise ValueError(
            "init_spectral_norm_state requires at least two-dimensional "
            f"weight, but got {weight_orig.ndim} dimensions"
        )

    with torch.no_grad():
        out_features = weight_orig.size(0)
        in_features = weight_orig[0].numel()
        u = F.normalize(weight_orig.new_empty(out_features).normal_(), dim=0, eps=eps)
        v = F.normalize(weight_orig.new_empty(in_features).normal_(), dim=0, eps=eps)
    return u, v


def spectral_norm_scratch_minimal(
    weight_orig,
    u,
    v,
    training,
    eps=1e-12,
):
    """Run one teaching-only power iteration for a weight tensor.

    Returns ``(normalized_weight, next_u, next_v, sigma)``. The singular
    vectors are updated without autograd, while ``sigma`` is recomputed with
    gradients enabled so the original weight receives gradients.
    """
    if weight_orig.ndim < 2:
        raise ValueError(
            "spectral_norm_scratch_minimal requires at least two-dimensional "
            f"weight, but got {weight_orig.ndim} dimensions"
        )

    weight_2d = weight_orig.reshape(weight_orig.size(0), -1)
    expected_u_shape = (weight_2d.shape[0],)
    expected_v_shape = (weight_2d.shape[1],)
    if tuple(u.shape) != expected_u_shape or tuple(v.shape) != expected_v_shape:
        raise ValueError(
            "u and v shapes must match the flattened weight: expected "
            f"{expected_u_shape} and {expected_v_shape}, got "
            f"{tuple(u.shape)} and {tuple(v.shape)}."
        )

    if training:
        with torch.no_grad():
            next_v = F.normalize(torch.mv(weight_2d.t(), u.detach()), dim=0, eps=eps)
            next_u = F.normalize(torch.mv(weight_2d, next_v), dim=0, eps=eps)
    else:
        next_u = u.detach()
        next_v = v.detach()

    sigma = torch.dot(next_u, torch.mv(weight_2d, next_v))
    normalized_weight = weight_orig / sigma
    return normalized_weight, next_u, next_v, sigma


__all__ = [
    "IMAGENETTE_IMAGE_SIZE",
    "IMAGENETTE_NUM_CLASSES",
    "CategoricalConditionalBatchNorm2d",
    "SNDiscriminator",
    "SNDiscriminatorResidualBlock",
    "SNGenerator",
    "SNGeneratorResidualBlock",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "init_spectral_norm_state",
    "spectral_norm_scratch_minimal",
    "uniform_dequantize_uint8",
    "validate_class_labels",
]
