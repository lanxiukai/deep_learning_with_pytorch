"""Model blocks shared by the compact SN-GAN/SAGAN/BigGAN lesson."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """SAGAN non-local attention with a learnable residual scale."""

    def __init__(self, channels):
        super().__init__()
        attention_channels = max(1, channels // 8)
        self.query = spectral_norm(nn.Conv2d(channels, attention_channels, 1))
        self.key = spectral_norm(nn.Conv2d(channels, attention_channels, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.output = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        query = self.query(inputs).flatten(2).transpose(1, 2)
        key = self.key(inputs).flatten(2)
        attention = torch.softmax(query @ key, dim=-1)
        value = self.value(inputs).flatten(2)
        attended = value @ attention.transpose(1, 2)
        attended = attended.view(batch, channels, height, width)
        return inputs + self.gamma * self.output(attended)


class ConditionalBatchNorm2d(nn.Module):
    """BigGAN-style BatchNorm whose scale and bias depend on the class."""

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


class GeneratorResidualBlock(nn.Module):
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


class ConditionalGenerator(nn.Module):
    """64x64 residual generator with shared class conditioning."""

    def __init__(
        self,
        z_dim=128,
        num_classes=10,
        base_channels=32,
        condition_dim=128,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.class_embedding = nn.Embedding(num_classes, condition_dim)
        self.input = spectral_norm(
            nn.Linear(z_dim + condition_dim, base_channels * 8 * 4 * 4)
        )
        self.block1 = GeneratorResidualBlock(
            base_channels * 8, base_channels * 4, condition_dim
        )
        self.block2 = GeneratorResidualBlock(
            base_channels * 4, base_channels * 2, condition_dim
        )
        self.attention = SelfAttention(base_channels * 2)
        self.block3 = GeneratorResidualBlock(
            base_channels * 2, base_channels, condition_dim
        )
        self.block4 = GeneratorResidualBlock(
            base_channels, base_channels, condition_dim
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        condition = self.class_embedding(labels)
        hidden = self.input(torch.cat([noise, condition], dim=1))
        hidden = hidden.view(noise.shape[0], -1, 4, 4)
        hidden = self.block1(hidden, condition)
        hidden = self.block2(hidden, condition)
        hidden = self.attention(hidden)
        hidden = self.block3(hidden, condition)
        hidden = self.block4(hidden, condition)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.first = first
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, inputs):
        hidden = inputs if self.first else F.relu(inputs, inplace=False)
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(hidden, inplace=True))
        hidden = F.avg_pool2d(hidden, 2)
        residual = F.avg_pool2d(self.skip(inputs), 2)
        return hidden + residual


class ProjectionDiscriminator(nn.Module):
    """Residual discriminator with class/feature inner-product projection."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        c = base_channels
        self.block1 = DiscriminatorResidualBlock(3, c, first=True)
        self.block2 = DiscriminatorResidualBlock(c, c * 2)
        self.attention = SelfAttention(c * 2)
        self.block3 = DiscriminatorResidualBlock(c * 2, c * 4)
        self.block4 = DiscriminatorResidualBlock(c * 4, c * 8)
        self.output = spectral_norm(nn.Linear(c * 8, 1))
        self.class_embedding = spectral_norm(nn.Embedding(num_classes, c * 8))

    def forward(self, images, labels):
        hidden = self.block1(images)
        hidden = self.block2(hidden)
        hidden = self.attention(hidden)
        hidden = self.block3(hidden)
        hidden = self.block4(hidden)
        hidden = F.relu(hidden, inplace=True).sum(dim=(2, 3))
        unconditional = self.output(hidden).squeeze(1)
        projection = (self.class_embedding(labels) * hidden).sum(dim=1)
        return unconditional + projection


def truncated_normal(shape, truncation, device):
    """Rejection-resample every N(0,1) coordinate outside the truncation."""
    if truncation <= 0:
        raise ValueError("truncation must be positive")
    samples = torch.randn(shape, device=device)
    invalid = samples.abs() > truncation
    while invalid.any():
        samples[invalid] = torch.randn_like(samples[invalid])
        invalid = samples.abs() > truncation
    return samples


def denormalize(images):
    return images.mul(0.5).add(0.5).clamp(0, 1)
