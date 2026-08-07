"""Conditional SN-GAN blocks, models, losses, and image helpers.

This compact baseline keeps class conditioning fixed across the SN-GAN,
SAGAN, and BigGAN lessons. The generator concatenates one class embedding to
the latent vector, while the discriminator uses projection conditioning.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class SNGeneratorResidualBlock(nn.Module):
    """Spectral-normalized residual block that upsamples by a factor of two."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
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

    def forward(self, inputs):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(self.norm2(hidden), inplace=True))
        return hidden + residual


class SNDiscriminatorResidualBlock(nn.Module):
    """Spectral-normalized residual block that downsamples by a factor of two."""

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


class SNGenerator(nn.Module):
    """Compact 64x64 conditional SN-GAN generator.

    Class information enters once at the generator input. The ``attention``
    attribute is an identity layer so the SAGAN lesson can replace exactly one
    component without changing this forward method.
    """

    def __init__(
        self,
        z_dim=120,
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
        self.block1 = SNGeneratorResidualBlock(
            base_channels * 8, base_channels * 4
        )
        self.block2 = SNGeneratorResidualBlock(
            base_channels * 4, base_channels * 2
        )
        self.attention = nn.Identity()
        self.block3 = SNGeneratorResidualBlock(
            base_channels * 2, base_channels
        )
        self.block4 = SNGeneratorResidualBlock(base_channels, base_channels)
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        # noise shape: (B, 120), labels shape: (B,)
        condition = self.class_embedding(labels)  # (B, 128)
        hidden = self.input(torch.cat([noise, condition], dim=1))  # (B, 248) -> (B, 4096)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)  # (B, 256, 4, 4)
        hidden = self.block1(hidden)     # (B, 128, 8, 8)
        hidden = self.block2(hidden)     # (B, 64, 16, 16)
        hidden = self.attention(hidden)  # (B, 64, 16, 16)
        hidden = self.block3(hidden)     # (B, 32, 32, 32)
        hidden = self.block4(hidden)     # (B, 32, 64, 64)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))  # (B, 3, 64, 64)


class ProjectionSNDiscriminator(nn.Module):
    """Compact 64x64 SN-GAN discriminator with projection conditioning."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        channels = base_channels
        self.block1 = SNDiscriminatorResidualBlock(
            3, channels, first=True
        )
        self.block2 = SNDiscriminatorResidualBlock(channels, channels * 2)
        self.attention = nn.Identity()
        self.block3 = SNDiscriminatorResidualBlock(
            channels * 2, channels * 4
        )
        self.block4 = SNDiscriminatorResidualBlock(
            channels * 4, channels * 8
        )
        self.output = spectral_norm(nn.Linear(channels * 8, 1))
        self.class_embedding = spectral_norm(
            nn.Embedding(num_classes, channels * 8)
        )

    def forward(self, images, labels):
        # images shape: (B, 3, 64, 64)，labels shape: (B,)
        hidden = self.block1(images)     # (B, 32, 32, 32)
        hidden = self.block2(hidden)     # (B, 64, 16, 16)
        hidden = self.attention(hidden)  # (B, 64, 16, 16)
        hidden = self.block3(hidden)     # (B, 128, 8, 8)
        hidden = self.block4(hidden)     # (B, 256, 4, 4)
        hidden = F.relu(hidden, inplace=True).sum(dim=(2, 3))  # (B, 256)

        unconditional = self.output(hidden).squeeze(1)
        # Projection discriminator: D(x, y) = f(h(x)) + <e(y), h(x)>.
        projection = (self.class_embedding(labels) * hidden).sum(dim=1)
        return unconditional + projection  # (B,)


def discriminator_hinge_loss(real_scores, fake_scores):
    """Return the discriminator hinge loss."""
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_hinge_loss(fake_scores):
    """Return the non-saturating hinge objective used by the generator."""
    return -fake_scores.mean()


def denormalize(images):
    """Map images from [-1, 1] to [0, 1]."""
    return images.mul(0.5).add(0.5).clamp(0, 1)


def init_spectral_norm_state(weight_orig, eps=1e-12):
    """Initialize unit vectors for the minimal spectral-normalization lesson.

    This helper accepts only a two-dimensional ``Linear`` weight and belongs
    to a pure-function teaching example. It is not a replacement for
    :func:`torch.nn.utils.spectral_norm`. Callers must retain the returned
    ``u`` and ``v`` across training steps.
    """
    if weight_orig.ndim != 2:
        raise ValueError(
            "init_spectral_norm_state requires a two-dimensional Linear "
            f"weight, but got {weight_orig.ndim} dimensions"
        )

    with torch.no_grad():
        height, width = weight_orig.shape
        u = F.normalize(weight_orig.new_empty(height).normal_(), dim=0, eps=eps)
        v = F.normalize(weight_orig.new_empty(width).normal_(), dim=0, eps=eps)
    return u, v


def spectral_norm_scratch_minimal(
    weight_orig,
    u,
    v,
    training,
    eps=1e-12,
):
    """Run one teaching-only power iteration for a 2D ``Linear`` weight.

    This is a two-dimensional, pure-function teaching implementation, not a
    replacement for :func:`torch.nn.utils.spectral_norm`. Callers must retain
    the returned ``u`` and ``v`` across training steps; this function owns no
    persistent state.

    Returns:
        ``(normalized_weight, next_u, next_v, sigma)``.
    """
    if weight_orig.ndim != 2:
        raise ValueError(
            "spectral_norm_scratch_minimal requires a two-dimensional Linear "
            f"weight, but got {weight_orig.ndim} dimensions"
        )

    if training:
        # Singular-vector state follows the weight without joining autograd.
        with torch.no_grad():
            next_v = F.normalize(
                torch.mv(weight_orig.t(), u.detach()), dim=0, eps=eps
            )
            next_u = F.normalize(
                torch.mv(weight_orig, next_v), dim=0, eps=eps
            )
    else:
        next_u = u.detach()
        next_v = v.detach()

    # Recompute sigma with grad enabled so weight_orig receives gradients.
    sigma = torch.dot(next_u, torch.mv(weight_orig, next_v))
    normalized_weight = weight_orig / sigma
    return normalized_weight, next_u, next_v, sigma
