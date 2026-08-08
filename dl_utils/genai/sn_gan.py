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
    """Compact 32x32 or 64x64 conditional SN-GAN generator.

    Class information enters once at the generator input. The ``attention``
    attribute is an identity layer so the SAGAN lesson can replace exactly one
    component without changing this forward method. For 32x32 output, the
    fourth upsampling block becomes an identity mapping.
    """

    def __init__(
        self,
        z_dim=120,
        num_classes=10,
        base_channels=32,
        condition_dim=128,
        image_size=32,
    ):
        super().__init__()
        if image_size not in (32, 64):
            raise ValueError("image_size must be 32 or 64.")
        self.z_dim = z_dim
        self.image_size = image_size
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
        self.block4 = (
            SNGeneratorResidualBlock(base_channels, base_channels)
            if image_size == 64
            else nn.Identity()
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        # B: batch size, C: base_channels, H: image_size.
        # noise: (B, z_dim), labels: (B,)
        condition = self.class_embedding(labels)  # (B, condition_dim)
        conditioned_noise = torch.cat(
            [noise, condition], dim=1
        )  # (B, z_dim + condition_dim)
        hidden = self.input(conditioned_noise)  # (B, 8C * 4 * 4)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)  # (B, 8C, 4, 4)
        hidden = self.block1(hidden)  # (B, 4C, 8, 8)
        hidden = self.block2(hidden)  # (B, 2C, 16, 16)
        hidden = self.attention(hidden)  # (B, 2C, 16, 16)
        hidden = self.block3(hidden)  # (B, C, 32, 32)
        hidden = self.block4(hidden)  # (B, C, H, H)
        hidden = self.output_norm(hidden)  # (B, C, H, H)
        hidden = F.relu(hidden, inplace=True)  # (B, C, H, H)
        images = torch.tanh(self.output(hidden))  # (B, 3, H, H)
        return images


class ProjectionSNDiscriminator(nn.Module):
    """Projection discriminator for compact 32x32 or 64x64 SN-GANs."""

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
        # B: batch size, C: base_channels, H: input height and width.
        # images: (B, 3, H, H), labels: (B,)
        hidden = self.block1(images)  # (B, C, H/2, H/2)
        hidden = self.block2(hidden)  # (B, 2C, H/4, H/4)
        hidden = self.attention(hidden)  # (B, 2C, H/4, H/4)
        hidden = self.block3(hidden)  # (B, 4C, H/8, H/8)
        hidden = self.block4(hidden)  # (B, 8C, H/16, H/16)
        hidden = F.relu(hidden, inplace=True)  # (B, 8C, H/16, H/16)
        hidden = hidden.sum(dim=(2, 3))  # (B, 8C)

        unconditional = self.output(hidden).squeeze(1)  # (B,)
        # Projection discriminator: D(x, y) = f(h(x)) + <e(y), h(x)>.
        class_features = self.class_embedding(labels)  # (B, 8C)
        projection = (class_features * hidden).sum(dim=1)  # (B,)
        scores = unconditional + projection  # (B,)
        return scores


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

    Accepts 2D ``Linear`` weights and higher-dimensional tensors (e.g. ``Conv2d``
    weights of shape ``[C_out, C_in, kH, kW]``).  For ``ndim > 2`` the weight
    is flattened to ``[dim0, dim1 * … * dim(n-1)]`` before power iteration, so
    ``u`` is always ``[dim0]`` and ``v`` is always ``[prod(remaining dims)]``.

    This is a pure-function teaching example, not a replacement for
    :func:`torch.nn.utils.spectral_norm`. Callers must retain the returned
    ``u`` and ``v`` across training steps.
    """
    if weight_orig.ndim < 2:
        raise ValueError(
            "init_spectral_norm_state requires at least two-dimensional weight, "
            f"but got {weight_orig.ndim} dimensions"
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

    Supports 2D ``Linear`` weights as well as higher-dimensional tensors
    (e.g. ``Conv2d`` weights of shape ``[C_out, C_in, kH, kW]``).  For
    ``ndim > 2`` the weight is flattened to ``[dim0, dim1 * … * dim(n-1)]``
    before power iteration and ``normalized_weight`` keeps the original shape.

    This is a pure-function teaching implementation, not a replacement for
    :func:`torch.nn.utils.spectral_norm`. Callers must retain the returned
    ``u`` and ``v`` across training steps; this function owns no persistent
    state.

    Returns:
        ``(normalized_weight, next_u, next_v, sigma)``.
    """
    if weight_orig.ndim < 2:
        raise ValueError(
            "spectral_norm_scratch_minimal requires at least two-dimensional "
            f"weight, but got {weight_orig.ndim} dimensions"
        )

    # Flatten to 2D for matrix–vector operations.
    weight_2d = weight_orig.view(weight_orig.size(0), -1)

    if training:
        # Singular-vector state follows the weight without joining autograd.
        with torch.no_grad():
            next_v = F.normalize(
                torch.mv(weight_2d.t(), u.detach()), dim=0, eps=eps
            )
            next_u = F.normalize(
                torch.mv(weight_2d, next_v), dim=0, eps=eps
            )
    else:
        next_u = u.detach()
        next_v = v.detach()

    # Recompute sigma with grad enabled so weight_orig receives gradients.
    sigma = torch.dot(next_u, torch.mv(weight_2d, next_v))
    normalized_weight = weight_orig / sigma
    return normalized_weight, next_u, next_v, sigma
