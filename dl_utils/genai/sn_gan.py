"""Unconditional CIFAR-10 SN-GAN models and teaching utilities.

The CIFAR-10 experiment in the spectral-normalization paper is an
unsupervised image-generation experiment.  Accordingly, this module contains
no label embeddings, conditional BatchNorm, or projection discriminator.
Those mechanisms first appear in :mod:`dl_utils.genai.sagan`, matching the
class-conditional SAGAN paper.

The generator follows the paper's CIFAR ResNet table: a 128-dimensional
latent vector is projected to a 4x4 feature map and passed through three
upsampling residual blocks.  Spectral normalization is deliberately applied
only to the discriminator; extending it to the generator is introduced by
SAGAN.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


def count_spectral_norm_layers(module):
    """Count modules wrapped by PyTorch's legacy spectral-normalization hook."""
    required_state = ("weight_orig", "weight_u", "weight_v")
    return sum(
        all(hasattr(child, name) for name in required_state)
        for child in module.modules()
    )


def uniform_dequantize_uint8(pixels, *, generator=None):
    """Map uint8 pixels with uniform dequantization to the paper's range."""
    if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
        raise TypeError("pixels must be a torch.uint8 tensor.")
    noise = torch.rand(
        pixels.shape,
        dtype=torch.float32,
        device=pixels.device,
        generator=generator,
    )
    return (pixels.to(torch.float32) + noise) / 128.0 - 1.0


def _initialize_affine_layer(module, gain=1.0):
    """Initialize one affine layer, including a spectral-norm wrapped layer."""
    weight = getattr(module, "weight_orig", None)
    if weight is None:
        weight = module.weight
    nn.init.xavier_uniform_(weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _initialize_resnet_weights(module):
    """Initialize non-residual maps and normalization affine parameters."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        _initialize_affine_layer(module)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _initialize_residual_branches(*blocks):
    """Use the paper implementation's sqrt(2) gain on residual 3x3 maps."""
    for block in blocks:
        _initialize_affine_layer(block.conv1, gain=math.sqrt(2))
        _initialize_affine_layer(block.conv2, gain=math.sqrt(2))


class SNGeneratorResidualBlock(nn.Module):
    """Unconditional pre-activation residual block with 2x upsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # Upsampling changes the representation even when the channel count
        # stays fixed, so the paper's ResBlock uses a learned projection on
        # every shortcut that crosses a resolution boundary.
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(self.norm2(hidden), inplace=True))
        return hidden + residual


class SNGenerator(nn.Module):
    """Paper-aligned unconditional 32x32 SN-GAN generator."""

    def __init__(self, z_dim=128, base_channels=256, image_size=32):
        super().__init__()
        if z_dim < 1:
            raise ValueError("z_dim must be positive.")
        if base_channels < 1:
            raise ValueError("base_channels must be positive.")
        if image_size != 32:
            raise ValueError("image_size must be 32.")

        self.z_dim = z_dim
        self.input = nn.Linear(z_dim, base_channels * 4 * 4)
        self.block1 = SNGeneratorResidualBlock(
            base_channels, base_channels
        )
        self.block2 = SNGeneratorResidualBlock(
            base_channels, base_channels
        )
        self.block3 = SNGeneratorResidualBlock(
            base_channels, base_channels
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.apply(_initialize_resnet_weights)
        _initialize_residual_branches(
            self.block1,
            self.block2,
            self.block3,
        )

    def forward(self, noise):
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        hidden = self.input(noise)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)
        hidden = self.block1(hidden)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
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
    ):
        super().__init__()
        self.first = first
        self.downsample = downsample
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
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
    """Paper-aligned unconditional CIFAR ResNet discriminator.

    Two residual blocks reduce 32x32 inputs to 8x8.  Two more blocks keep the
    resolution and width fixed before ReLU, global sum pooling, and one scalar
    output.  Every learned map is spectrally normalized.
    """

    def __init__(self, base_channels=128):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive.")

        self.feature_channels = base_channels
        self.block1 = SNDiscriminatorResidualBlock(
            3, base_channels, first=True
        )
        self.block2 = SNDiscriminatorResidualBlock(
            base_channels, base_channels
        )
        self.block3 = SNDiscriminatorResidualBlock(
            base_channels, base_channels, downsample=False
        )
        self.block4 = SNDiscriminatorResidualBlock(
            base_channels, base_channels, downsample=False
        )
        self.output = spectral_norm(nn.Linear(base_channels, 1, bias=False))
        self.apply(_initialize_resnet_weights)
        _initialize_residual_branches(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

    def extract_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Expected images with shape (batch, 3, height, width), "
                f"got {tuple(images.shape)}."
            )
        hidden = self.block1(images)
        hidden = self.block2(hidden)
        hidden = self.block3(hidden)
        hidden = self.block4(hidden)
        hidden = F.relu(hidden, inplace=True)
        return hidden.sum(dim=(2, 3))

    def forward(self, images):
        return self.output(self.extract_features(images)).squeeze(1)

    def forward_with_features(self, images):
        """Return scalar scores together with pooled discriminator features."""
        features = self.extract_features(images)
        return self.output(features).squeeze(1), features


def discriminator_diagnostic_sums(
    real_scores,
    fake_scores,
    real_features,
):
    """Return batch sums for hinge, score, and feature-health diagnostics."""
    detached_real_scores = real_scores.detach().float()
    detached_fake_scores = fake_scores.detach().float()
    detached_features = real_features.detach().float()
    if detached_features.ndim != 2:
        raise ValueError("real_features must have shape (batch, feature_dim).")
    return torch.stack(
        [
            (detached_real_scores < 1).sum(dtype=torch.float32),
            (detached_fake_scores > -1).sum(dtype=torch.float32),
            detached_real_scores.sum(),
            detached_fake_scores.sum(),
            detached_real_scores.square().sum(),
            detached_fake_scores.square().sum(),
            (detached_features > 0).float().mean(dim=1).sum(),
            detached_features.square().mean(dim=1).sqrt().sum(),
        ]
    )


@torch.no_grad()
def evaluate_unconditional_images(generated_images, real_images, pool_size=8):
    """Compare unconditional samples with real images using light diagnostics.

    These pooled-pixel statistics are deliberately dependency-free collapse
    checks, not Inception-feature FID or KID.  Both inputs must have shape
    ``[sample, channel, height, width]`` and the same number of samples.
    """
    if generated_images.shape != real_images.shape:
        raise ValueError("Generated and real images must have identical shapes.")
    if generated_images.ndim != 4:
        raise ValueError(
            "Images must have shape (samples, channels, height, width)."
        )
    if generated_images.shape[0] < 2:
        raise ValueError("At least two generated and real images are required.")
    if not 1 <= pool_size <= min(generated_images.shape[-2:]):
        raise ValueError("pool_size must fit inside the image dimensions.")

    generated = generated_images.detach().float().cpu()
    real = real_images.detach().float().cpu()

    def pool(images):
        return F.adaptive_avg_pool2d(images, (pool_size, pool_size)).flatten(1)

    def pairwise_rmse(images):
        return (
            2 * images.var(dim=0, correction=1).mean()
        ).clamp_min(0).sqrt()

    def effective_rank(features):
        centered = features - features.mean(dim=0, keepdim=True)
        spectrum = torch.linalg.svdvals(centered).square()
        if spectrum.sum() <= torch.finfo(spectrum.dtype).eps:
            return spectrum.new_zeros(())
        probabilities = spectrum / spectrum.sum()
        entropy = -(
            probabilities
            * probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ).log()
        ).sum()
        return entropy.exp()

    def frechet_distance(first, second):
        first = first.to(torch.float64)
        second = second.to(torch.float64)
        first_mean = first.mean(dim=0)
        second_mean = second.mean(dim=0)
        first_centered = first - first_mean
        second_centered = second - second_mean
        first_covariance = (
            first_centered.T @ first_centered / (first.shape[0] - 1)
        )
        second_covariance = (
            second_centered.T @ second_centered / (second.shape[0] - 1)
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(first_covariance)
        covariance_root = (
            eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
        ) @ eigenvectors.T
        product = covariance_root @ second_covariance @ covariance_root
        product = (product + product.T) * 0.5
        root_trace = torch.linalg.eigvalsh(product).clamp_min(0).sqrt().sum()
        return (
            (first_mean - second_mean).square().sum()
            + torch.trace(first_covariance)
            + torch.trace(second_covariance)
            - 2 * root_trace
        ).clamp_min(0)

    generated_features = pool(generated)
    real_features = pool(real)
    generated_pairwise = pairwise_rmse(generated)
    real_pairwise = pairwise_rmse(real)
    generated_rank = effective_rank(generated_features)
    real_rank = effective_rank(real_features)
    feature_distances = torch.cdist(
        generated_features, real_features
    ) / math.sqrt(generated_features.shape[1])

    return {
        "generated_pairwise_rmse": generated_pairwise.item(),
        "real_pairwise_rmse": real_pairwise.item(),
        "pairwise_rmse_real_ratio": (
            generated_pairwise / real_pairwise.clamp_min(1e-12)
        ).item(),
        "generated_effective_rank_8x8": generated_rank.item(),
        "real_effective_rank_8x8": real_rank.item(),
        "effective_rank_real_ratio": (
            generated_rank / real_rank.clamp_min(1e-12)
        ).item(),
        "pooled_frechet_distance_8x8": frechet_distance(
            generated_features, real_features
        ).item(),
        "generated_to_real_nearest_rmse_8x8": (
            feature_distances.amin(dim=1).mean().item()
        ),
        "real_to_generated_nearest_rmse_8x8": (
            feature_distances.amin(dim=0).mean().item()
        ),
    }


def discriminator_hinge_loss(real_scores, fake_scores):
    """Return the discriminator hinge loss."""
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_hinge_loss(fake_scores):
    """Return the non-saturating hinge objective used by the generator."""
    return -fake_scores.mean()


def init_spectral_norm_state(weight_orig, eps=1e-12):
    """Initialize unit vectors for the minimal spectral-normalization lesson.

    Accepts 2D ``Linear`` weights and higher-dimensional tensors (for example,
    ``Conv2d`` weights shaped ``[C_out, C_in, kH, kW]``).  Tensors with more
    than two dimensions are flattened to ``[dim0, prod(remaining dims)]``.

    This is a pure-function teaching example, not a replacement for
    :func:`torch.nn.utils.spectral_norm`.  Callers retain the returned ``u``
    and ``v`` across training steps.
    """
    if weight_orig.ndim < 2:
        raise ValueError(
            "init_spectral_norm_state requires at least two-dimensional "
            f"weight, but got {weight_orig.ndim} dimensions"
        )

    with torch.no_grad():
        out_features = weight_orig.size(0)
        in_features = weight_orig[0].numel()
        u = F.normalize(
            weight_orig.new_empty(out_features).normal_(), dim=0, eps=eps
        )
        v = F.normalize(
            weight_orig.new_empty(in_features).normal_(), dim=0, eps=eps
        )
    return u, v


def spectral_norm_scratch_minimal(
    weight_orig,
    u,
    v,
    training,
    eps=1e-12,
):
    """Run one teaching-only power iteration for a weight tensor.

    Returns ``(normalized_weight, next_u, next_v, sigma)``.  The singular
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
            next_v = F.normalize(
                torch.mv(weight_2d.t(), u.detach()), dim=0, eps=eps
            )
            next_u = F.normalize(
                torch.mv(weight_2d, next_v), dim=0, eps=eps
            )
    else:
        next_u = u.detach()
        next_v = v.detach()

    sigma = torch.dot(next_u, torch.mv(weight_2d, next_v))
    normalized_weight = weight_orig / sigma
    return normalized_weight, next_u, next_v, sigma
