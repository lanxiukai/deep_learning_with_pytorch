"""Conditional SN-GAN blocks, models, losses, and evaluation helpers.

The compact SN-GAN and SAGAN generators use block-local class embeddings for
conditional BatchNorm. SN-GAN can additionally feed the complete latent vector
to every conditional BatchNorm block; the BigGAN lesson remains distinct by
using one shared class embedding and hierarchical latent chunks. All
discriminators use projection conditioning.
"""

import math

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


def make_fixed_class_latent_grid(
    num_classes,
    samples_per_class,
    z_dim,
    device,
    *,
    base_noise=None,
):
    """Return noise and labels whose latent columns repeat across classes."""
    if num_classes < 1 or samples_per_class < 1 or z_dim < 1:
        raise ValueError(
            "num_classes, samples_per_class, and z_dim must be positive."
        )
    if base_noise is None:
        base_noise = torch.randn(samples_per_class, z_dim, device=device)
    elif tuple(base_noise.shape) != (samples_per_class, z_dim):
        raise ValueError(
            "base_noise must have shape "
            f"({samples_per_class}, {z_dim}), got {tuple(base_noise.shape)}."
        )
    else:
        base_noise = base_noise.to(device)

    labels = torch.arange(num_classes, device=device).repeat_interleave(
        samples_per_class
    )
    noise = base_noise.repeat(num_classes, 1)
    return noise, labels


def cyclically_mismatched_labels(labels, num_classes):
    """Map every class to a guaranteed-different comparison class."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 for mismatched labels.")
    return (labels + 1) % num_classes


def count_spectral_norm_layers(module):
    """Count modules wrapped by PyTorch's legacy spectral-normalization hook."""
    required_state = ("weight_orig", "weight_u", "weight_v")
    return sum(
        all(hasattr(child, name) for name in required_state)
        for child in module.modules()
    )


def discriminator_diagnostic_sums(
    real_scores,
    fake_scores,
    correct_projection,
    mismatched_projection,
    real_features=None,
):
    """Return summed discriminator health statistics for one batch.

    The first three values preserve the original teaching API: real and fake
    hinge-active counts followed by the correct-minus-mismatched projection
    gap. Supplying ``real_features`` appends score moments plus per-example
    feature activity and RMS magnitude, which makes a discriminator stuck at
    the neutral hinge value visible in saved metrics.
    """
    detached_real_scores = real_scores.detach().float()
    detached_fake_scores = fake_scores.detach().float()
    values = [
        (detached_real_scores < 1).sum(dtype=torch.float32),
        (detached_fake_scores > -1).sum(dtype=torch.float32),
        (
            correct_projection.detach().float()
            - mismatched_projection.detach().float()
        ).sum(),
    ]
    if real_features is not None:
        detached_features = real_features.detach().float()
        if detached_features.ndim != 2:
            raise ValueError(
                "real_features must have shape (batch, feature_dim)."
            )
        values.extend(
            [
                detached_real_scores.sum(),
                detached_fake_scores.sum(),
                detached_real_scores.square().sum(),
                detached_fake_scores.square().sum(),
                (detached_features > 0).float().mean(dim=1).sum(),
                detached_features.square().mean(dim=1).sqrt().sum(),
            ]
        )
    return torch.stack(values)


class SNConditionalBatchNorm2d(nn.Module):
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


class SNGeneratorResidualBlock(nn.Module):
    """Conditional residual block that upsamples by a factor of two.

    When ``z_dim`` is supplied, the complete latent vector is concatenated to
    the block-local class embedding before both conditional BatchNorm affine
    maps. This is a small skip-conditioning path, not BigGAN's hierarchical
    latent split.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        condition_dim,
        z_dim=None,
    ):
        super().__init__()
        if z_dim is not None and z_dim < 1:
            raise ValueError("z_dim must be positive when supplied.")
        self.z_dim = z_dim
        self.class_embedding = nn.Embedding(num_classes, condition_dim)
        block_condition_dim = condition_dim + (z_dim or 0)
        self.norm1 = SNConditionalBatchNorm2d(
            in_channels, block_condition_dim
        )
        self.norm2 = SNConditionalBatchNorm2d(
            out_channels, block_condition_dim
        )
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

    def forward(self, inputs, labels, noise=None):
        condition = self.class_embedding(labels)
        if self.z_dim is not None:
            if noise is None or tuple(noise.shape) != (
                inputs.shape[0],
                self.z_dim,
            ):
                actual_shape = None if noise is None else tuple(noise.shape)
                raise ValueError(
                    "Expected block noise with shape "
                    f"({inputs.shape[0]}, {self.z_dim}), got {actual_shape}."
                )
            condition = torch.cat([condition, noise], dim=1)
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs, condition), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(
            F.relu(self.norm2(hidden, condition), inplace=True)
        )
        return hidden + residual


class SNDiscriminatorResidualBlock(nn.Module):
    """Spectral-normalized residual block with optional downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
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
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if in_channels != out_channels
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


class SNGenerator(nn.Module):
    """Compact 32x32 conditional SN-GAN generator.

    Noise initializes the 4x4 feature map. Each residual block owns a class
    embedding that conditions its BatchNorm affine parameters; optional latent
    reinjection also supplies the complete z vector to those affine maps. The
    ``attention`` attribute is an identity layer so the SAGAN lesson can
    replace exactly one component without changing this forward method.
    """

    def __init__(
        self,
        z_dim=120,
        num_classes=10,
        base_channels=32,
        condition_dim=128,
        image_size=32,
        latent_reinjection=False,
    ):
        super().__init__()
        if image_size != 32:
            raise ValueError("image_size must be 32.")
        self.z_dim = z_dim
        block_z_dim = z_dim if latent_reinjection else None
        self.input = spectral_norm(
            nn.Linear(z_dim, base_channels * 8 * 4 * 4)
        )
        self.block1 = SNGeneratorResidualBlock(
            base_channels * 8,
            base_channels * 4,
            num_classes,
            condition_dim,
            block_z_dim,
        )
        self.block2 = SNGeneratorResidualBlock(
            base_channels * 4,
            base_channels * 2,
            num_classes,
            condition_dim,
            block_z_dim,
        )
        self.attention = nn.Identity()
        self.block3 = SNGeneratorResidualBlock(
            base_channels * 2,
            base_channels,
            num_classes,
            condition_dim,
            block_z_dim,
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        # B: batch size, C: base_channels, H: image_size.
        # noise: (B, z_dim), labels: (B,)
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        hidden = self.input(noise)  # (B, 8C * 4 * 4)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)  # (B, 8C, 4, 4)
        hidden = self.block1(hidden, labels, noise)  # (B, 4C, 8, 8)
        hidden = self.block2(hidden, labels, noise)  # (B, 2C, 16, 16)
        hidden = self.attention(hidden)  # (B, 2C, 16, 16)
        hidden = self.block3(hidden, labels, noise)  # (B, C, 32, 32)
        hidden = self.output_norm(hidden)  # (B, C, 32, 32)
        hidden = F.relu(hidden, inplace=True)  # (B, C, 32, 32)
        images = torch.tanh(self.output(hidden))  # (B, 3, 32, 32)
        return images


class ProjectionSNDiscriminator(nn.Module):
    """Projection discriminator with the paper's CIFAR ResNet topology.

    The first two blocks downsample 32x32 inputs to 8x8. Two same-resolution
    residual blocks then refine a constant-width feature map before global sum
    pooling, matching the discriminator structure used for CIFAR experiments
    more closely than repeatedly reducing the map to 2x2.
    """

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        channels = base_channels * 4
        self.feature_channels = channels
        self.block1 = SNDiscriminatorResidualBlock(
            3, channels, first=True
        )
        self.block2 = SNDiscriminatorResidualBlock(channels, channels)
        self.attention = nn.Identity()
        self.block3 = SNDiscriminatorResidualBlock(
            channels, channels, downsample=False
        )
        self.block4 = SNDiscriminatorResidualBlock(
            channels, channels, downsample=False
        )
        self.output = spectral_norm(nn.Linear(channels, 1))
        self.class_embedding = spectral_norm(
            nn.Embedding(num_classes, channels)
        )

    def _extract_features(self, images):
        # B: batch size, C: base_channels, H: input height and width.
        # images: (B, 3, H, H)
        hidden = self.block1(images)  # (B, 4C, H/2, H/2)
        hidden = self.block2(hidden)  # (B, 4C, H/4, H/4)
        hidden = self.attention(hidden)  # (B, 4C, H/4, H/4)
        hidden = self.block3(hidden)  # (B, 4C, H/4, H/4)
        hidden = self.block4(hidden)  # (B, 4C, H/4, H/4)
        hidden = F.relu(hidden, inplace=True)  # (B, 4C, H/4, H/4)
        hidden = hidden.sum(dim=(2, 3))  # (B, 4C)
        return hidden

    def forward(self, images, labels):
        hidden = self._extract_features(images)
        unconditional = self.output(hidden).squeeze(1)  # (B,)
        # Projection discriminator: D(x, y) = f(h(x)) + <e(y), h(x)>.
        class_features = self.class_embedding(labels)  # (B, 4C)
        projection = (class_features * hidden).sum(dim=1)  # (B,)
        scores = unconditional + projection  # (B,)
        return scores

    def forward_with_projection_diagnostics(
        self,
        images,
        labels,
        mismatched_labels,
        *,
        return_features=False,
    ):
        """Score images and compare correct versus mismatched projections."""
        if labels.shape != mismatched_labels.shape:
            raise ValueError(
                "labels and mismatched_labels must have the same shape."
            )

        hidden = self._extract_features(images)
        unconditional = self.output(hidden).squeeze(1)
        paired_labels = torch.cat([labels, mismatched_labels])
        paired_features = self.class_embedding(paired_labels)
        correct_features, mismatched_features = paired_features.chunk(
            2, dim=0
        )
        correct_projection = (correct_features * hidden).sum(dim=1)
        mismatched_projection = (mismatched_features * hidden).sum(dim=1)
        scores = unconditional + correct_projection
        diagnostics = (scores, correct_projection, mismatched_projection)
        if return_features:
            return (*diagnostics, hidden)
        return diagnostics


@torch.no_grad()
def evaluate_class_conditional_images(
    generated_images,
    real_images,
    pool_size=8,
):
    """Return aggregate and per-class collapse diagnostics against real data.

    Images must be shaped ``[class, sample, channel, height, width]``. The
    pooled Fréchet and nearest-neighbor values are dependency-free diagnostics,
    not Inception-feature FID/KID.
    """
    if generated_images.shape != real_images.shape:
        raise ValueError("Generated and real images must have identical shapes.")
    if generated_images.ndim != 5:
        raise ValueError(
            "Images must have shape (classes, samples, channels, height, width)."
        )
    num_classes, samples_per_class = generated_images.shape[:2]
    if num_classes < 2 or samples_per_class < 2:
        raise ValueError("At least two classes and two samples are required.")
    if not 1 <= pool_size <= min(generated_images.shape[-2:]):
        raise ValueError("pool_size must fit inside the image dimensions.")

    generated = generated_images.detach().float().cpu()
    real = real_images.detach().float().cpu()

    def pool(images):
        return F.adaptive_avg_pool2d(
            images.flatten(0, 1), (pool_size, pool_size)
        ).reshape(num_classes, samples_per_class, -1)

    def diversity(images, features):
        pairwise_rmse = (
            2 * images.var(dim=1, correction=1).flatten(1).mean(dim=1)
        ).sqrt()
        centered = features - features.mean(dim=1, keepdim=True)
        ranks = []
        for class_features in centered:
            spectrum = torch.linalg.svdvals(class_features).square()
            if spectrum.sum() <= torch.finfo(spectrum.dtype).eps:
                ranks.append(spectrum.new_zeros(()))
                continue
            probabilities = spectrum / spectrum.sum()
            entropy = -(
                probabilities
                * probabilities.clamp_min(
                    torch.finfo(probabilities.dtype).tiny
                ).log()
            ).sum()
            ranks.append(entropy.exp())
        ranks = torch.stack(ranks)
        between = features.mean(dim=1).var(dim=0, correction=1).mean()
        within = features.var(dim=1, correction=1).mean()
        centroid_ratio = (between / within.clamp_min(1e-12)).sqrt()
        return pairwise_rmse, ranks, centroid_ratio

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
    generated_pairwise, generated_ranks, generated_centroid_ratio = diversity(
        generated, generated_features
    )
    real_pairwise, real_ranks, real_centroid_ratio = diversity(
        real, real_features
    )

    feature_distances = torch.cdist(
        generated_features, real_features
    ) / math.sqrt(generated_features.shape[-1])
    generated_to_real = feature_distances.amin(dim=2).mean(dim=1)
    real_to_generated = feature_distances.amin(dim=1).mean(dim=1)
    class_frechet = torch.stack(
        [
            frechet_distance(generated_features[index], real_features[index])
            for index in range(num_classes)
        ]
    )
    global_frechet = frechet_distance(
        generated_features.flatten(0, 1), real_features.flatten(0, 1)
    )
    label_effect = generated.var(dim=0, correction=1).mean().sqrt()
    latent_effect = generated.var(dim=1, correction=1).mean().sqrt()

    aggregate = {
        "generated_pairwise_rmse": generated_pairwise.mean().item(),
        "real_pairwise_rmse": real_pairwise.mean().item(),
        "pairwise_rmse_real_ratio": (
            generated_pairwise.mean() / real_pairwise.mean().clamp_min(1e-12)
        ).item(),
        "generated_effective_rank_8x8": generated_ranks.mean().item(),
        "real_effective_rank_8x8": real_ranks.mean().item(),
        "effective_rank_real_ratio": (
            generated_ranks.mean() / real_ranks.mean().clamp_min(1e-12)
        ).item(),
        "generated_class_centroid_within_ratio": (
            generated_centroid_ratio.item()
        ),
        "real_class_centroid_within_ratio": real_centroid_ratio.item(),
        "label_effect_rmse": label_effect.item(),
        "latent_effect_rmse": latent_effect.item(),
        "label_to_latent_effect_ratio": (
            label_effect / latent_effect.clamp_min(1e-12)
        ).item(),
        "pooled_frechet_distance": global_frechet.item(),
        "class_conditional_pooled_frechet_distance": class_frechet.mean().item(),
        "generated_to_real_nearest_rmse": generated_to_real.mean().item(),
        "real_to_generated_nearest_rmse": real_to_generated.mean().item(),
    }
    per_class = [
        {
            "class_index": index,
            "generated_pairwise_rmse": generated_pairwise[index].item(),
            "real_pairwise_rmse": real_pairwise[index].item(),
            "generated_effective_rank_8x8": generated_ranks[index].item(),
            "real_effective_rank_8x8": real_ranks[index].item(),
            "pooled_frechet_distance_8x8": class_frechet[index].item(),
            "generated_to_real_nearest_rmse_8x8": (
                generated_to_real[index].item()
            ),
            "real_to_generated_nearest_rmse_8x8": (
                real_to_generated[index].item()
            ),
        }
        for index in range(num_classes)
    ]
    return aggregate, per_class


@torch.no_grad()
def update_ema(ema_module, module, decay):
    """Move evaluation parameters toward training parameters and copy buffers."""
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1).")
    for ema_parameter, parameter in zip(
        ema_module.parameters(), module.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter, 1 - decay)
    for ema_buffer, buffer in zip(
        ema_module.buffers(), module.buffers(), strict=True
    ):
        ema_buffer.copy_(buffer)


def discriminator_hinge_loss(real_scores, fake_scores):
    """Return the discriminator hinge loss."""
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_hinge_loss(fake_scores):
    """Return the non-saturating hinge objective used by the generator."""
    return -fake_scores.mean()


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
