"""Class-conditional SAGAN models and projection-discriminator utilities.

SAGAN is the first lesson in this sequence whose original paper explicitly
uses conditional BatchNorm in the generator and a projection discriminator.
Keeping those mechanisms here makes the preceding SN-GAN lesson genuinely
unconditional while allowing BigGAN to inherit the same conditional baseline.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from dl_utils.genai.sn_gan import SNDiscriminatorResidualBlock


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
    """Return labels and latent columns repeated across every class row."""
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
    """Map every class to a guaranteed-different diagnostic class."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 for mismatched labels.")
    return (labels + 1) % num_classes


def projection_diagnostic_sums(
    real_scores,
    fake_scores,
    correct_projection,
    mismatched_projection,
):
    """Return hinge activity and correct-minus-wrong projection batch sums."""
    return torch.stack(
        [
            (real_scores.detach().float() < 1).sum(dtype=torch.float32),
            (fake_scores.detach().float() > -1).sum(dtype=torch.float32),
            (
                correct_projection.detach().float()
                - mismatched_projection.detach().float()
            ).sum(),
        ]
    )


class SelfAttention(nn.Module):
    """SAGAN non-local attention with a zero-initialized residual gate."""

    def __init__(self, channels):
        super().__init__()
        attention_channels = max(1, channels // 8)
        self.query = spectral_norm(
            nn.Conv2d(channels, attention_channels, 1)
        )
        self.key = spectral_norm(
            nn.Conv2d(channels, attention_channels, 1)
        )
        self.value = spectral_norm(
            nn.Conv2d(channels, attention_channels, 1)
        )
        self.output = spectral_norm(
            nn.Conv2d(attention_channels, channels, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(()))
        self._capture_attention_entropy = False
        self.last_attention_entropy = None

    def capture_next_attention_entropy(self):
        """Capture normalized entropy from the next attention forward pass."""
        self.last_attention_entropy = None
        self._capture_attention_entropy = True

    def forward(self, inputs):
        batch, _, height, width = inputs.shape
        query = self.query(inputs).flatten(2).transpose(1, 2)
        key = self.key(inputs).flatten(2)
        attention = torch.softmax(query @ key, dim=-1)

        if self._capture_attention_entropy:
            self._capture_attention_entropy = False
            probabilities = attention.detach().to(
                device="cpu", dtype=torch.float32
            )
            probabilities = probabilities.clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            maximum_entropy = math.log(probabilities.shape[-1])
            normalized_entropy = (
                entropy.mean() / maximum_entropy
                if maximum_entropy > 0
                else entropy.new_zeros(())
            )
            self.last_attention_entropy = normalized_entropy.clamp(
                0, 1
            ).item()

        value = self.value(inputs).flatten(2)
        attended = value @ attention.transpose(1, 2)
        attended = attended.view(batch, -1, height, width)
        return inputs + self.gamma * self.output(attended)


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm with class-indexed gain and bias lookup tables."""

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


class SAGANGeneratorResidualBlock(nn.Module):
    """Spectral-normalized conditional residual block with 2x upsampling."""

    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.norm1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.norm2 = ConditionalBatchNorm2d(out_channels, num_classes)
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

    def forward(self, inputs, labels):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs, labels), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(
            F.relu(self.norm2(hidden, labels), inplace=True)
        )
        return hidden + residual


class SAGANGenerator(nn.Module):
    """Compact class-conditional 32x32 SAGAN generator."""

    def __init__(
        self,
        z_dim=120,
        num_classes=10,
        base_channels=32,
        image_size=32,
    ):
        super().__init__()
        if image_size != 32:
            raise ValueError("image_size must be 32.")
        self.z_dim = z_dim
        self.input = spectral_norm(
            nn.Linear(z_dim, base_channels * 8 * 4 * 4)
        )
        self.block1 = SAGANGeneratorResidualBlock(
            base_channels * 8,
            base_channels * 4,
            num_classes,
        )
        self.block2 = SAGANGeneratorResidualBlock(
            base_channels * 4,
            base_channels * 2,
            num_classes,
        )
        self.attention = SelfAttention(base_channels * 2)
        self.block3 = SAGANGeneratorResidualBlock(
            base_channels * 2,
            base_channels,
            num_classes,
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        if labels.ndim != 1 or labels.shape[0] != noise.shape[0]:
            raise ValueError("labels must have shape (batch,).")

        hidden = self.input(noise)
        hidden = hidden.view(noise.shape[0], -1, 4, 4)
        hidden = self.block1(hidden, labels)
        hidden = self.block2(hidden, labels)
        hidden = self.attention(hidden)
        hidden = self.block3(hidden, labels)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class SAGANDiscriminator(nn.Module):
    """Projection discriminator with self-attention on 8x8 features.

    The score is ``u(h(x)) + <e(y), h(x)>``.  This is the first model in the
    lesson sequence that owns label conditioning in the discriminator.
    """

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        channels = base_channels * 4
        self.feature_channels = channels
        self.block1 = SNDiscriminatorResidualBlock(
            3, channels, first=True
        )
        self.block2 = SNDiscriminatorResidualBlock(channels, channels)
        self.attention = SelfAttention(channels)
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

    def extract_features(self, images):
        hidden = self.block1(images)
        hidden = self.block2(hidden)
        hidden = self.attention(hidden)
        hidden = self.block3(hidden)
        hidden = self.block4(hidden)
        hidden = F.relu(hidden, inplace=True)
        return hidden.sum(dim=(2, 3))

    def forward(self, images, labels):
        hidden = self.extract_features(images)
        unconditional = self.output(hidden).squeeze(1)
        projection = (self.class_embedding(labels) * hidden).sum(dim=1)
        return unconditional + projection

    def forward_with_projection_diagnostics(
        self,
        images,
        labels,
        mismatched_labels,
    ):
        """Score images and compare correct with deliberately wrong labels."""
        if labels.shape != mismatched_labels.shape:
            raise ValueError(
                "labels and mismatched_labels must have the same shape."
            )
        hidden = self.extract_features(images)
        unconditional = self.output(hidden).squeeze(1)
        paired_labels = torch.cat([labels, mismatched_labels])
        correct_features, mismatched_features = self.class_embedding(
            paired_labels
        ).chunk(2, dim=0)
        correct_projection = (correct_features * hidden).sum(dim=1)
        mismatched_projection = (mismatched_features * hidden).sum(dim=1)
        scores = unconditional + correct_projection
        return scores, correct_projection, mismatched_projection
