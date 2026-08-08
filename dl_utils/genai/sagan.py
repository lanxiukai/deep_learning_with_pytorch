"""Self-attention additions for the compact SAGAN lesson."""

import math

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from dl_utils.genai.sn_gan import ProjectionSNDiscriminator, SNGenerator


class SelfAttention(nn.Module):
    """SAGAN non-local attention with a learnable residual scale."""

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


class SAGANGenerator(SNGenerator):
    """SN-GAN generator with self-attention on middle feature maps."""

    def __init__(
        self,
        z_dim=120,
        num_classes=10,
        base_channels=32,
        condition_dim=128,
        image_size=32,
    ):
        super().__init__(
            z_dim=z_dim,
            num_classes=num_classes,
            base_channels=base_channels,
            condition_dim=condition_dim,
            image_size=image_size,
        )
        self.attention = SelfAttention(base_channels * 2)


class SAGANDiscriminator(ProjectionSNDiscriminator):
    """Projection SN-GAN discriminator with attention on middle features."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__(
            num_classes=num_classes,
            base_channels=base_channels,
        )
        self.attention = SelfAttention(base_channels * 2)
