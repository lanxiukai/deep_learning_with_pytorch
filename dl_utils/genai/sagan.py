"""Self-attention additions for the compact SAGAN lesson."""

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


class SAGANGenerator(SNGenerator):
    """SN-GAN generator with self-attention on 16x16 features."""

    def __init__(
        self,
        z_dim=120,
        num_classes=10,
        base_channels=32,
        condition_dim=128,
    ):
        super().__init__(
            z_dim=z_dim,
            num_classes=num_classes,
            base_channels=base_channels,
            condition_dim=condition_dim,
        )
        self.attention = SelfAttention(base_channels * 2)


class SAGANDiscriminator(ProjectionSNDiscriminator):
    """Projection SN-GAN discriminator with attention on 16x16 features."""

    def __init__(self, num_classes=10, base_channels=32):
        super().__init__(
            num_classes=num_classes,
            base_channels=base_channels,
        )
        self.attention = SelfAttention(base_channels * 2)
