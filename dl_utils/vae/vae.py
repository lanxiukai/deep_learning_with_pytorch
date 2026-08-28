"""Shared 256x256 VAE and its diagonal-Gaussian teaching primitives."""

import torch
import torch.nn.functional as F
from torch import nn


def diagonal_gaussian_kl(mu, std):
    """Return KL[N(mu, std^2) || N(0, 1)] per sample and dimension."""
    if mu.shape != std.shape or mu.ndim != 2:
        raise ValueError("mu and std must have matching [batch, latent] shapes.")
    return 0.5 * (
        mu.square()
        + std.square()
        - 1.0
        - 2.0 * torch.log(std.clamp_min(1e-8))
    )  # (B, z_dim)


def reparameterize(mu, std):
    """Draw a differentiable sample from a diagonal Gaussian posterior."""
    if mu.shape != std.shape:
        raise ValueError("mu and std must have matching shapes.")
    return mu + std * torch.randn_like(mu)


class VAEEncoder(nn.Module):
    """Map 256x256 RGB images to diagonal-Gaussian posterior parameters."""

    # Manual forward (not Sequential) — the network branches into μ and log(σ)
    def __init__(self, z_dim=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)  # (B, 8, 128, 128)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)  # (B, 16, 64, 64)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)  # (B, 32, 31, 31)
        self.linear1 = nn.Linear(32 * 31 * 31, 1024)  # (B, 1024)
        self.linear2 = nn.Linear(1024, z_dim)  # μ
        self.linear3 = nn.Linear(1024, z_dim)  # log(σ)

    def statistics(self, inputs):
        """Encode an image batch into posterior mean and standard deviation."""
        # inputs: (B, 3, 256, 256)
        hidden = F.relu(self.conv1(inputs))  # (B, 8, 128, 128)
        hidden = F.relu(self.batch2(self.conv2(hidden)))  # (B, 16, 64, 64)
        hidden = F.relu(self.conv3(hidden))  # (B, 32, 31, 31)
        hidden = torch.flatten(hidden, start_dim=1)  # (B, 32 * 31 * 31)
        hidden = F.relu(self.linear1(hidden))  # (B, 1024)
        mu = self.linear2(hidden)  # μ: (B, z_dim)
        # Bound only the exponential's numerical range, not the sampled z.
        log_std = self.linear3(hidden).clamp(-12.0, 12.0)  # log(σ): (B, z_dim)
        std = torch.exp(log_std)  # σ: (B, z_dim)
        return mu, std

    def forward(self, inputs):
        mu, std = self.statistics(inputs)
        z = reparameterize(mu, std)  # (B, z_dim)
        return mu, std, z


class VAEDecoder(nn.Module):
    """Decode latent vectors into 256x256 RGB Bernoulli means."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 31 * 31),
            nn.ReLU(inplace=True),
        )
        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(32, 31, 31),
        )  # (B, 32, 31, 31)
        # padding: symmetric crop from output edges
        # 0 ≤ output_padding < stride:
        # extra rows/cols on right & bottom
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                                output_padding=1),             # (B, 16, 63, 63)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                                padding=1, output_padding=1),  # (B, 8, 128, 128)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                                padding=1, output_padding=1))  # (B, 3, 256, 256)

    def forward(self, z):
        # latent vector z: (B, z_dim)
        hidden = self.linear(z)               # (B, 32 * 31 * 31)
        hidden = self.unflatten(hidden)       # (B, 32, 31, 31)
        hidden = self.conv_transpose(hidden)  # (B, 3, 256, 256)
        # Pixel-wise Gaussian mean; the fixed likelihood scale is absorbed by
        # the reconstruction-loss coefficient in the lesson script.
        decoded_images = torch.sigmoid(hidden)
        return decoded_images


class VAE(nn.Module):
    """Standard convolutional VAE for 256x256 RGB images."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.encoder = VAEEncoder(z_dim)
        self.decoder = VAEDecoder(z_dim)

    def reconstruct(self, inputs, *, sample=True):
        """Reconstruct from a posterior sample or deterministically from mu."""
        mu, std = self.encoder.statistics(inputs)
        z = reparameterize(mu, std) if sample else mu
        return mu, std, self.decoder(z)

    def forward(self, inputs):
        return self.reconstruct(inputs, sample=True)


__all__ = [
    "VAE",
    "VAEDecoder",
    "VAEEncoder",
    "diagonal_gaussian_kl",
    "reparameterize",
]
