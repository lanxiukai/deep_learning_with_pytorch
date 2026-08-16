"""Shared 256x256 VAE and its diagonal-Gaussian teaching primitives."""

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.runtime.devices import try_gpu

device = try_gpu()


def diagonal_gaussian_kl(mu, std):
    """Return KL[N(mu, std^2) || N(0, 1)] per sample and dimension."""
    if mu.shape != std.shape or mu.ndim != 2:
        raise ValueError("mu and std must have matching [batch, latent] shapes.")
    return 0.5 * (
        mu.square()
        + std.square()
        - 1.0
        - 2.0 * torch.log(std.clamp_min(1e-8))
    )


def reparameterize(mu, std):
    """Draw a differentiable sample from a diagonal Gaussian posterior."""
    if mu.shape != std.shape:
        raise ValueError("mu and std must have matching shapes.")
    return mu + std * torch.randn_like(mu)


class VAEEncoder(nn.Module):
    # Manual forward (not Sequential) — the network branches into μ and log σ
    def __init__(self, latent_dims=100):
        super().__init__()
        # input 256 by 256 by 3 channels
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        # 128 by 128 with 8 channels
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        # 64 by 64 with 16 channels
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        # 31 by 31 with 32 channels
        self.linear1 = nn.Linear(32*31*31, 1024)
        self.linear2 = nn.Linear(1024, latent_dims)   # μ
        self.linear3 = nn.Linear(1024, latent_dims)   # log σ

    def statistics(self, x):
        """Encode an image batch into posterior mean and standard deviation."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)                         # (B, latent_dims)
        # Bound only the exponential's numerical range, not the sampled z.
        log_std = self.linear3(x).clamp(-12.0, 12.0)
        std = torch.exp(log_std)                     # (B, latent_dims)
        return mu, std

    def forward(self, x):
        mu, std = self.statistics(x)
        z = reparameterize(mu, std)                  # (B, 100)
        return mu, std, z


class VAEDecoder(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32*31*31),
            nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1,
                  unflattened_size=(32,31,31))
        # padding: symmetric crop from output edges
        # 0 ≤ output_padding < stride:
        # extra rows/cols on right & bottom
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                                output_padding=1),      # 31→63
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                                padding=1, output_padding=1),  # 63→128
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                                padding=1, output_padding=1))  # 128→256

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)  # → (B, 3, 256, 256)
        # Pixel-wise Gaussian mean; the fixed likelihood scale is absorbed by
        # the reconstruction-loss coefficient in the lesson script.
        x = torch.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.encoder = VAEEncoder(latent_dims)
        self.decoder = VAEDecoder(latent_dims)

    def reconstruct(self, x, *, sample=True):
        """Reconstruct from a posterior sample or deterministically from mu."""
        mu, std = self.encoder.statistics(x)
        z = reparameterize(mu, std) if sample else mu
        return mu, std, self.decoder(z)

    def forward(self, x):
        return self.reconstruct(x, sample=True)


__all__ = [
    "VAE",
    "VAEDecoder",
    "VAEEncoder",
    "device",
    "diagonal_gaussian_kl",
    "reparameterize",
]
