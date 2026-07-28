"""Shared Variational AutoEncoder with 63,332,411 trainable parameters (~63.3M)."""

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.devices.selection import get_device

device = get_device()

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

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)                         # (B, 100)
        std = torch.exp(self.linear3(x))             # (B, 100)
        z = mu + std * torch.randn_like(mu)          # (B, 100)
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
        x = torch.sigmoid(x)      # pixel-wise Gaussian mean p(x|z), σ²=1
        return x


class VAE(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        self.encoder = VAEEncoder(latent_dims)
        self.decoder = VAEDecoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        mu, std, z = self.encoder(x)
        return mu, std, self.decoder(z)
