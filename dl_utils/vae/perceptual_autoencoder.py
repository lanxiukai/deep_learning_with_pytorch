"""Shared network blocks for VAE-GAN, VQGAN, and KL-AE lessons.

Only architectures and small loss primitives live here.  Each lesson keeps
its optimizer phases and algorithm-specific reduction visible in the script.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl_utils.vae.quantization import VectorQuantizer
from dl_utils.vae.vae_common import reparameterize_logvar


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class PerceptualEncoder32(nn.Module):
    """32x32 RGB image to an 8x8 latent grid."""

    def __init__(self, out_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        groups = min(32, hidden_channels)
        while hidden_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels // 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PerceptualDecoder32(nn.Module):
    """8x8 latent grid to a 32x32 RGB image in [-1, 1]."""

    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        groups = min(32, hidden_channels)
        while hidden_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @property
    def last_layer(self) -> nn.Parameter:
        layer = self.net[-2]
        if not isinstance(layer, nn.ConvTranspose2d):
            raise RuntimeError("decoder final layer changed unexpectedly")
        return layer.weight

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class VQPerceptualAutoencoder32(nn.Module):
    """VQGAN first-stage model without hiding its loss in the module."""

    def __init__(
        self,
        *,
        latent_channels: int = 64,
        codebook_size: int = 512,
        hidden_channels: int = 128,
        commitment: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = PerceptualEncoder32(latent_channels, hidden_channels)
        self.quantizer = VectorQuantizer(
            codebook_size, latent_channels, commitment
        )
        self.decoder = PerceptualDecoder32(latent_channels, hidden_channels)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(x))

    def decode_indices(self, indices: Tensor) -> Tensor:
        return self.decoder(self.quantizer.lookup(indices))

    def reconstruct(self, x: Tensor) -> Tensor:
        z_st, _, _, _ = self.encode(x)
        return self.decoder(z_st)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, quantizer_loss, diagnostics = self.encode(x)
        return self.decoder(z_st), indices, quantizer_loss, diagnostics


class KLPerceptualAutoencoder32(nn.Module):
    """Continuous KL-regularized autoencoder used before latent diffusion."""

    def __init__(
        self, *, latent_channels: int = 4, hidden_channels: int = 128
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = PerceptualEncoder32(2 * latent_channels, hidden_channels)
        self.decoder = PerceptualDecoder32(latent_channels, hidden_channels)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-12.0, 12.0)

    def encode_latent(
        self,
        x: Tensor,
        *,
        sample: bool = True,
        latent_scale: float = 1.0,
    ) -> Tensor:
        """Encode the downstream latent and apply one checkpoint-level scale."""
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        mu, logvar = self.encode(x)
        z = reparameterize_logvar(mu, logvar) if sample else mu
        return z * latent_scale

    def decode_latent(self, z: Tensor, *, latent_scale: float = 1.0) -> Tensor:
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        return self.decoder(z / latent_scale)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.decode_latent(
            self.encode_latent(x, sample=False), latent_scale=1.0
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize_logvar(mu, logvar)
        return self.decoder(z), mu, logvar, z


class PatchDiscriminator32(nn.Module):
    """Small PatchGAN returning a spatial grid of local real/fake logits."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> list[nn.Module]:
            return [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.features = nn.ModuleList(
            [
                nn.Sequential(*block(3, base_channels)),
                nn.Sequential(*block(base_channels, base_channels * 2)),
                nn.Sequential(*block(base_channels * 2, base_channels * 4)),
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 3, 1, 1),
        )

    def extract_features(self, x: Tensor, *, layer: int = 1) -> Tensor:
        if not 0 <= layer < len(self.features):
            raise ValueError("feature layer is outside the discriminator")
        for index, block in enumerate(self.features):
            x = block(x)
            if index == layer:
                return x
        raise RuntimeError("unreachable feature layer")

    def forward(self, x: Tensor) -> Tensor:
        for block in self.features:
            x = block(x)
        return self.head(x)


class RandomFeaturePerceptualLoss(nn.Module):
    """Frozen multiscale conv features for offline smoke tests.

    This is deliberately named *random* and must not be reported as LPIPS.
    It lets the algorithmic gradient path be tested without downloading
    pretrained weights.  Real training should use ``VGGPerceptualLoss``.
    """

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(channels, channels * 2, 3, padding=1),
                    nn.ReLU(),
                ),
            ]
        )
        self.eval().requires_grad_(False)

    def forward(
        self, prediction: Tensor, target: Tensor, *, reduction: str = "mean"
    ) -> Tensor:
        per_sample = prediction.new_zeros(prediction.shape[0])
        for block in self.blocks:
            prediction = block(prediction)
            target = block(target)
            per_sample = per_sample + (prediction - target).abs().flatten(1).mean(1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'none' or 'mean'")


class VGGPerceptualLoss(nn.Module):
    """Frozen ImageNet-VGG16 multi-layer feature L1 distance."""

    def __init__(self) -> None:
        super().__init__()
        # Keep torchvision import lazy: smoke tests need no model download.
        from torchvision import models

        features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList(
            [features[:4], features[4:9], features[9:16]]
        )
        self.eval().requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(
        self, prediction: Tensor, target: Tensor, *, reduction: str = "mean"
    ) -> Tensor:
        prediction = (prediction.add(1.0).mul(0.5) - self.mean) / self.std
        target = (target.add(1.0).mul(0.5) - self.mean) / self.std
        per_sample = prediction.new_zeros(prediction.shape[0])
        for block in self.blocks:
            prediction = block(prediction)
            target = block(target)
            per_sample = per_sample + (prediction - target).abs().flatten(1).mean(1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'none' or 'mean'")


def discriminator_hinge_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return 0.5 * (
        F.relu(1.0 - real_logits).mean()
        + F.relu(1.0 + fake_logits).mean()
    )


def adaptive_adversarial_weight(
    base_loss: Tensor,
    adversarial_loss: Tensor,
    last_layer: nn.Parameter,
    *,
    scale: float = 1.0,
    maximum: float = 1e4,
) -> Tensor:
    """Match last-decoder-layer gradient norms and stop the ratio's gradient."""
    base_gradient = torch.autograd.grad(
        base_loss, last_layer, retain_graph=True
    )[0]
    adversarial_gradient = torch.autograd.grad(
        adversarial_loss, last_layer, retain_graph=True
    )[0]
    ratio = base_gradient.norm() / (adversarial_gradient.norm() + 1e-4)
    return (float(scale) * ratio.clamp(0.0, maximum)).detach()


__all__ = [
    "KLPerceptualAutoencoder32",
    "PatchDiscriminator32",
    "PerceptualDecoder32",
    "PerceptualEncoder32",
    "RandomFeaturePerceptualLoss",
    "ResidualBlock",
    "VGGPerceptualLoss",
    "VQPerceptualAutoencoder32",
    "adaptive_adversarial_weight",
    "discriminator_hinge_loss",
]
