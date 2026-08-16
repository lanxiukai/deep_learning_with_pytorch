"""Compact 128x128 StyleGAN models with AdaIN and per-layer styles."""

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.gan.stylegan_common import (
    RESOLUTIONS,
    EqualizedConv2d,
    EqualizedLinear,
    MinibatchStandardDeviation,
    NoiseInjection,
    PixelNorm,
    denormalize,
    filtered_downsample2d,
    filtered_upsample2d,
    make_channel_map,
    validate_alpha,
    validate_resolution,
)


class MappingNetwork(nn.Module):
    """Map normalized Z vectors into StyleGAN's intermediate W space."""

    def __init__(self, z_dim, style_dim, layers=8):
        super().__init__()
        if layers <= 0:
            raise ValueError("mapping network must contain at least one layer.")
        if z_dim <= 0 or style_dim <= 0:
            raise ValueError("z_dim and style_dim must be positive.")
        modules = [PixelNorm()]
        in_features = z_dim
        for _ in range(layers):
            modules.extend(
                [
                    EqualizedLinear(
                        in_features,
                        style_dim,
                        learning_rate_multiplier=0.01,
                        gain=math.sqrt(2),
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_features = style_dim
        self.net = nn.Sequential(*modules)

    def forward(self, z):
        return self.net(z)


class AdaptiveInstanceNorm(nn.Module):
    """Normalize features, then apply sample-specific scale and bias from W."""

    def __init__(self, channels, style_dim):
        super().__init__()
        self.channels = int(channels)
        self.affine = EqualizedLinear(style_dim, self.channels * 2)

    def forward(self, inputs, style):
        mean = inputs.mean(dim=(2, 3), keepdim=True)
        variance = inputs.var(dim=(2, 3), unbiased=False, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + 1e-8)
        scale, bias = self.affine(style).chunk(2, dim=1)
        scale = scale.view(-1, self.channels, 1, 1) + 1.0
        bias = bias.view(-1, self.channels, 1, 1)
        return normalized * scale + bias


class StyledActivation(nn.Module):
    """Apply StyleGAN noise, activation, and AdaIN in the original order."""

    def __init__(
        self,
        channels,
        style_dim,
        resolution,
        fixed_noise_seed,
    ):
        super().__init__()
        self.noise = NoiseInjection(
            channels,
            resolution,
            fixed_noise_seed,
        )
        self.adain = AdaptiveInstanceNorm(channels, style_dim)

    def forward(self, inputs, style, noise_mode="random"):
        hidden = self.noise(inputs, noise_mode)
        hidden = F.leaky_relu(hidden, 0.2)
        return self.adain(hidden, style)


class SynthesisBlock(nn.Module):
    """Generate one StyleGAN resolution using two independent W inputs."""

    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        resolution,
        fixed_noise_seed,
        *,
        first=False,
    ):
        super().__init__()
        self.first = bool(first)
        self.conv1 = (
            None
            if self.first
            else EqualizedConv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
            )
        )
        self.activation1 = StyledActivation(
            out_channels,
            style_dim,
            resolution,
            fixed_noise_seed,
        )
        self.conv2 = EqualizedConv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
        )
        self.activation2 = StyledActivation(
            out_channels,
            style_dim,
            resolution,
            fixed_noise_seed + 1,
        )

    def forward(self, inputs, styles, noise_mode="random"):
        if self.first:
            hidden = inputs
        else:
            hidden = filtered_upsample2d(inputs)
            assert self.conv1 is not None
            hidden = self.conv1(hidden)
        hidden = self.activation1(hidden, styles[:, 0], noise_mode)
        hidden = self.conv2(hidden)
        return self.activation2(hidden, styles[:, 1], noise_mode)


class StyleGANGenerator(nn.Module):
    """Progressive AdaIN generator with style mixing and truncation."""

    def __init__(
        self,
        z_dim=128,
        style_dim=128,
        base_channels=32,
        mapping_layers=8,
        w_avg_beta=0.995,
    ):
        super().__init__()
        if not 0.0 <= w_avg_beta < 1.0:
            raise ValueError("w_avg_beta must be in [0, 1).")
        self.z_dim = int(z_dim)
        self.style_dim = int(style_dim)
        self.base_channels = int(base_channels)
        if min(self.z_dim, self.style_dim, self.base_channels) <= 0:
            raise ValueError(
                "z_dim, style_dim, and base_channels must be positive."
            )
        self.w_avg_beta = float(w_avg_beta)
        self.resolutions = RESOLUTIONS
        self.styles_per_block = 2
        self.num_ws = len(self.resolutions) * self.styles_per_block
        self.channels = make_channel_map(self.base_channels, self.resolutions)
        self.mapping = MappingNetwork(
            self.z_dim,
            self.style_dim,
            mapping_layers,
        )
        self.register_buffer("w_avg", torch.zeros(self.style_dim))
        self.constant = nn.Parameter(
            torch.randn(1, self.channels[4], 4, 4)
        )
        self.blocks = nn.ModuleList(
            [
                SynthesisBlock(
                    self.channels[resolution // 2]
                    if resolution > 4
                    else self.channels[4],
                    self.channels[resolution],
                    self.style_dim,
                    resolution,
                    fixed_noise_seed=2 * index,
                    first=resolution == 4,
                )
                for index, resolution in enumerate(self.resolutions)
            ]
        )
        self.to_rgbs = nn.ModuleList(
            [
                EqualizedConv2d(
                    self.channels[resolution],
                    3,
                    1,
                    gain=1.0,
                )
                for resolution in self.resolutions
            ]
        )

    def num_ws_for_resolution(self, resolution):
        resolution = validate_resolution(resolution, self.resolutions)
        return (
            self.resolutions.index(resolution) + 1
        ) * self.styles_per_block

    def make_ws(
        self,
        z,
        *,
        resolution=128,
        mixing_z=None,
        mixing_cutoff=None,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
        resolution = validate_resolution(resolution, self.resolutions)
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(
                f"expected z shape [B, {self.z_dim}], got {tuple(z.shape)}"
            )
        first_w = self.mapping(z)
        if update_w_avg:
            with torch.no_grad():
                batch_average = first_w.detach().mean(dim=0)
                self.w_avg.lerp_(batch_average, 1.0 - self.w_avg_beta)
        ws = first_w[:, None, :].repeat(1, self.num_ws, 1)

        active_ws = self.num_ws_for_resolution(resolution)
        if mixing_z is not None:
            if mixing_z.shape != z.shape:
                raise ValueError("mixing_z must have the same shape as z.")
            second_w = self.mapping(mixing_z)
            if mixing_cutoff is None:
                mixing_cutoff = random.randint(1, active_ws - 1)
            if not 0 < mixing_cutoff < active_ws:
                raise ValueError(
                    "mixing_cutoff must be inside the active style stack"
                )
            ws = torch.cat(
                [
                    ws[:, :mixing_cutoff],
                    second_w[:, None, :].repeat(
                        1,
                        self.num_ws - mixing_cutoff,
                        1,
                    ),
                ],
                dim=1,
            )

        truncation_psi = float(truncation_psi)
        if truncation_psi < 0:
            raise ValueError("truncation_psi must be non-negative.")
        if truncation_cutoff is None:
            truncation_cutoff = active_ws
        if not 0 <= truncation_cutoff <= active_ws:
            raise ValueError(
                "truncation_cutoff is outside the active style stack."
            )
        if truncation_psi != 1.0 and truncation_cutoff > 0:
            truncated = torch.lerp(
                self.w_avg.view(1, 1, -1),
                ws[:, :truncation_cutoff],
                truncation_psi,
            )
            ws = torch.cat([truncated, ws[:, truncation_cutoff:]], dim=1)
        return ws

    def synthesize(
        self,
        ws,
        *,
        resolution=128,
        alpha=1.0,
        noise_mode="random",
    ):
        resolution = validate_resolution(resolution, self.resolutions)
        alpha = validate_alpha(alpha)
        if ws.ndim != 3 or ws.shape[1:] != (
            self.num_ws,
            self.style_dim,
        ):
            raise ValueError(
                f"expected ws shape [B, {self.num_ws}, {self.style_dim}]"
            )

        hidden = self.constant.expand(ws.shape[0], -1, -1, -1)
        hidden = self.blocks[0](
            hidden,
            ws[:, : self.styles_per_block],
            noise_mode,
        )
        if resolution == 4:
            return self.to_rgbs[0](hidden)

        target_index = self.resolutions.index(resolution)
        for index in range(1, target_index + 1):
            previous = hidden
            start = index * self.styles_per_block
            hidden = self.blocks[index](
                hidden,
                ws[:, start : start + self.styles_per_block],
                noise_mode,
            )
            if index != target_index:
                continue
            new_rgb = self.to_rgbs[index](hidden)
            if alpha == 1.0:
                return new_rgb
            old_rgb = filtered_upsample2d(self.to_rgbs[index - 1](previous))
            return torch.lerp(old_rgb, new_rgb, alpha)
        raise AssertionError("validated resolution was not reached")

    def forward(
        self,
        z,
        *,
        resolution=128,
        alpha=1.0,
        mixing_z=None,
        mixing_cutoff=None,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
        return_ws=False,
        noise_mode="random",
    ):
        ws = self.make_ws(
            z,
            resolution=resolution,
            mixing_z=mixing_z,
            mixing_cutoff=mixing_cutoff,
            update_w_avg=update_w_avg,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        image = self.synthesize(
            ws,
            resolution=resolution,
            alpha=alpha,
            noise_mode=noise_mode,
        )
        return (image, ws) if return_ws else image


class DiscriminatorBlock(nn.Module):
    """Reduce one StyleGAN discriminator resolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
        )
        self.conv2 = EqualizedConv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
        )

    def forward(self, inputs):
        hidden = F.leaky_relu(self.conv1(inputs), 0.2)
        hidden = F.leaky_relu(self.conv2(hidden), 0.2)
        return filtered_downsample2d(hidden)


class StyleGANDiscriminator(nn.Module):
    """Progressive discriminator paired with the AdaIN generator."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = int(base_channels)
        self.resolutions = RESOLUTIONS
        self.channels = make_channel_map(self.base_channels, self.resolutions)
        self.from_rgbs = nn.ModuleDict(
            {
                str(resolution): EqualizedConv2d(
                    3,
                    self.channels[resolution],
                    1,
                )
                for resolution in self.resolutions
            }
        )
        self.blocks = nn.ModuleDict(
            {
                str(resolution): DiscriminatorBlock(
                    self.channels[resolution],
                    self.channels[resolution // 2],
                )
                for resolution in self.resolutions[1:]
            }
        )
        final_channels = self.channels[4]
        self.minibatch_std = MinibatchStandardDeviation()
        self.final_conv = EqualizedConv2d(
            final_channels + 1,
            final_channels,
            3,
            padding=1,
        )
        self.final_linear = EqualizedLinear(
            final_channels * 4 * 4,
            final_channels,
            gain=math.sqrt(2),
        )
        self.output = EqualizedLinear(final_channels, 1)

    def forward(self, images, resolution=128, alpha=1.0):
        resolution = validate_resolution(resolution, self.resolutions)
        alpha = validate_alpha(alpha)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected RGB images in NCHW format.")
        if images.shape[-2:] != (resolution, resolution):
            raise ValueError(
                f"expected {resolution}x{resolution} images, "
                f"got {images.shape[-2]}x{images.shape[-1]}"
            )

        current_index = self.resolutions.index(resolution)
        if resolution == 4:
            hidden = F.leaky_relu(self.from_rgbs["4"](images), 0.2)
        else:
            new_hidden = F.leaky_relu(
                self.from_rgbs[str(resolution)](images),
                0.2,
            )
            new_hidden = self.blocks[str(resolution)](new_hidden)
            old_images = filtered_downsample2d(images)
            old_hidden = F.leaky_relu(
                self.from_rgbs[str(resolution // 2)](old_images),
                0.2,
            )
            hidden = torch.lerp(old_hidden, new_hidden, alpha)
            for index in range(current_index - 1, 0, -1):
                block_resolution = self.resolutions[index]
                hidden = self.blocks[str(block_resolution)](hidden)

        hidden = self.minibatch_std(hidden)
        hidden = F.leaky_relu(self.final_conv(hidden), 0.2)
        hidden = hidden.flatten(1)
        hidden = F.leaky_relu(self.final_linear(hidden), 0.2)
        return self.output(hidden).squeeze(1)


__all__ = [
    "AdaptiveInstanceNorm",
    "DiscriminatorBlock",
    "MappingNetwork",
    "StyleGANDiscriminator",
    "StyleGANGenerator",
    "StyledActivation",
    "SynthesisBlock",
    "denormalize",
]
