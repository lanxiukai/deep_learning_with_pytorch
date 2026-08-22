"""Compact 128x128 StyleGAN2 without progressive growing or AdaIN.

The generator keeps weight modulation/demodulation, per-layer stochastic
noise, skip-connected RGB outputs, style mixing, truncation, and the original
overlapping W-slot indexing. The discriminator keeps residual filtered
downsampling and a minibatch-standard-deviation feature.
"""

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.gan.stylegan_common import (
    RESOLUTIONS,
    EqualizedConv2d,
    EqualizedLinear,
    MappingNetwork,
    MinibatchStandardDeviation,
    NoiseInjection,
    denormalize,
    filtered_downsample2d,
    filtered_upsample2d,
    make_channel_map,
    validate_resolution,
)


class ModulatedConv2d(nn.Module):
    """Apply per-sample weight modulation and optional demodulation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        *,
        demodulate=True,
        upsample=False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.style_dim = int(style_dim)
        if min(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.style_dim,
        ) <= 0:
            raise ValueError("convolution dimensions must be positive.")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.demodulate = bool(demodulate)
        self.upsample = bool(upsample)
        self.padding = self.kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(
                1,
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.weight_scale = 1 / math.sqrt(
            self.in_channels * self.kernel_size * self.kernel_size
        )
        self.modulation = EqualizedLinear(
            style_dim,
            self.in_channels,
            bias_init=1.0,
        )

    def forward(self, inputs, style):
        batch, channels, height, width = inputs.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )
        if style.ndim != 2 or style.shape != (batch, self.style_dim):
            raise ValueError("style must be a [B, style_dim] tensor.")

        style_scale = self.modulation(style).view(
            batch,
            1,
            self.in_channels,
            1,
            1,
        )
        weight = self.weight * self.weight_scale * style_scale
        if self.demodulate:
            demodulation = torch.rsqrt(
                weight.square().sum(dim=(2, 3, 4)) + 1e-8
            )
            weight = weight * demodulation.view(
                batch,
                self.out_channels,
                1,
                1,
                1,
            )
        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        if self.upsample:
            inputs = filtered_upsample2d(inputs)
            height, width = height * 2, width * 2
        grouped_inputs = inputs.reshape(
            1,
            batch * self.in_channels,
            height,
            width,
        )
        outputs = F.conv2d(
            grouped_inputs,
            weight,
            padding=self.padding,
            groups=batch,
        )
        return outputs.view(
            batch,
            self.out_channels,
            outputs.shape[-2],
            outputs.shape[-1],
        )


class StyledConv(nn.Module):
    """StyleGAN2 modulated convolution followed by noise and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        resolution,
        fixed_noise_seed,
        *,
        upsample=False,
    ):
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels,
            out_channels,
            3,
            style_dim,
            demodulate=True,
            upsample=upsample,
        )
        self.noise = NoiseInjection(
            out_channels,
            resolution,
            fixed_noise_seed,
            per_channel=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, inputs, style, noise_mode="random"):
        hidden = self.convolution(inputs, style)
        hidden = self.noise(hidden, noise_mode)
        return F.leaky_relu(hidden + self.bias, 0.2) * math.sqrt(2)


class ToRGB(nn.Module):
    """Convert features to RGB with modulation but no demodulation."""

    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels,
            3,
            1,
            style_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, inputs, style):
        return self.convolution(inputs, style) + self.bias


class SynthesisBlock(nn.Module):
    """One StyleGAN2 resolution block plus a skip-connected RGB output."""

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
        self.num_ws = 2 if self.first else 3
        self.conv1 = StyledConv(
            in_channels,
            out_channels,
            style_dim,
            resolution,
            fixed_noise_seed,
            upsample=not self.first,
        )
        self.conv2 = (
            None
            if self.first
            else StyledConv(
                out_channels,
                out_channels,
                style_dim,
                resolution,
                fixed_noise_seed + 1,
            )
        )
        self.to_rgb = ToRGB(out_channels, style_dim)

    def forward(self, inputs, styles, rgb, noise_mode="random"):
        if styles.ndim != 3 or styles.shape[1] != self.num_ws:
            raise ValueError(
                f"expected {self.num_ws} W inputs for this synthesis block."
            )
        hidden = self.conv1(inputs, styles[:, 0], noise_mode)
        if self.first:
            rgb_style = styles[:, 1]
        else:
            assert self.conv2 is not None
            hidden = self.conv2(hidden, styles[:, 1], noise_mode)
            rgb_style = styles[:, 2]
        new_rgb = self.to_rgb(hidden, rgb_style)
        if rgb is not None:
            new_rgb = new_rgb + filtered_upsample2d(rgb)
        return hidden, new_rgb


class StyleGenerator(nn.Module):
    """Full-resolution StyleGAN2 generator with skip RGB connections."""

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
        self.mapping = MappingNetwork(
            self.z_dim,
            self.style_dim,
            mapping_layers,
        )
        self.register_buffer("w_avg", torch.zeros(self.style_dim))
        self.resolutions = RESOLUTIONS
        # Every new resolution contributes two W slots. ToRGB consumes the
        # next slot, which is also reused by the following block's first conv.
        self.styles_per_block = 2
        self.num_ws = len(self.resolutions) * self.styles_per_block
        channels = make_channel_map(self.base_channels, self.resolutions)
        self.constant = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.blocks = nn.ModuleList(
            [
                SynthesisBlock(
                    channels[resolution // 2]
                    if resolution > 4
                    else channels[4],
                    channels[resolution],
                    self.style_dim,
                    resolution,
                    fixed_noise_seed=max(0, 2 * index - 1),
                    first=resolution == 4,
                )
                for index, resolution in enumerate(self.resolutions)
            ]
        )

    def num_ws_for_resolution(self, resolution):
        """Return the cumulative W slots through one synthesis resolution."""
        resolution = validate_resolution(resolution, self.resolutions)
        return (
            self.resolutions.index(resolution) + 1
        ) * self.styles_per_block

    def make_ws(
        self,
        z,
        mixing_z=None,
        mixing_cutoff=None,
        *,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
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

        if mixing_z is not None:
            if mixing_z.shape != z.shape:
                raise ValueError("mixing_z must have the same shape as z.")
            second_w = self.mapping(mixing_z)
            if mixing_cutoff is None:
                mixing_cutoff = random.randint(1, self.num_ws - 1)
            if not 0 < mixing_cutoff < self.num_ws:
                raise ValueError(
                    "mixing_cutoff must be inside the style stack"
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
            truncation_cutoff = self.num_ws
        if not 0 <= truncation_cutoff <= self.num_ws:
            raise ValueError("truncation_cutoff is outside the style stack.")
        if truncation_psi != 1.0 and truncation_cutoff > 0:
            truncated = torch.lerp(
                self.w_avg.view(1, 1, -1),
                ws[:, :truncation_cutoff],
                truncation_psi,
            )
            ws = torch.cat([truncated, ws[:, truncation_cutoff:]], dim=1)
        return ws

    def synthesize(self, ws, noise_mode="random"):
        if ws.ndim != 3 or ws.shape[1:] != (
            self.num_ws,
            self.style_dim,
        ):
            raise ValueError(
                f"expected ws shape [B, {self.num_ws}, {self.style_dim}]"
            )
        hidden = self.constant.expand(ws.shape[0], -1, -1, -1)
        rgb = None
        for index, block in enumerate(self.blocks):
            start = 0 if index == 0 else 2 * index - 1
            block_styles = ws[:, start : start + block.num_ws]
            hidden, rgb = block(
                hidden,
                block_styles,
                rgb,
                noise_mode,
            )
        assert rgb is not None
        return rgb

    def forward(
        self,
        z,
        mixing_z=None,
        mixing_cutoff=None,
        return_ws=False,
        noise_mode="random",
        *,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
        ws = self.make_ws(
            z,
            mixing_z,
            mixing_cutoff,
            update_w_avg=update_w_avg,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        image = self.synthesize(ws, noise_mode)
        return (image, ws) if return_ws else image


class DiscriminatorResidualBlock(nn.Module):
    """StyleGAN2 residual downsampling block with a filtered skip path."""

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
        self.skip = EqualizedConv2d(
            in_channels,
            out_channels,
            1,
            bias=False,
            gain=1.0,
        )
        self.scale = 1 / math.sqrt(2)

    def forward(self, inputs):
        residual = filtered_downsample2d(self.skip(inputs))
        hidden = F.leaky_relu(self.conv1(inputs), 0.2)
        hidden = F.leaky_relu(self.conv2(hidden), 0.2)
        hidden = filtered_downsample2d(hidden)
        return (hidden + residual) * self.scale


class StyleDiscriminator(nn.Module):
    """Full 128x128 residual discriminator used by compact StyleGAN2."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = int(base_channels)
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        self.resolutions = RESOLUTIONS
        self.channels = make_channel_map(
            self.base_channels,
            self.resolutions,
        )
        final_resolution = self.resolutions[-1]
        self.from_rgb = EqualizedConv2d(
            3,
            self.channels[final_resolution],
            1,
        )
        self.blocks = nn.Sequential(
            *[
                DiscriminatorResidualBlock(
                    self.channels[resolution],
                    self.channels[resolution // 2],
                )
                for resolution in reversed(self.resolutions[1:])
            ]
        )
        self.minibatch_std = MinibatchStandardDeviation()
        self.final_conv = EqualizedConv2d(
            self.channels[4] + 1,
            self.channels[4],
            3,
            padding=1,
        )
        self.final = nn.Sequential(
            nn.Flatten(),
            EqualizedLinear(
                self.channels[4] * 4 * 4,
                self.channels[4],
                gain=math.sqrt(2),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(self.channels[4], 1),
        )

    def forward(self, images):
        final_resolution = self.resolutions[-1]
        expected = (3, final_resolution, final_resolution)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(
                "StyleDiscriminator expects "
                f"[B, 3, {final_resolution}, {final_resolution}]."
            )
        hidden = F.leaky_relu(self.from_rgb(images), 0.2)
        hidden = self.blocks(hidden)
        hidden = self.minibatch_std(hidden)
        hidden = F.leaky_relu(self.final_conv(hidden), 0.2)
        return self.final(hidden).squeeze(1)


__all__ = [
    "DiscriminatorResidualBlock",
    "MappingNetwork",
    "ModulatedConv2d",
    "StyleDiscriminator",
    "StyleGenerator",
    "StyledConv",
    "SynthesisBlock",
    "ToRGB",
    "denormalize",
]
