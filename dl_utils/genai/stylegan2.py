"""Model blocks shared by the compact StyleGAN2 lesson."""

import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class PixelNorm(nn.Module):
    def forward(self, inputs):
        return inputs * torch.rsqrt(
            inputs.square().mean(dim=1, keepdim=True) + 1e-8
        )


class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bias_init=0.0,
        learning_rate_multiplier=1.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / learning_rate_multiplier
        )
        self.bias = (
            nn.Parameter(torch.full((out_features,), bias_init))
            if bias
            else None
        )
        self.scale = learning_rate_multiplier / math.sqrt(in_features)
        self.learning_rate_multiplier = learning_rate_multiplier

    def forward(self, inputs):
        bias = (
            self.bias * self.learning_rate_multiplier
            if self.bias is not None
            else None
        )
        return F.linear(inputs, self.weight * self.scale, bias)


class MappingNetwork(nn.Module):
    """Map normalized z vectors into the intermediate W space."""

    def __init__(self, z_dim, style_dim, layers=4):
        super().__init__()
        modules = [PixelNorm()]
        in_features = z_dim
        for _ in range(layers):
            modules.extend(
                [
                    EqualizedLinear(
                        in_features,
                        style_dim,
                        learning_rate_multiplier=0.01,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_features = style_dim
        self.net = nn.Sequential(*modules)

    def forward(self, noise):
        return self.net(noise)


class ModulatedConv2d(nn.Module):
    """Per-sample convolution with StyleGAN2 weight demodulation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(
                1, out_channels, in_channels, kernel_size, kernel_size
            )
        )
        self.weight_scale = 1 / math.sqrt(
            in_channels * kernel_size * kernel_size
        )
        self.modulation = EqualizedLinear(
            style_dim, in_channels, bias_init=1.0
        )

    def forward(self, inputs, style):
        batch, channels, height, width = inputs.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )

        style = self.modulation(style).view(
            batch, 1, self.in_channels, 1, 1
        )
        weight = self.weight * self.weight_scale * style
        if self.demodulate:
            scale = torch.rsqrt(
                weight.square().sum(dim=(2, 3, 4)) + 1e-8
            )
            weight = weight * scale.view(
                batch, self.out_channels, 1, 1, 1
            )
        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        if self.upsample:
            inputs = F.interpolate(inputs, scale_factor=2, mode="nearest")
            height, width = height * 2, width * 2
        inputs = inputs.reshape(
            1, batch * self.in_channels, height, width
        )
        outputs = F.conv2d(
            inputs, weight, padding=self.padding, groups=batch
        )
        return outputs.view(
            batch, self.out_channels, outputs.shape[-2], outputs.shape[-1]
        )


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs, noise=None, randomize=True):
        if not randomize:
            return inputs
        if noise is None:
            noise = torch.randn(
                inputs.shape[0],
                1,
                inputs.shape[2],
                inputs.shape[3],
                device=inputs.device,
            )
        return inputs + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=False):
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels,
            out_channels,
            3,
            style_dim,
            demodulate=True,
            upsample=upsample,
        )
        self.noise = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, inputs, style, randomize_noise=True):
        hidden = self.convolution(inputs, style)
        hidden = self.noise(hidden, randomize=randomize_noise)
        return F.leaky_relu(hidden + self.bias, 0.2, inplace=True)


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels, 3, 1, style_dim, demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, inputs, style):
        return self.convolution(inputs, style) + self.bias


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, first=False):
        super().__init__()
        self.conv1 = StyledConv(
            in_channels, out_channels, style_dim, upsample=not first
        )
        self.conv2 = StyledConv(out_channels, out_channels, style_dim)
        self.to_rgb = ToRGB(out_channels, style_dim)

    def forward(self, inputs, styles, rgb, randomize_noise=True):
        hidden = self.conv1(inputs, styles[:, 0], randomize_noise)
        hidden = self.conv2(hidden, styles[:, 1], randomize_noise)
        new_rgb = self.to_rgb(hidden, styles[:, 2])
        if rgb is not None:
            rgb = F.interpolate(
                rgb, scale_factor=2, mode="bilinear", align_corners=False
            )
            new_rgb = new_rgb + rgb
        return hidden, new_rgb


class StyleGenerator(nn.Module):
    """32x32 mapping-and-synthesis generator with 12 style inputs."""

    def __init__(
        self,
        z_dim=128,
        style_dim=128,
        base_channels=32,
        mapping_layers=4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.style_dim = style_dim
        self.mapping = MappingNetwork(z_dim, style_dim, mapping_layers)
        channels = {
            4: base_channels * 8,
            8: base_channels * 8,
            16: base_channels * 4,
            32: base_channels * 2,
        }
        self.constant = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.blocks = nn.ModuleList(
            [
                SynthesisBlock(
                    channels[4], channels[4], style_dim, first=True
                ),
                SynthesisBlock(channels[4], channels[8], style_dim),
                SynthesisBlock(channels[8], channels[16], style_dim),
                SynthesisBlock(channels[16], channels[32], style_dim),
            ]
        )
        self.num_ws = len(self.blocks) * 3

    def make_ws(self, z, mixing_z=None, mixing_cutoff=None):
        first_w = self.mapping(z)
        ws = first_w[:, None, :].repeat(1, self.num_ws, 1)
        if mixing_z is not None:
            second_w = self.mapping(mixing_z)
            if mixing_cutoff is None:
                mixing_cutoff = random.randint(1, self.num_ws - 1)
            if not 0 < mixing_cutoff < self.num_ws:
                raise ValueError("mixing_cutoff must be inside the style stack")
            ws = torch.cat(
                [
                    ws[:, :mixing_cutoff],
                    second_w[:, None, :].repeat(
                        1, self.num_ws - mixing_cutoff, 1
                    ),
                ],
                dim=1,
            )
        return ws

    def synthesize(self, ws, randomize_noise=True):
        if ws.shape[1:] != (self.num_ws, self.style_dim):
            raise ValueError(
                f"expected ws shape [B, {self.num_ws}, {self.style_dim}]"
            )
        hidden = self.constant.expand(ws.shape[0], -1, -1, -1)
        rgb = None
        for index, block in enumerate(self.blocks):
            block_styles = ws[:, 3 * index : 3 * index + 3]
            hidden, rgb = block(
                hidden, block_styles, rgb, randomize_noise
            )
        assert rgb is not None
        return rgb

    def forward(
        self,
        z,
        mixing_z=None,
        mixing_cutoff=None,
        return_ws=False,
        randomize_noise=True,
    ):
        ws = self.make_ws(z, mixing_z, mixing_cutoff)
        image = self.synthesize(ws, randomize_noise)
        return (image, ws) if return_ws else image


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.scale = 1 / math.sqrt(2)

    def forward(self, inputs):
        residual = F.avg_pool2d(self.skip(inputs), 2)
        hidden = F.leaky_relu(self.conv1(inputs), 0.2, inplace=False)
        hidden = F.leaky_relu(self.conv2(hidden), 0.2, inplace=False)
        hidden = F.avg_pool2d(hidden, 2)
        return (hidden + residual) * self.scale


class MinibatchStandardDeviation(nn.Module):
    def __init__(self, maximum_group=4):
        super().__init__()
        self.maximum_group = maximum_group

    def forward(self, inputs):
        batch, channels, height, width = inputs.shape
        group = min(batch, self.maximum_group)
        while batch % group:
            group -= 1
        grouped = inputs.view(group, -1, channels, height, width)
        deviation = torch.sqrt(grouped.var(dim=0, unbiased=False) + 1e-8)
        feature = deviation.mean(dim=(1, 2, 3), keepdim=True)
        feature = feature.repeat(group, 1, height, width)
        return torch.cat([inputs, feature], dim=1)


class StyleDiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        channels = {
            32: base_channels * 2,
            16: base_channels * 4,
            8: base_channels * 8,
            4: base_channels * 8,
        }
        self.from_rgb = nn.Conv2d(3, channels[32], 1)
        self.blocks = nn.Sequential(
            DiscriminatorResidualBlock(channels[32], channels[16]),
            DiscriminatorResidualBlock(channels[16], channels[8]),
            DiscriminatorResidualBlock(channels[8], channels[4]),
        )
        self.minibatch_std = MinibatchStandardDeviation()
        self.final_conv = nn.Conv2d(
            channels[4] + 1, channels[4], 3, padding=1
        )
        self.final = nn.Sequential(
            nn.Flatten(),
            EqualizedLinear(channels[4] * 4 * 4, channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(channels[4], 1),
        )

    def forward(self, images):
        hidden = F.leaky_relu(self.from_rgb(images), 0.2, inplace=False)
        hidden = self.blocks(hidden)
        hidden = self.minibatch_std(hidden)
        hidden = F.leaky_relu(self.final_conv(hidden), 0.2, inplace=False)
        return self.final(hidden).squeeze(1)


def denormalize(images):
    return images.mul(0.5).add(0.5).clamp(0, 1)
