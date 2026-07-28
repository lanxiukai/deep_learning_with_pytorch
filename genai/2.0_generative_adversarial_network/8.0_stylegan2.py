"""
Compact StyleGAN2: mapping, mod/demod synthesis, R1 and path regularization.

The implementation is intentionally written with ordinary PyTorch operations
so the important tensor transformations stay visible:

    z -> mapping network -> w
    learned 4x4 constant
    per-layer weight modulation and demodulation
    independent stochastic noise at each convolution
    skip-connected RGB outputs

Training uses StyleGAN2's non-saturating logistic losses, lazy R1
regularization for D, lazy path-length regularization for G, and optional style
mixing.  This 32x32 CIFAR-10 version omits the paper's fused CUDA kernels,
filtered resampling, distributed training and large-resolution configuration.

Run:
    python genai/2.0_generative_adversarial_network/8.0_stylegan2.py --smoke-test
    python genai/2.0_generative_adversarial_network/8.0_stylegan2.py
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PixelNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-8)


class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bias_init: float = 0.0,
        learning_rate_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / learning_rate_multiplier
        )
        self.bias = (
            nn.Parameter(torch.full((out_features,), bias_init))
            if bias
            else None
        )
        self.scale = (
            1.0 / math.sqrt(in_features) * learning_rate_multiplier
        )
        self.learning_rate_multiplier = learning_rate_multiplier

    def forward(self, x: Tensor) -> Tensor:
        bias = (
            self.bias * self.learning_rate_multiplier
            if self.bias is not None
            else None
        )
        return F.linear(x, self.weight * self.scale, bias)


class MappingNetwork(nn.Module):
    def __init__(
        self, z_dim: int, style_dim: int, layers: int = 4
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [PixelNorm()]
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

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class ModulatedConv2d(nn.Module):
    """Per-sample convolution weights with optional StyleGAN2 demodulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.weight_scale = 1.0 / math.sqrt(
            in_channels * kernel_size * kernel_size
        )
        # Bias 1 means the initial style approximately preserves each channel.
        self.modulation = EqualizedLinear(
            style_dim, in_channels, bias_init=1.0
        )

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )
        style_scale = self.modulation(style).view(
            batch, 1, self.in_channels, 1, 1
        )
        weight = self.weight * self.weight_scale * style_scale
        if self.demodulate:
            demodulation = torch.rsqrt(
                weight.square().sum(dim=(2, 3, 4)) + 1e-8
            )
            weight = weight * demodulation.view(
                batch, self.out_channels, 1, 1, 1
            )
        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            height, width = height * 2, width * 2
        x = x.reshape(1, batch * self.in_channels, height, width)
        output = F.conv2d(
            x, weight, padding=self.padding, groups=batch
        )
        return output.view(
            batch, self.out_channels, output.shape[-2], output.shape[-1]
        )


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device
            )
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        upsample: bool = False,
    ) -> None:
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

    def forward(
        self, x: Tensor, style: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        x = self.convolution(x, style)
        x = self.noise(x, noise)
        return F.leaky_relu(x + self.bias, 0.2, inplace=True)


class ToRGB(nn.Module):
    def __init__(
        self, in_channels: int, style_dim: int
    ) -> None:
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels,
            3,
            1,
            style_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        return self.convolution(x, style) + self.bias


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        first: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = StyledConv(
            in_channels, out_channels, style_dim, upsample=not first
        )
        self.conv2 = StyledConv(out_channels, out_channels, style_dim)
        self.to_rgb = ToRGB(out_channels, style_dim)

    def forward(
        self, x: Tensor, styles: Tensor, rgb: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        x = self.conv1(x, styles[:, 0])
        x = self.conv2(x, styles[:, 1])
        new_rgb = self.to_rgb(x, styles[:, 2])
        if rgb is not None:
            rgb = F.interpolate(
                rgb, scale_factor=2, mode="bilinear", align_corners=False
            )
            new_rgb = new_rgb + rgb
        return x, new_rgb


class StyleGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 128,
        style_dim: int = 128,
        base_channels: int = 32,
        mapping_layers: int = 4,
    ) -> None:
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

    def make_ws(
        self,
        z: Tensor,
        mixing_z: Tensor | None = None,
        mixing_cutoff: int | None = None,
    ) -> Tensor:
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

    def synthesize(self, ws: Tensor) -> Tensor:
        if ws.shape[1:] != (self.num_ws, self.style_dim):
            raise ValueError(
                f"expected ws shape [B, {self.num_ws}, {self.style_dim}]"
            )
        x = self.constant.expand(ws.shape[0], -1, -1, -1)
        rgb: Tensor | None = None
        for index, block in enumerate(self.blocks):
            block_styles = ws[:, 3 * index : 3 * index + 3]
            x, rgb = block(x, block_styles, rgb)
        assert rgb is not None
        return rgb

    def forward(
        self,
        z: Tensor,
        mixing_z: Tensor | None = None,
        mixing_cutoff: int | None = None,
        return_ws: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ws = self.make_ws(z, mixing_z, mixing_cutoff)
        image = self.synthesize(ws)
        return (image, ws) if return_ws else image


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        residual = F.avg_pool2d(self.skip(x), 2)
        h = F.leaky_relu(self.conv1(x), 0.2, inplace=False)
        h = F.leaky_relu(self.conv2(h), 0.2, inplace=False)
        h = F.avg_pool2d(h, 2)
        return (h + residual) * self.scale


class MinibatchStandardDeviation(nn.Module):
    def __init__(self, maximum_group: int = 4) -> None:
        super().__init__()
        self.maximum_group = maximum_group

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        group = min(batch, self.maximum_group)
        while batch % group:
            group -= 1
        grouped = x.view(group, -1, channels, height, width)
        deviation = torch.sqrt(grouped.var(dim=0, unbiased=False) + 1e-8)
        feature = deviation.mean(dim=(1, 2, 3), keepdim=True)
        feature = feature.repeat(group, 1, height, width)
        return torch.cat([x, feature], dim=1)


class StyleDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
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
        self.final_conv = nn.Conv2d(channels[4] + 1, channels[4], 3, padding=1)
        self.final = nn.Sequential(
            nn.Flatten(),
            EqualizedLinear(channels[4] * 4 * 4, channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(channels[4], 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = F.leaky_relu(self.from_rgb(x), 0.2, inplace=False)
        h = self.blocks(h)
        h = self.minibatch_std(h)
        h = F.leaky_relu(self.final_conv(h), 0.2, inplace=False)
        return self.final(h).squeeze(1)


def discriminator_logistic_loss(
    real_scores: Tensor, fake_scores: Tensor
) -> Tensor:
    return F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()


def generator_nonsaturating_loss(fake_scores: Tensor) -> Tensor:
    return F.softplus(-fake_scores).mean()


def r1_penalty(real_scores: Tensor, real_images: Tensor) -> Tensor:
    gradients = torch.autograd.grad(
        real_scores.sum(), real_images, create_graph=True
    )[0]
    return gradients.square().flatten(1).sum(dim=1).mean()


def path_length_penalty(
    generated_images: Tensor,
    ws: Tensor,
    running_mean: Tensor,
    decay: float = 0.01,
) -> tuple[Tensor, Tensor]:
    noise = torch.randn_like(generated_images) / math.sqrt(
        generated_images.shape[2] * generated_images.shape[3]
    )
    gradients = torch.autograd.grad(
        (generated_images * noise).sum(), ws, create_graph=True
    )[0]
    path_lengths = torch.sqrt(
        gradients.square().sum(dim=2).mean(dim=1) + 1e-8
    )
    current_mean = path_lengths.mean().detach()
    updated_mean = running_mean.lerp(current_mean, decay)
    penalty = (path_lengths - updated_mean).square().mean()
    return penalty, updated_mean


def smoke_test() -> None:
    torch.manual_seed(7)
    generator = StyleGenerator(
        z_dim=16, style_dim=16, base_channels=4, mapping_layers=2
    )
    discriminator = StyleDiscriminator(base_channels=4)
    real = torch.randn(2, 3, 32, 32, requires_grad=True)
    z1 = torch.randn(2, 16)
    z2 = torch.randn(2, 16)

    fake, ws = generator(z1, mixing_z=z2, mixing_cutoff=5, return_ws=True)
    d_loss = discriminator_logistic_loss(
        discriminator(real), discriminator(fake.detach())
    )
    penalty_r1 = r1_penalty(discriminator(real), real)
    (d_loss + 0.5 * penalty_r1).backward()
    discriminator.zero_grad(set_to_none=True)

    discriminator.requires_grad_(False)
    fake, ws = generator(z1, return_ws=True)
    g_loss = generator_nonsaturating_loss(discriminator(fake))
    penalty_path, path_mean = path_length_penalty(
        fake, ws, torch.zeros(())
    )
    (g_loss + penalty_path).backward()
    discriminator.requires_grad_(True)

    first_modulation = generator.blocks[0].conv1.convolution.modulation
    assert fake.shape == real.shape
    assert ws.shape == (2, generator.num_ws, 16)
    assert first_modulation.weight.grad is not None
    print(
        f"smoke test passed: G={g_loss.item():.3f}, D={d_loss.item():.3f}, "
        f"R1={penalty_r1.item():.3f}, path={penalty_path.item():.3f}, "
        f"path_mean={path_mean.item():.3f}"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "stylegan2"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    generator = StyleGenerator(
        args.z_dim, args.style_dim, args.base_channels
    ).to(device)
    discriminator = StyleDiscriminator(args.base_channels).to(device)
    # Lazy regularization changes how often each optimizer sees a penalty.
    # Compensate the learning rate and beta time constants as StyleGAN2 does.
    g_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr * g_ratio,
        betas=(0.0, 0.99**g_ratio),
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )
    fixed_z = torch.randn(64, args.z_dim, device=device)
    path_mean = torch.zeros((), device=device)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(4, device=device)
        examples = 0
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            batch_size = real.shape[0]

            use_r1 = global_step % args.d_reg_every == 0
            real.requires_grad_(use_r1)
            z = torch.randn(batch_size, args.z_dim, device=device)
            mixing_z = (
                torch.randn_like(z)
                if random.random() < args.style_mixing_probability
                else None
            )
            with torch.no_grad():
                fake = generator(z, mixing_z=mixing_z)
            real_scores = discriminator(real)
            fake_scores = discriminator(fake)
            d_main = discriminator_logistic_loss(real_scores, fake_scores)
            if use_r1:
                d_r1 = r1_penalty(real_scores, real)
                d_loss = (
                    d_main
                    + 0.5
                    * args.r1_gamma
                    * args.d_reg_every
                    * d_r1
                )
            else:
                d_r1 = real.new_zeros(())
                d_loss = d_main
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()
            real.requires_grad_(False)

            discriminator.requires_grad_(False)
            z = torch.randn(batch_size, args.z_dim, device=device)
            mixing_z = (
                torch.randn_like(z)
                if random.random() < args.style_mixing_probability
                else None
            )
            fake = generator(z, mixing_z=mixing_z)
            g_main = generator_nonsaturating_loss(discriminator(fake))
            use_path = global_step % args.g_reg_every == 0
            if use_path:
                path_batch = max(1, batch_size // args.path_batch_shrink)
                path_z = torch.randn(
                    path_batch, args.z_dim, device=device
                )
                path_images, path_ws = generator(path_z, return_ws=True)
                g_path, path_mean = path_length_penalty(
                    path_images, path_ws, path_mean
                )
                g_loss = (
                    g_main
                    + args.path_weight * args.g_reg_every * g_path
                )
            else:
                g_path = real.new_zeros(())
                g_loss = g_main
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
            discriminator.requires_grad_(True)

            sums += torch.stack(
                [
                    g_main.detach(),
                    d_main.detach(),
                    d_r1.detach(),
                    g_path.detach(),
                ]
            ) * batch_size
            examples += batch_size
            global_step += 1

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: G={means[0]:.4f}, D={means[1]:.4f}, "
            f"R1={means[2]:.4f}, path={means[3]:.4f}, "
            f"path_mean={path_mean.item():.3f}"
        )
        generator.eval()
        with torch.inference_mode():
            samples = generator(fixed_z)
        generator.train()
        save_image(
            samples.mul(0.5).add(0.5),
            out_dir / f"epoch_{epoch:03d}.png",
            nrow=8,
        )

    torch.save(
        {
            "model": generator.state_dict(),
            "z_dim": args.z_dim,
            "style_dim": args.style_dim,
            "base_channels": args.base_channels,
            "path_mean": path_mean,
        },
        out_dir / "stylegan2_generator.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    parser.add_argument("--d-reg-every", type=int, default=16)
    parser.add_argument("--path-weight", type=float, default=2.0)
    parser.add_argument("--g-reg-every", type=int, default=4)
    parser.add_argument("--path-batch-shrink", type=int, default=2)
    parser.add_argument("--style-mixing-probability", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
