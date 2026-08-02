"""
SN-GAN -> SAGAN -> BigGAN, distilled into one small conditional model.

The script keeps the mechanisms that transfer beyond the original scale:

* spectral normalization on discriminator and generator convolutions;
* self-attention for non-local interactions;
* shared class embeddings + conditional batch normalization in G;
* a projection discriminator that scores image/class compatibility;
* hinge adversarial losses;
* coordinate-wise truncated-normal sampling (not merely scaling z).

This is a 64x64 CIFAR-10 learning implementation, not a BigGAN reproduction:
the original result also depends on far larger models, batches and compute.

Run:
    python genai/2.0_generative_adversarial_network/7.0_sn_sagan_biggan.py \
        --smoke-test
    python genai/2.0_generative_adversarial_network/7.0_sn_sagan_biggan.py
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()


class SelfAttention(nn.Module):
    """SAGAN non-local block with a learnable residual scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        query_channels = max(1, channels // 8)
        self.query = spectral_norm(nn.Conv2d(channels, query_channels, 1))
        self.key = spectral_norm(nn.Conv2d(channels, query_channels, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.output = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2)
        attention = torch.softmax(query @ key, dim=-1)
        value = self.value(x).flatten(2)
        attended = value @ attention.transpose(1, 2)
        attended = attended.view(batch, channels, height, width)
        return x + self.gamma * self.output(attended)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm2d(channels, affine=False)
        self.affine = spectral_norm(nn.Linear(condition_dim, 2 * channels))
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return self.normalization(x) * (1.0 + gamma) + beta


class GeneratorResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, condition_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = ConditionalBatchNorm2d(in_channels, condition_dim)
        self.norm2 = ConditionalBatchNorm2d(out_channels, condition_dim)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        residual = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(residual)
        h = F.relu(self.norm1(x, condition), inplace=True)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.relu(self.norm2(h, condition), inplace=True))
        return h + residual


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 10,
        base_channels: int = 32,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.class_embedding = nn.Embedding(num_classes, condition_dim)
        self.input = spectral_norm(
            nn.Linear(z_dim + condition_dim, base_channels * 8 * 4 * 4)
        )
        self.block1 = GeneratorResidualBlock(
            base_channels * 8, base_channels * 4, condition_dim
        )
        self.block2 = GeneratorResidualBlock(
            base_channels * 4, base_channels * 2, condition_dim
        )
        self.attention = SelfAttention(base_channels * 2)
        self.block3 = GeneratorResidualBlock(
            base_channels * 2, base_channels, condition_dim
        )
        self.block4 = GeneratorResidualBlock(
            base_channels, base_channels, condition_dim
        )
        self.output_norm = nn.BatchNorm2d(base_channels)
        self.output = spectral_norm(nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        condition = self.class_embedding(labels)
        h = self.input(torch.cat([z, condition], dim=1))
        h = h.view(z.shape[0], -1, 4, 4)
        h = self.block1(h, condition)
        h = self.block2(h, condition)
        h = self.attention(h)
        h = self.block3(h, condition)
        h = self.block4(h, condition)
        return torch.tanh(self.output(F.relu(self.output_norm(h), inplace=True)))


class DiscriminatorResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, first: bool = False
    ) -> None:
        super().__init__()
        self.first = first
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        h = x if self.first else F.relu(x, inplace=False)
        h = self.conv1(h)
        h = self.conv2(F.relu(h, inplace=True))
        h = F.avg_pool2d(h, 2)
        residual = F.avg_pool2d(self.skip(x), 2)
        return h + residual


class ProjectionDiscriminator(nn.Module):
    def __init__(
        self, num_classes: int = 10, base_channels: int = 32
    ) -> None:
        super().__init__()
        c = base_channels
        self.block1 = DiscriminatorResidualBlock(3, c, first=True)
        self.block2 = DiscriminatorResidualBlock(c, c * 2)
        self.attention = SelfAttention(c * 2)
        self.block3 = DiscriminatorResidualBlock(c * 2, c * 4)
        self.block4 = DiscriminatorResidualBlock(c * 4, c * 8)
        self.output = spectral_norm(nn.Linear(c * 8, 1))
        self.class_embedding = spectral_norm(nn.Embedding(num_classes, c * 8))

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.attention(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(h, inplace=True).sum(dim=(2, 3))
        unconditional = self.output(h).squeeze(1)
        projection = (self.class_embedding(labels) * h).sum(dim=1)
        return unconditional + projection


def discriminator_hinge_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def truncated_normal(
    shape: tuple[int, ...],
    truncation: float,
    device: torch.device,
) -> Tensor:
    """Sample N(0,1) and rejection-resample each out-of-range coordinate."""
    if truncation <= 0:
        raise ValueError("truncation must be positive")
    samples = torch.randn(shape, device=device)
    invalid = samples.abs() > truncation
    while invalid.any():
        samples[invalid] = torch.randn(
            int(invalid.sum().item()), device=device
        )
        invalid = samples.abs() > truncation
    return samples


def discriminator_step(
    generator: ConditionalGenerator,
    discriminator: ProjectionDiscriminator,
    real: Tensor,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    z = torch.randn(real.shape[0], generator.z_dim, device=real.device)
    with torch.no_grad():
        fake = generator(z, labels)
    loss = discriminator_hinge_loss(
        discriminator(real, labels), discriminator(fake, labels)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def generator_step(
    generator: ConditionalGenerator,
    discriminator: ProjectionDiscriminator,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    z = torch.randn(labels.shape[0], generator.z_dim, device=labels.device)
    discriminator.requires_grad_(False)
    try:
        fake = generator(z, labels)
        loss = -discriminator(fake, labels).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    finally:
        discriminator.requires_grad_(True)
    return loss.detach()


def smoke_test() -> None:
    torch.manual_seed(7)
    generator = ConditionalGenerator(base_channels=8, condition_dim=32, z_dim=32)
    discriminator = ProjectionDiscriminator(base_channels=8)
    real = torch.randn(4, 3, 64, 64)
    labels = torch.arange(4) % 10
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    d_loss = discriminator_step(
        generator, discriminator, real, labels, d_optimizer
    )
    g_loss = generator_step(generator, discriminator, labels, g_optimizer)
    z = truncated_normal((4, 32), 1.0, torch.device("cpu"))
    fake = generator(z, labels)
    assert fake.shape == real.shape
    assert z.abs().max() <= 1.0
    assert generator.attention.gamma.grad is not None
    print(
        f"smoke test passed: G={g_loss.item():.3f}, D={d_loss.item():.3f}, "
        f"attention_gamma={generator.attention.gamma.item():.4f}"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "sagan"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
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
    generator = ConditionalGenerator(
        z_dim=args.z_dim, base_channels=args.base_channels
    ).to(device)
    discriminator = ProjectionDiscriminator(
        base_channels=args.base_channels
    ).to(device)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.generator_lr, betas=(0.0, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.0, 0.999)
    )
    fixed_labels = torch.arange(64, device=device) % 10

    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(2, device=device)
        examples = 0
        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            d_loss = discriminator_step(
                generator, discriminator, real, labels, d_optimizer
            )
            sampled_labels = torch.randint(
                0, 10, (real.shape[0],), device=device
            )
            g_loss = generator_step(
                generator, discriminator, sampled_labels, g_optimizer
            )
            sums += torch.stack([g_loss, d_loss]) * real.shape[0]
            examples += real.shape[0]

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: G={means[0]:.4f}, D={means[1]:.4f}, "
            f"gamma_G={generator.attention.gamma.item():.3f}, "
            f"gamma_D={discriminator.attention.gamma.item():.3f}"
        )
        generator.eval()
        with torch.inference_mode():
            fixed_z = truncated_normal(
                (64, args.z_dim), args.truncation, device
            )
            samples = generator(fixed_z, fixed_labels)
        generator.train()
        save_image(
            samples.mul(0.5).add(0.5),
            out_dir / f"epoch_{epoch:03d}.png",
            nrow=8,
        )

    torch.save(generator.state_dict(), out_dir / "conditional_generator.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=4e-4)
    parser.add_argument("--truncation", type=float, default=1.0)
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
