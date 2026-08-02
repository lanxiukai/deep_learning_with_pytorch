"""
pix2pix: paired image-to-image translation with U-Net + PatchGAN.

The conditional discriminator sees (source, target) pairs.  Its patch logits
enforce local realism and source/target compatibility, while the generator's
L1 term preserves the paired global content:

    L_G = BCE(D(source, G(source)), real) + lambda_L1 * |target - G(source)|

Expected data layout (matching filenames in both folders):

    data/pix2pix/train/input/0001.png
    data/pix2pix/train/target/0001.png

Run:
    python genai/2.0_generative_adversarial_network/6.0_pix2pix.py --smoke-test
    python genai/2.0_generative_adversarial_network/6.0_pix2pix.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from dl_utils.data.images import load_rgb_image
from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load filename-aligned pairs and apply the same random flip to both."""

    def __init__(self, root: Path, image_size: int = 256) -> None:
        self.input_dir = root / "input"
        self.target_dir = root / "target"
        if not self.input_dir.is_dir() or not self.target_dir.is_dir():
            raise FileNotFoundError(
                f"expected paired folders {self.input_dir} and {self.target_dir}"
            )
        self.names = sorted(
            path.name
            for path in self.input_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
            and (self.target_dir / path.name).is_file()
        )
        if not self.names:
            raise FileNotFoundError("no matching image filenames were found")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        name = self.names[index]
        source = load_rgb_image(self.input_dir / name)
        target = load_rgb_image(self.target_dir / name)
        size = [self.image_size, self.image_size]
        source = TF.resize(source, size, InterpolationMode.BICUBIC)
        target = TF.resize(target, size, InterpolationMode.BICUBIC)
        if random.random() < 0.5:
            source = TF.hflip(source)
            target = TF.hflip(target)
        source_tensor = TF.normalize(TF.to_tensor(source), (0.5,) * 3, (0.5,) * 3)
        target_tensor = TF.normalize(TF.to_tensor(target), (0.5,) * 3, (0.5,) * 3)
        return source_tensor, target_tensor


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, normalize: bool = True
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([self.net(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """Eight-level U-Net for 256x256 paired translation."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.down1 = DownBlock(3, c, normalize=False)       # 128
        self.down2 = DownBlock(c, c * 2)                    # 64
        self.down3 = DownBlock(c * 2, c * 4)                # 32
        self.down4 = DownBlock(c * 4, c * 8)                # 16
        self.down5 = DownBlock(c * 8, c * 8)                # 8
        self.down6 = DownBlock(c * 8, c * 8)                # 4
        self.down7 = DownBlock(c * 8, c * 8)                # 2
        self.down8 = DownBlock(c * 8, c * 8, normalize=False)  # 1

        self.up1 = UpBlock(c * 8, c * 8, dropout=0.5)
        self.up2 = UpBlock(c * 16, c * 8, dropout=0.5)
        self.up3 = UpBlock(c * 16, c * 8, dropout=0.5)
        self.up4 = UpBlock(c * 16, c * 8)
        self.up5 = UpBlock(c * 16, c * 4)
        self.up6 = UpBlock(c * 8, c * 2)
        self.up7 = UpBlock(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.down8(d7)
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class ConditionalPatchDiscriminator(nn.Module):
    """70x70 PatchGAN; source and candidate target are concatenated."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()

        def block(
            in_channels: int, out_channels: int, stride: int, normalize: bool
        ) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels, out_channels, 4, stride, 1, bias=not normalize
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        c = base_channels
        self.net = nn.Sequential(
            *block(6, c, 2, False),
            *block(c, c * 2, 2, True),
            *block(c * 2, c * 4, 2, True),
            *block(c * 4, c * 8, 1, True),
            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return self.net(torch.cat([source, target], dim=1))


def discriminator_loss(
    discriminator: ConditionalPatchDiscriminator,
    generator: UNetGenerator,
    source: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        fake = generator(source)
    real_logits = discriminator(source, target)
    fake_logits = discriminator(source, fake)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    )
    return loss, fake


def generator_loss(
    discriminator: ConditionalPatchDiscriminator,
    generator: UNetGenerator,
    source: Tensor,
    target: Tensor,
    lambda_l1: float = 100.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    discriminator.requires_grad_(False)
    try:
        fake = generator(source)
        fake_logits = discriminator(source, fake)
        adversarial = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
        reconstruction = F.l1_loss(fake, target)
        loss = adversarial + lambda_l1 * reconstruction
    finally:
        discriminator.requires_grad_(True)
    return loss, adversarial.detach(), reconstruction.detach(), fake


def smoke_test() -> None:
    torch.manual_seed(7)
    generator = UNetGenerator(base_channels=8)
    discriminator = ConditionalPatchDiscriminator(base_channels=8)
    source = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    d_loss, _ = discriminator_loss(discriminator, generator, source, target)
    d_loss.backward()
    discriminator.zero_grad(set_to_none=True)
    g_loss, adversarial, reconstruction, fake = generator_loss(
        discriminator, generator, source, target
    )
    g_loss.backward()
    assert fake.shape == target.shape
    assert discriminator(source, target).shape[-2:] == (30, 30)
    assert generator.down1.net[0].weight.grad is not None
    print(
        f"smoke test passed: G={g_loss.item():.3f}, D={d_loss.item():.3f}, "
        f"adversarial={adversarial.item():.3f}, L1={reconstruction.item():.3f}"
    )


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedImageDataset(args.data_root, image_size=256)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    out_dir = PROJECT_ROOT / "output" / "pix2pix"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = UNetGenerator().to(device)
    discriminator = ConditionalPatchDiscriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )

    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(4, device=device)
        examples = 0
        for source, target in loader:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            d_loss, _ = discriminator_loss(
                discriminator, generator, source, target
            )
            discriminator_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            discriminator_optimizer.step()

            g_loss, adversarial, reconstruction, fake = generator_loss(
                discriminator, generator, source, target, args.lambda_l1
            )
            generator_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            generator_optimizer.step()

            batch_size = source.shape[0]
            sums += torch.stack(
                [g_loss.detach(), d_loss.detach(), adversarial, reconstruction]
            ) * batch_size
            examples += batch_size

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: G={means[0]:.4f}, D={means[1]:.4f}, "
            f"adversarial={means[2]:.4f}, L1={means[3]:.4f}"
        )
        save_image(
            torch.cat([source[:4], target[:4], fake[:4]]).mul(0.5).add(0.5),
            out_dir / f"epoch_{epoch:03d}.png",
            nrow=4,
        )

    torch.save(generator.state_dict(), out_dir / "pix2pix_generator.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "pix2pix" / "train",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=2e-4)
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
