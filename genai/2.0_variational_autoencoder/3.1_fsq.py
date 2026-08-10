"""
Finite Scalar Quantization (FSQ): a codebook-free discrete autoencoder.

VQ-VAE searches a learned table of vectors.  FSQ instead bounds each latent
channel, rounds it to a fixed number of scalar levels, and packs the resulting
digits into one integer token.  There is no nearest-neighbour search, codebook
loss, commitment loss, or dead embedding to maintain.

The encoder still receives a straight-through gradient and the output remains
an integer grid suitable for the same second-stage prior used by VQ-VAE.

Run:
    python genai/2.0_variational_autoencoder/3.1_fsq.py --smoke-test
    python genai/2.0_variational_autoencoder/3.1_fsq.py
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class FiniteScalarQuantizer(nn.Module):
    """Quantize D scalar channels and reversibly pack them into one token."""

    def __init__(self, levels: tuple[int, ...] = (8, 8, 5, 5)) -> None:
        super().__init__()
        if not levels or any(level < 2 for level in levels):
            raise ValueError("every FSQ channel needs at least two levels")
        levels_tensor = torch.tensor(levels, dtype=torch.long)
        basis = torch.ones_like(levels_tensor)
        if len(levels) > 1:
            basis[1:] = torch.cumprod(levels_tensor[:-1], dim=0)
        self.register_buffer("levels", levels_tensor)
        self.register_buffer("basis", basis)
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

    def _digits_to_values(self, digits: Tensor) -> Tensor:
        levels = self.levels.to(dtype=digits.dtype)
        return 2.0 * digits / (levels - 1.0) - 1.0

    def _values_to_digits(self, bounded: Tensor) -> Tensor:
        levels = self.levels.to(dtype=bounded.dtype)
        return torch.round((bounded + 1.0) * (levels - 1.0) / 2.0)

    def pack(self, digits: Tensor) -> Tensor:
        """Pack [..., D] digits using a mixed-radix representation."""
        if digits.shape[-1] != self.dim:
            raise ValueError(f"expected {self.dim} scalar digits")
        return (digits.long() * self.basis).sum(dim=-1)

    def unpack(self, indices: Tensor) -> Tensor:
        """Inverse of pack: [...] integer tokens -> [..., D] digits."""
        return (
            indices[..., None] // self.basis % self.levels
        ).long()

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if z_e.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} latent channels")
        # Channel-last layout makes the per-channel level arithmetic explicit.
        bounded = torch.tanh(z_e).permute(0, 2, 3, 1).contiguous()
        digits = self._values_to_digits(bounded)
        quantized = self._digits_to_values(digits)
        z_st = bounded + (quantized - bounded).detach()
        indices = self.pack(digits)

        counts = torch.bincount(
            indices.reshape(-1), minlength=self.codebook_size
        ).float()
        probabilities = counts / counts.sum()
        perplexity = torch.exp(
            -(probabilities * torch.log(probabilities + 1e-10)).sum()
        )
        diagnostics = {
            "perplexity": perplexity.detach(),
            "active_codes": (counts > 0).sum().detach(),
        }
        return z_st.permute(0, 3, 1, 2).contiguous(), indices, diagnostics

    def indices_to_values(self, indices: Tensor) -> Tensor:
        digits = self.unpack(indices).to(dtype=torch.float32)
        values = self._digits_to_values(digits)
        return values.permute(0, 3, 1, 2).contiguous()


class FSQAutoencoder(nn.Module):
    """32x32 autoencoder with an 8x8 FSQ token grid."""

    def __init__(
        self,
        levels: tuple[int, ...] = (8, 8, 5, 5),
        image_channels: int = 3,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.quantizer = FiniteScalarQuantizer(levels)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.quantizer.dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.quantizer.dim, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, image_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(x))

    def decode_indices(self, indices: Tensor) -> Tensor:
        return self.decoder(self.quantizer.indices_to_values(indices))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, diagnostics = self.encode(x)
        return self.decoder(z_st), indices, diagnostics


def smoke_test() -> None:
    torch.manual_seed(7)
    model = FSQAutoencoder(levels=(4, 4, 4), hidden_channels=32)
    x = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    reconstruction, indices, diagnostics = model(x)
    loss = F.mse_loss(reconstruction, x)
    loss.backward()

    all_indices = torch.arange(model.quantizer.codebook_size)
    digits = model.quantizer.unpack(all_indices)
    restored = model.quantizer.pack(digits)
    assert torch.equal(all_indices, restored), "index packing is not reversible"
    assert reconstruction.shape == x.shape
    assert indices.shape == (4, 8, 8)
    assert model.encoder[0].weight.grad is not None
    print(
        f"smoke test passed: loss={loss.item():.3f}, "
        f"vocabulary={model.quantizer.codebook_size}, "
        f"perplexity={diagnostics['perplexity'].item():.2f}, "
        f"active={diagnostics['active_codes'].item()}"
    )


def parse_levels(text: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from error
    if not levels or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError("every level count must be >= 2")
    return levels


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "fsq"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
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
    model = FSQAutoencoder(args.levels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(3, device=device)
        examples = 0
        model.train()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, _, diagnostics = model(x)
            loss = F.mse_loss(reconstruction, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = x.shape[0]
            sums += torch.stack(
                [
                    loss.detach(),
                    diagnostics["perplexity"],
                    diagnostics["active_codes"].float(),
                ]
            ) * batch_size
            examples += batch_size
        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: recon={means[0]:.4f}, "
            f"perplexity={means[1]:.1f}, active_codes={means[2]:.1f}/"
            f"{model.quantizer.codebook_size}"
        )
        save_image(
            torch.cat([x[:8], reconstruction[:8]]).mul(0.5).add(0.5),
            out_dir / f"epoch_{epoch:03d}.png",
            nrow=8,
        )

    torch.save(
        {
            "model": model.state_dict(),
            "levels": args.levels,
            "codebook_size": model.quantizer.codebook_size,
        },
        out_dir / "fsq_autoencoder.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--levels", type=parse_levels, default=(8, 8, 5, 5))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
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
