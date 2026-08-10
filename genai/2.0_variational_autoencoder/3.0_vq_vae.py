"""
VQ-VAE with a learnable codebook and a small autoregressive token prior.

Stage 1 trains the tokenizer:

    image -> encoder -> nearest codebook vector -> decoder -> reconstruction

The straight-through estimator copies decoder gradients to the encoder, while
the codebook and commitment losses keep encoder outputs and selected vectors
together.  Stage 2 freezes the tokenizer and trains a masked-convolution prior
on the integer index grid.  Reconstruction and unconditional generation are
therefore two different paths, as they should be.

Quick checks and training:
    python genai/2.0_variational_autoencoder/3.0_vq_vae.py --smoke-test
    python genai/2.0_variational_autoencoder/3.0_vq_vae.py --stage all
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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


class VectorQuantizer(nn.Module):
    """Nearest-neighbour VQ with gradient-updated embeddings."""

    def __init__(
        self, codebook_size: int = 512, embedding_dim: int = 64, commitment: float = 0.25
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment = commitment
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(
        self, z_e: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if z_e.shape[1] != self.embedding_dim:
            raise ValueError("encoder channel count must equal embedding_dim")
        flat = z_e.permute(0, 2, 3, 1).contiguous().reshape(-1, self.embedding_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(
            z_e.shape[0], z_e.shape[2], z_e.shape[3], self.embedding_dim
        )
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment * F.mse_loss(z_e, z_q.detach())
        quantizer_loss = codebook_loss + commitment_loss
        # Forward value is exactly z_q; local gradient with respect to z_e is 1.
        z_st = z_e + (z_q - z_e).detach()

        one_hot = F.one_hot(indices, self.codebook_size).float()
        probabilities = one_hot.mean(dim=0)
        perplexity = torch.exp(
            -(probabilities * torch.log(probabilities + 1e-10)).sum()
        )
        active_codes = (probabilities > 0).sum()
        index_grid = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        diagnostics = {
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "perplexity": perplexity.detach(),
            "active_codes": active_codes.detach(),
        }
        return z_st, index_grid, quantizer_loss, diagnostics

    def lookup(self, indices: Tensor) -> Tensor:
        z_q = self.embedding(indices)
        return z_q.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    """32x32 image tokenizer with an 8x8 token grid."""

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        codebook_size: int = 512,
        commitment: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )
        self.quantizer = VectorQuantizer(
            codebook_size, embedding_dim, commitment
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, image_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(x))

    def decode_indices(self, indices: Tensor) -> Tensor:
        return self.decoder(self.quantizer.lookup(indices))

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, quantizer_loss, diagnostics = self.encode(x)
        return self.decoder(z_st), indices, quantizer_loss, diagnostics


class MaskedConv2d(nn.Conv2d):
    """PixelCNN mask: A excludes the current token, B includes it."""

    def __init__(self, mask_type: str, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        mask = torch.ones_like(self.weight)
        center_h, center_w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        first_blocked = center_w if mask_type == "A" else center_w + 1
        mask[:, :, center_h, first_blocked:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNPrior(nn.Module):
    """A compact causal prior for the 8x8 code index grid."""

    def __init__(
        self, codebook_size: int, hidden_channels: int = 128, layers: int = 7
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(codebook_size, hidden_channels)
        blocks: list[nn.Module] = [
            MaskedConv2d(
                "A", hidden_channels, hidden_channels, 7, padding=3
            ),
            nn.ReLU(inplace=True),
        ]
        for _ in range(layers - 1):
            blocks.extend(
                [
                    MaskedConv2d(
                        "B", hidden_channels, hidden_channels, 3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
        blocks.extend(
            [
                nn.Conv2d(hidden_channels, hidden_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, codebook_size, 1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, indices: Tensor) -> Tensor:
        embedded = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.net(embedded)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        height: int,
        width: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
        for row in range(height):
            for column in range(width):
                logits = self(indices)[:, :, row, column] / temperature
                indices[:, row, column] = torch.multinomial(
                    logits.softmax(dim=1), 1
                ).squeeze(1)
        return indices


def tokenizer_loss(
    reconstruction: Tensor, x: Tensor, quantizer_loss: Tensor
) -> tuple[Tensor, Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, x)
    return reconstruction_loss + quantizer_loss, reconstruction_loss.detach()


def smoke_test() -> None:
    torch.manual_seed(7)
    tokenizer = VQVAE(
        hidden_channels=32, embedding_dim=16, codebook_size=32
    )
    prior = PixelCNNPrior(codebook_size=32, hidden_channels=24, layers=3)
    x = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    reconstruction, indices, quantizer_loss, diagnostics = tokenizer(x)
    loss, _ = tokenizer_loss(reconstruction, x, quantizer_loss)
    loss.backward()
    logits = prior(indices.detach())
    prior_loss = F.cross_entropy(logits, indices)
    prior_loss.backward()
    decoded = tokenizer.decode_indices(indices)
    assert reconstruction.shape == x.shape == decoded.shape
    assert indices.shape == (4, 8, 8)
    assert logits.shape == (4, 32, 8, 8)
    assert tokenizer.encoder[0].weight.grad is not None
    print(
        f"smoke test passed: tokenizer_loss={loss.item():.3f}, "
        f"prior_loss={prior_loss.item():.3f}, "
        f"perplexity={diagnostics['perplexity'].item():.2f}, "
        f"active={diagnostics['active_codes'].item()}"
    )


def make_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
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
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def train_tokenizer(
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> VQVAE:
    model = VQVAE(
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.tokenizer_epochs + 1):
        sums = torch.zeros(4, device=device)
        examples = 0
        model.train()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, _, quantizer_loss, diagnostics = model(x)
            loss, reconstruction_loss = tokenizer_loss(
                reconstruction, x, quantizer_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = x.shape[0]
            sums += torch.stack(
                [
                    loss.detach(),
                    reconstruction_loss,
                    diagnostics["perplexity"],
                    diagnostics["active_codes"].float(),
                ]
            ) * batch_size
            examples += batch_size
        means = (sums / examples).tolist()
        print(
            f"tokenizer epoch {epoch:03d}: loss={means[0]:.4f}, "
            f"recon={means[1]:.4f}, perplexity={means[2]:.1f}, "
            f"active_codes={means[3]:.1f}/{args.codebook_size}"
        )
        save_image(
            torch.cat([x[:8], reconstruction[:8]]).mul(0.5).add(0.5),
            out_dir / f"tokenizer_epoch_{epoch:03d}.png",
            nrow=8,
        )

    torch.save(
        {
            "model": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "codebook_size": args.codebook_size,
        },
        out_dir / "vq_vae.pth",
    )
    return model


def load_tokenizer(
    args: argparse.Namespace, device: torch.device, checkpoint_path: Path
) -> VQVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VQVAE(
        embedding_dim=checkpoint["embedding_dim"],
        codebook_size=checkpoint["codebook_size"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    if checkpoint["codebook_size"] != args.codebook_size:
        raise ValueError("checkpoint and --codebook-size disagree")
    return model


def train_prior(
    args: argparse.Namespace,
    tokenizer: VQVAE,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> PixelCNNPrior:
    tokenizer.eval().requires_grad_(False)
    prior = PixelCNNPrior(args.codebook_size).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.prior_lr)
    grid_height = grid_width = 8
    for epoch in range(1, args.prior_epochs + 1):
        total_loss = 0.0
        examples = 0
        prior.train()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.inference_mode():
                _, indices, _, _ = tokenizer.encode(x)
            logits = prior(indices)
            loss = F.cross_entropy(logits, indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
            examples += x.shape[0]
        bits_per_token = total_loss / examples / math.log(2.0)
        print(
            f"prior epoch {epoch:03d}: nll={total_loss / examples:.4f}, "
            f"bits/token={bits_per_token:.3f}"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            prior.eval()
            sampled_indices = prior.sample(
                64,
                grid_height,
                grid_width,
                device,
                temperature=args.temperature,
            )
            samples = tokenizer.decode_indices(sampled_indices)
            save_image(
                samples.mul(0.5).add(0.5),
                out_dir / f"prior_epoch_{epoch:03d}.png",
                nrow=8,
            )

    torch.save(
        {"model": prior.state_dict(), "codebook_size": args.codebook_size},
        out_dir / "pixelcnn_prior.pth",
    )
    return prior


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vq_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args, device)
    checkpoint_path = out_dir / "vq_vae.pth"

    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(args, loader, device, out_dir)
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{checkpoint_path} does not exist; train stage 1 first"
            )
        tokenizer = load_tokenizer(args, device, checkpoint_path)

    if args.stage in {"prior", "all"}:
        train_prior(args, tokenizer, loader, device, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--stage", choices=("tokenizer", "prior", "all"), default="all"
    )
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=5)
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
