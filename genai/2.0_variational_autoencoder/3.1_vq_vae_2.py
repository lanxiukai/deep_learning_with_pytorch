"""VQ-VAE-2: add two spatial scales to the discrete VQ-VAE bottleneck.

For a 32x32 teaching input, this compact tokenizer produces a 4x4 top grid
and an 8x8 bottom grid.  The top code is decoded to the bottom resolution and
conditions both bottom quantization and final reconstruction.  After freezing
the tokenizer, generation factorizes as

    p(top, bottom) = p(top) p(bottom | top).

The priors are deliberately small PixelCNNs rather than the paper's large
PixelSNAIL-inspired systems; they preserve the two-stage, top-before-bottom
algorithm while fitting comfortably on a 12 GB card.  A coarse/fine semantic
split is an empirical outcome to inspect, not a structural guarantee.  To
keep hierarchy as the visible increment, this lesson reuses 3.0's gradient-
updated codebooks; the paper tokenizer used EMA codebook updates.
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
from dl_utils.vae.quantization import (
    ImageDecoder32,
    ImageEncoder32,
    ResidualBlock,
    TokenUsageAccumulator,
    VectorQuantizer,
)
from dl_utils.vae.token_prior import PixelCNNPrior


PROJECT_ROOT = infer_project_root()


class VQVAE2(nn.Module):
    """Two-level VQ tokenizer with 4x4 top and 8x8 bottom index grids."""

    def __init__(
        self,
        *,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        top_codebook_size: int = 512,
        bottom_codebook_size: int = 512,
        commitment: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.bottom_encoder = ImageEncoder32(
            hidden_channels, hidden_channels=hidden_channels
        )
        self.top_encoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )
        self.top_quantizer = VectorQuantizer(
            top_codebook_size, embedding_dim, commitment
        )
        self.top_to_bottom = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
        )
        self.bottom_projection = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )
        self.bottom_quantizer = VectorQuantizer(
            bottom_codebook_size, embedding_dim, commitment
        )
        self.decoder = ImageDecoder32(
            hidden_channels + embedding_dim, hidden_channels=hidden_channels
        )

    def encode(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        bottom_features = self.bottom_encoder(x)
        top_e = self.top_encoder(bottom_features)
        top_st, top_indices, top_loss, top_diagnostics = self.top_quantizer(top_e)
        top_condition = self.top_to_bottom(top_st)
        bottom_e = self.bottom_projection(
            torch.cat((bottom_features, top_condition), dim=1)
        )
        bottom_st, bottom_indices, bottom_loss, bottom_diagnostics = (
            self.bottom_quantizer(bottom_e)
        )
        diagnostics = {
            "top_perplexity": top_diagnostics["perplexity"],
            "top_active": top_diagnostics["active_codes"],
            "top_quantization_mse": top_diagnostics["quantization_mse"],
            "bottom_perplexity": bottom_diagnostics["perplexity"],
            "bottom_active": bottom_diagnostics["active_codes"],
            "bottom_quantization_mse": bottom_diagnostics["quantization_mse"],
        }
        return (
            top_condition,
            bottom_st,
            top_indices,
            bottom_indices,
            top_loss + bottom_loss,
            diagnostics,
        )

    def top_condition_from_indices(self, top_indices: Tensor) -> Tensor:
        return self.top_to_bottom(self.top_quantizer.lookup(top_indices))

    def decode_indices(self, top_indices: Tensor, bottom_indices: Tensor) -> Tensor:
        top_condition = self.top_condition_from_indices(top_indices)
        bottom = self.bottom_quantizer.lookup(bottom_indices)
        return self.decoder(torch.cat((top_condition, bottom), dim=1))

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        top_condition, bottom, top_indices, bottom_indices, vq_loss, diagnostics = (
            self.encode(x)
        )
        reconstruction = self.decoder(torch.cat((top_condition, bottom), dim=1))
        return reconstruction, top_indices, bottom_indices, vq_loss, diagnostics


def smoke_test() -> None:
    torch.manual_seed(7)
    tokenizer = VQVAE2(
        hidden_channels=32,
        embedding_dim=8,
        top_codebook_size=16,
        bottom_codebook_size=16,
    )
    top_prior = PixelCNNPrior(16, hidden_channels=24, layers=2)
    bottom_prior = PixelCNNPrior(
        16, hidden_channels=24, layers=2, condition_channels=32
    )
    x = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    reconstruction, top, bottom, vq_loss, diagnostics = tokenizer(x)
    tokenizer_loss = F.mse_loss(reconstruction, x) + vq_loss
    tokenizer_loss.backward()
    condition = tokenizer.top_condition_from_indices(top.detach())
    top_loss = F.cross_entropy(top_prior(top.detach()), top.detach())
    bottom_loss = F.cross_entropy(
        bottom_prior(bottom.detach(), condition.detach()), bottom.detach()
    )
    (top_loss + bottom_loss).backward()
    decoded = tokenizer.decode_indices(top, bottom)
    assert reconstruction.shape == decoded.shape == x.shape
    assert top.shape == (4, 4, 4)
    assert bottom.shape == (4, 8, 8)
    assert tokenizer.top_encoder[0].weight.grad is not None
    print(
        f"smoke test passed: tokenizer={tokenizer_loss.item():.3f}, "
        f"top_prior={top_loss.item():.3f}, bottom_prior={bottom_loss.item():.3f}, "
        f"PPL(top,bottom)=({diagnostics['top_perplexity'].item():.1f},"
        f"{diagnostics['bottom_perplexity'].item():.1f})"
    )


def make_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = datasets.CIFAR10(
        PROJECT_ROOT / "data" / "cifar10",
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
) -> VQVAE2:
    config = {
        "hidden_channels": args.hidden_channels,
        "embedding_dim": args.embedding_dim,
        "top_codebook_size": args.top_codebook_size,
        "bottom_codebook_size": args.bottom_codebook_size,
        "commitment": args.commitment,
    }
    model = VQVAE2(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fixed_bits = (
        4 * 4 * math.ceil(math.log2(args.top_codebook_size))
        + 8 * 8 * math.ceil(math.log2(args.bottom_codebook_size))
    )
    print(f"two-level fixed-length upper bound={fixed_bits} bits/image")
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        sums = torch.zeros(5, device=device)
        examples = 0
        top_usage = TokenUsageAccumulator(args.top_codebook_size)
        bottom_usage = TokenUsageAccumulator(args.bottom_codebook_size)
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, top, bottom, vq_loss, diagnostics = model(x)
            distortion = F.mse_loss(reconstruction, x)
            loss = distortion + vq_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                distortion.detach(),
                vq_loss.detach(),
                diagnostics["top_quantization_mse"],
                diagnostics["bottom_quantization_mse"],
            ]
            sums += torch.stack(values) * x.shape[0]
            top_usage.update(top)
            bottom_usage.update(bottom)
            examples += x.shape[0]
        means = (sums / examples).tolist()
        top_statistics = top_usage.statistics()
        bottom_statistics = bottom_usage.statistics()
        print(
            f"tokenizer {epoch:03d}: loss={means[0]:.4f}, D={means[1]:.4f}, "
            f"VQ={means[2]:.4f}, quant(top,bottom)=({means[3]:.4f},{means[4]:.4f}), "
            f"top PPL/active={top_statistics['perplexity'].item():.1f}/"
            f"{top_statistics['active_codes'].item()}, bottom PPL/active="
            f"{bottom_statistics['perplexity'].item():.1f}/"
            f"{bottom_statistics['active_codes'].item()}"
        )
        save_image(
            torch.cat((x[:8], reconstruction[:8])).mul(0.5).add(0.5),
            out_dir / f"tokenizer_{epoch:03d}.png",
            nrow=8,
        )
    torch.save(
        {
            "format_version": 1,
            "model_name": "vq_vae_2_tokenizer",
            "state_dict": model.state_dict(),
            "model_config": config,
            "top_grid": (4, 4),
            "bottom_grid": (8, 8),
            "index_order": "row-major",
        },
        out_dir / "tokenizer.pth",
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(path: Path, device: torch.device) -> VQVAE2:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "vq_vae_2_tokenizer":
        raise ValueError("checkpoint is not a VQ-VAE-2 tokenizer")
    model = VQVAE2(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def train_priors(
    args: argparse.Namespace,
    tokenizer: VQVAE2,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    tokenizer.eval().requires_grad_(False)
    top_vocabulary = tokenizer.top_quantizer.codebook_size
    bottom_vocabulary = tokenizer.bottom_quantizer.codebook_size
    top_prior = PixelCNNPrior(
        top_vocabulary,
        hidden_channels=args.prior_hidden_channels,
        layers=args.prior_layers,
    ).to(device)
    bottom_prior = PixelCNNPrior(
        bottom_vocabulary,
        hidden_channels=args.prior_hidden_channels,
        layers=args.prior_layers,
        condition_channels=tokenizer.hidden_channels,
    ).to(device)
    top_optimizer = torch.optim.Adam(top_prior.parameters(), lr=args.prior_lr)
    bottom_optimizer = torch.optim.Adam(bottom_prior.parameters(), lr=args.prior_lr)

    for epoch in range(1, args.prior_epochs + 1):
        top_prior.train()
        bottom_prior.train()
        top_sum = bottom_sum = 0.0
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.inference_mode():
                _, _, top, bottom, _, _ = tokenizer.encode(x)
                condition = tokenizer.top_condition_from_indices(top)

            top_loss = F.cross_entropy(top_prior(top), top)
            top_optimizer.zero_grad(set_to_none=True)
            top_loss.backward()
            top_optimizer.step()

            bottom_loss = F.cross_entropy(bottom_prior(bottom, condition), bottom)
            bottom_optimizer.zero_grad(set_to_none=True)
            bottom_loss.backward()
            bottom_optimizer.step()
            top_sum += top_loss.item() * x.shape[0]
            bottom_sum += bottom_loss.item() * x.shape[0]
            examples += x.shape[0]

        top_nll, bottom_nll = top_sum / examples, bottom_sum / examples
        print(
            f"priors {epoch:03d}: train top={top_nll / math.log(2):.3f} "
            f"bits/token, train bottom|top={bottom_nll / math.log(2):.3f} "
            "bits/token"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            top_prior.eval()
            bottom_prior.eval()
            top = top_prior.sample(
                64, 4, 4, device=device, temperature=args.temperature
            )
            condition = tokenizer.top_condition_from_indices(top)
            bottom = bottom_prior.sample(
                64,
                8,
                8,
                device=device,
                condition=condition,
                temperature=args.temperature,
            )
            samples = tokenizer.decode_indices(top, bottom)
            save_image(
                samples.mul(0.5).add(0.5),
                out_dir / f"priors_{epoch:03d}.png",
                nrow=8,
            )

    common = {"format_version": 1, "tokenizer_checkpoint": "tokenizer.pth"}
    torch.save(
        {
            **common,
            "model_name": "vq_vae_2_top_prior",
            "state_dict": top_prior.state_dict(),
            "model_config": {
                "vocabulary_size": top_vocabulary,
                "hidden_channels": args.prior_hidden_channels,
                "layers": args.prior_layers,
            },
        },
        out_dir / "top_prior.pth",
    )
    torch.save(
        {
            **common,
            "model_name": "vq_vae_2_bottom_conditional_prior",
            "state_dict": bottom_prior.state_dict(),
            "model_config": {
                "vocabulary_size": bottom_vocabulary,
                "hidden_channels": args.prior_hidden_channels,
                "layers": args.prior_layers,
                "condition_channels": tokenizer.hidden_channels,
            },
        },
        out_dir / "bottom_prior.pth",
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vq_vae_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args, device)
    path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(args, loader, device, out_dir)
    else:
        if not path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the priors")
        tokenizer = load_tokenizer(path, device)
    if args.stage in {"prior", "all"}:
        train_priors(args, tokenizer, loader, device, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stage", choices=("tokenizer", "prior", "all"), default="all")
    parser.add_argument("--tokenizer-epochs", type=int, default=30)
    parser.add_argument("--prior-epochs", type=int, default=30)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--top-codebook-size", type=int, default=512)
    parser.add_argument("--bottom-codebook-size", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--prior-hidden-channels", type=int, default=128)
    parser.add_argument("--prior-layers", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--prior-lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
