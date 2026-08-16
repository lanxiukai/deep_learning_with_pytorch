"""VQ-VAE: learn a discrete tokenizer, then learn its token prior.

Stage 1 is the original three-way VQ objective:

    reconstruction + codebook loss + commitment loss

The decoder sees a nearest code in the forward pass, while the straight-
through estimator copies its gradient to the encoder.  Stage 2 freezes the
entire tokenizer and trains a compact causal PixelCNN on the 8x8 integer grid.
Thus reconstruction and unconditional generation remain visibly different
paths.  Under the stage-one fixed uniform prior, every position's KL is the
parameter-independent constant ``log(codebook_size)`` and is omitted.

Examples:
    python genai/2.0_variational_autoencoder/3.0_vq_vae.py --smoke-test
    python genai/2.0_variational_autoencoder/3.0_vq_vae.py --stage tokenizer
    python genai/2.0_variational_autoencoder/3.0_vq_vae.py --stage prior
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.quantization import TokenUsageAccumulator, VQVAE32
from dl_utils.vae.token_prior import PixelCNNPrior


PROJECT_ROOT = infer_project_root()


def tokenizer_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    quantizer_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distortion = F.mse_loss(reconstruction, x)
    return distortion + quantizer_loss, distortion.detach()


def smoke_test() -> None:
    torch.manual_seed(7)
    tokenizer = VQVAE32(
        hidden_channels=32, embedding_dim=16, codebook_size=32
    )
    prior = PixelCNNPrior(32, hidden_channels=24, layers=3)
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
    assert tokenizer.encoder.net[0].weight.grad is not None
    print(
        f"smoke test passed: tokenizer={loss.item():.3f}, "
        f"prior={prior_loss.item():.3f}, "
        f"PPL={diagnostics['perplexity'].item():.2f}, "
        f"active={diagnostics['active_codes'].item()}/32"
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
) -> VQVAE32:
    config = {
        "hidden_channels": args.hidden_channels,
        "embedding_dim": args.embedding_dim,
        "codebook_size": args.codebook_size,
        "commitment": args.commitment,
    }
    model = VQVAE32(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fixed_bits = 8 * 8 * math.ceil(math.log2(args.codebook_size))
    print(f"fixed-length upper bound={fixed_bits} bits/image")
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        examples = 0
        usage = TokenUsageAccumulator(args.codebook_size)
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, quantizer_loss, diagnostics = model(x)
            loss, distortion = tokenizer_loss(reconstruction, x, quantizer_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                distortion,
                diagnostics["codebook_loss"],
                diagnostics["commitment_loss"],
            ]
            sums += torch.stack(values) * x.shape[0]
            usage.update(indices)
            examples += x.shape[0]
        means = (sums / examples).tolist()
        epoch_usage = usage.statistics()
        print(
            f"tokenizer {epoch:03d}: loss={means[0]:.4f}, D={means[1]:.4f}, "
            f"code={means[2]:.4f}, commit={means[3]:.4f}, "
            f"PPL={epoch_usage['perplexity'].item():.1f}, "
            f"active={epoch_usage['active_codes'].item()}/{args.codebook_size}"
        )
        save_image(
            torch.cat((x[:8], reconstruction[:8])).mul(0.5).add(0.5),
            out_dir / f"tokenizer_{epoch:03d}.png",
            nrow=8,
        )
    torch.save(
        {
            "format_version": 1,
            "model_name": "vq_vae_tokenizer",
            "state_dict": model.state_dict(),
            "model_config": config,
            "token_grid": (8, 8),
            "index_order": "row-major",
        },
        out_dir / "tokenizer.pth",
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(checkpoint_path: Path, device: torch.device) -> VQVAE32:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "vq_vae_tokenizer":
        raise ValueError("checkpoint is not a VQ-VAE tokenizer")
    model = VQVAE32(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def train_prior(
    args: argparse.Namespace,
    tokenizer: VQVAE32,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> PixelCNNPrior:
    tokenizer.eval().requires_grad_(False)
    vocabulary_size = tokenizer.quantizer.codebook_size
    prior = PixelCNNPrior(
        vocabulary_size,
        hidden_channels=args.prior_hidden_channels,
        layers=args.prior_layers,
    ).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.prior_lr)
    for epoch in range(1, args.prior_epochs + 1):
        prior.train()
        nll_sum = 0.0
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.inference_mode():
                _, indices, _, _ = tokenizer.encode(x)
            loss = F.cross_entropy(prior(indices), indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            nll_sum += loss.item() * x.shape[0]
            examples += x.shape[0]
        nll = nll_sum / examples
        print(
            f"prior {epoch:03d}: train nll={nll:.4f}, "
            f"train bits/token={nll / math.log(2):.3f}"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            prior.eval()
            indices = prior.sample(
                64,
                8,
                8,
                device=device,
                temperature=args.temperature,
            )
            samples = tokenizer.decode_indices(indices)
            save_image(
                samples.mul(0.5).add(0.5),
                out_dir / f"prior_{epoch:03d}.png",
                nrow=8,
            )
    torch.save(
        {
            "format_version": 1,
            "model_name": "vq_vae_pixelcnn_prior",
            "state_dict": prior.state_dict(),
            "model_config": {
                "vocabulary_size": vocabulary_size,
                "hidden_channels": args.prior_hidden_channels,
                "layers": args.prior_layers,
            },
            "tokenizer_checkpoint": "tokenizer.pth",
            "token_grid": (8, 8),
        },
        out_dir / "pixelcnn_prior.pth",
    )
    return prior


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vq_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args, device)
    checkpoint_path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(args, loader, device, out_dir)
    else:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the prior")
        tokenizer = load_tokenizer(checkpoint_path, device)
    if args.stage in {"prior", "all"}:
        train_prior(args, tokenizer, loader, device, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stage", choices=("tokenizer", "prior", "all"), default="all")
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--prior-hidden-channels", type=int, default=128)
    parser.add_argument("--prior-layers", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
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
