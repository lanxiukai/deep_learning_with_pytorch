"""FSQ: replace learned vector lookup with fixed scalar quantization.

This is a parallel tokenizer branch, not a successor to VQ-VAE-2.  Relative
to ``3.0_vq_vae.py``, the encoder, decoder, straight-through gradient, frozen
tokenizer boundary, and second-stage PixelCNN remain.  The quantizer changes:

* each low-dimensional channel is bounded and rounded to fixed levels;
* mixed-radix packing maps the scalar tuple to one integer token;
* no learned codebook, codebook loss, or commitment loss exists.

Removing dead *entries* does not guarantee that the encoder visits every
scalar combination, so usage entropy and reconstruction are still logged.
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
from dl_utils.genai.quantization import FSQAutoencoder32, TokenUsageAccumulator
from dl_utils.genai.token_prior import PixelCNNPrior


PROJECT_ROOT = infer_project_root()


def parse_levels(text: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "levels must be comma-separated integers"
        ) from error
    if not levels or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError("every level count must be >= 2")
    return levels


def smoke_test() -> None:
    torch.manual_seed(7)
    model = FSQAutoencoder32(levels=(4, 4, 4), hidden_channels=32)
    prior = PixelCNNPrior(64, hidden_channels=24, layers=3)
    x = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    reconstruction, indices, diagnostics = model(x)
    loss = F.mse_loss(reconstruction, x)
    loss.backward()
    prior_loss = F.cross_entropy(prior(indices.detach()), indices.detach())
    prior_loss.backward()

    all_indices = torch.arange(model.quantizer.codebook_size)
    restored = model.quantizer.pack(model.quantizer.unpack(all_indices))
    assert torch.equal(all_indices, restored), "mixed-radix mapping is not reversible"
    assert reconstruction.shape == x.shape
    assert indices.shape == (4, 8, 8)
    assert model.encoder.net[0].weight.grad is not None
    print(
        f"smoke test passed: D={loss.item():.3f}, prior={prior_loss.item():.3f}, "
        f"vocabulary={model.quantizer.codebook_size}, "
        f"PPL={diagnostics['perplexity'].item():.2f}, "
        f"active={diagnostics['active_codes'].item()}"
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
) -> FSQAutoencoder32:
    config = {"levels": args.levels, "hidden_channels": args.hidden_channels}
    model = FSQAutoencoder32(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    vocabulary_size = model.quantizer.codebook_size
    fixed_bits = 8 * 8 * math.ceil(math.log2(vocabulary_size))
    print(
        f"FSQ levels={args.levels}, vocabulary={vocabulary_size}, "
        f"fixed-length upper bound={fixed_bits} bits/image"
    )
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        sums = torch.zeros(2, device=device)
        examples = 0
        usage = TokenUsageAccumulator(vocabulary_size)
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, diagnostics = model(x)
            loss = F.mse_loss(reconstruction, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                diagnostics["quantization_mse"],
            ]
            sums += torch.stack(values) * x.shape[0]
            usage.update(indices)
            examples += x.shape[0]
        means = (sums / examples).tolist()
        epoch_usage = usage.statistics()
        print(
            f"tokenizer {epoch:03d}: D={means[0]:.4f}, quant={means[1]:.4f}, "
            f"PPL={epoch_usage['perplexity'].item():.1f}, "
            f"active={epoch_usage['active_codes'].item()}/{vocabulary_size}, "
            f"epoch marginal entropy="
            f"{epoch_usage['token_entropy_nats'].item() / math.log(2):.2f} bits/token"
        )
        save_image(
            torch.cat((x[:8], reconstruction[:8])).mul(0.5).add(0.5),
            out_dir / f"tokenizer_{epoch:03d}.png",
            nrow=8,
        )
    torch.save(
        {
            "format_version": 1,
            "model_name": "fsq_tokenizer",
            "state_dict": model.state_dict(),
            "model_config": config,
            "vocabulary_size": vocabulary_size,
            "token_grid": (8, 8),
            "index_order": "row-major",
        },
        out_dir / "tokenizer.pth",
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(path: Path, device: torch.device) -> FSQAutoencoder32:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "fsq_tokenizer":
        raise ValueError("checkpoint is not an FSQ tokenizer")
    model = FSQAutoencoder32(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def train_prior(
    args: argparse.Namespace,
    tokenizer: FSQAutoencoder32,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
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
                _, indices, _ = tokenizer.encode(x)
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
            "model_name": "fsq_pixelcnn_prior",
            "state_dict": prior.state_dict(),
            "model_config": {
                "vocabulary_size": vocabulary_size,
                "hidden_channels": args.prior_hidden_channels,
                "layers": args.prior_layers,
            },
            "tokenizer_checkpoint": "tokenizer.pth",
        },
        out_dir / "pixelcnn_prior.pth",
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "fsq"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args, device)
    path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(args, loader, device, out_dir)
    else:
        if not path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the prior")
        tokenizer = load_tokenizer(path, device)
    if args.stage in {"prior", "all"}:
        train_prior(args, tokenizer, loader, device, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stage", choices=("tokenizer", "prior", "all"), default="all")
    parser.add_argument("--levels", type=parse_levels, default=(8, 8, 5, 5))
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--prior-epochs", type=int, default=20)
    parser.add_argument("--hidden-channels", type=int, default=128)
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
