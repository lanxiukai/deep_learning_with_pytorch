"""FSQ: replace learned vector lookup with fixed scalar quantization.

This is a parallel tokenizer branch, not a successor to VQ-VAE-2.  Relative
to ``6.0_vq_vae.py``, the encoder, decoder, straight-through gradient, frozen
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
from torchvision.utils import save_image

from dl_utils.data.cifar10 import (
    make_cifar10_loader,
    normalized_cifar10_transform,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import (
    model_state_fingerprint,
    reproducibility_metadata,
)
from dl_utils.vae.quantization import FSQAutoencoder32, TokenUsageAccumulator
from dl_utils.vae.token_prior import PixelCNNPrior


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


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    transform = normalized_cifar10_transform(horizontal_flip=False)
    data_root = PROJECT_ROOT / "data" / "cifar10"
    return (
        make_cifar10_loader(
            data_root,
            args.batch_size,
            device,
            train=True,
            transform=transform,
            num_workers=args.workers,
        ),
        make_cifar10_loader(
            data_root,
            args.batch_size,
            device,
            train=False,
            transform=transform,
            num_workers=args.workers,
        ),
    )


@torch.inference_mode()
def evaluate_tokenizer(
    model: FSQAutoencoder32,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    vocabulary_size = model.quantizer.codebook_size
    distortion = 0.0
    quantization_mse = 0.0
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, indices, diagnostics = model(x)
        distortion += float(F.mse_loss(reconstruction, x)) * x.shape[0]
        quantization_mse += (
            float(diagnostics["quantization_mse"]) * x.shape[0]
        )
        usage.update(indices)
        examples += x.shape[0]
    statistics = usage.statistics()
    entropy_bits = float(statistics["token_entropy_nats"]) / math.log(2)
    return {
        "mse": distortion / examples,
        "quantization_mse": quantization_mse / examples,
        "perplexity": float(statistics["perplexity"]),
        "active_codes": float(statistics["active_codes"]),
        "usage_fraction": float(statistics["usage_fraction"]),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": 64 * entropy_bits,
        "fixed_length_bits_per_image": (
            64 * math.ceil(math.log2(vocabulary_size))
        ),
    }


@torch.inference_mode()
def evaluate_prior(
    tokenizer: FSQAutoencoder32,
    prior: PixelCNNPrior,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    prior.eval()
    nll = 0.0
    examples = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, indices, _ = tokenizer.encode(x)
        loss = F.cross_entropy(prior(indices), indices)
        nll += float(loss) * x.shape[0]
        examples += x.shape[0]
    nll /= examples
    return {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": 64 * nll / math.log(2),
    }


def train_tokenizer(
    args: argparse.Namespace,
    train_loader: DataLoader,
    validation_loader: DataLoader,
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
        for x, _ in train_loader:
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
        if epoch == 1 or epoch % args.sample_every == 0:
            save_image(
                torch.cat((x[:8], reconstruction[:8])).mul(0.5).add(0.5),
                out_dir / f"tokenizer_{epoch:03d}.png",
                nrow=8,
            )
    validation = evaluate_tokenizer(
        model, validation_loader, device=device
    )
    interface_id = model_state_fingerprint(model)
    torch.save(
        {
            "format_version": 2,
            "model_name": "fsq_tokenizer",
            "roadmap_role": "focused_branch",
            "roadmap_step": "7X",
            "direct_baseline": "VQ-VAE tokenizer interface",
            "visible_increment": "fixed scalar levels without a codebook loss",
            "interface_id": interface_id,
            "state_dict": model.state_dict(),
            "model_config": config,
            "vocabulary_size": vocabulary_size,
            "token_grid": (8, 8),
            "index_order": "row-major",
            "dataset": "CIFAR-10",
            "image_size": 32,
            "value_range": [-1.0, 1.0],
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"fsq_tokenizer": model},
                seed=args.seed,
                training_budget={
                    "epochs": args.tokenizer_epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.tokenizer_epochs
                    * len(train_loader),
                },
                data_preprocessing={
                    "spatial": "native 32x32",
                    "value_range": [-1.0, 1.0],
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "tokenizer.pth",
    )
    print(
        f"validation tokenizer: MSE={validation['mse']:.4f}, "
        f"entropy={validation['marginal_entropy_bits_per_token']:.3f} "
        f"bits/token, active={validation['active_codes']:.0f}/"
        f"{vocabulary_size}"
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(path: Path, device: torch.device) -> FSQAutoencoder32:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "fsq_tokenizer":
        raise ValueError("checkpoint is not an FSQ tokenizer")
    model = FSQAutoencoder32(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("interface_id") != model_state_fingerprint(model):
        raise ValueError(
            "tokenizer checkpoint has no valid interface identity; retrain stage 1"
        )
    return model.eval().requires_grad_(False)


def train_prior(
    args: argparse.Namespace,
    tokenizer: FSQAutoencoder32,
    train_loader: DataLoader,
    validation_loader: DataLoader,
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
        for x, _ in train_loader:
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
    validation = evaluate_prior(
        tokenizer, prior, validation_loader, device=device
    )
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    torch.save(
        {
            "format_version": 2,
            "model_name": "fsq_pixelcnn_prior",
            "roadmap_role": "focused_branch_stage_two",
            "roadmap_step": "7X",
            "direct_baseline": "frozen FSQ tokenizer",
            "visible_increment": "autoregressive prior over frozen FSQ tokens",
            "state_dict": prior.state_dict(),
            "model_config": {
                "vocabulary_size": vocabulary_size,
                "hidden_channels": args.prior_hidden_channels,
                "layers": args.prior_layers,
            },
            "tokenizer_checkpoint": "tokenizer.pth",
            "tokenizer_interface_id": tokenizer_interface_id,
            "dataset": "CIFAR-10",
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"fsq_pixelcnn_prior": prior},
                seed=args.seed,
                training_budget={
                    "epochs": args.prior_epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.prior_epochs
                    * len(train_loader),
                },
                data_preprocessing={
                    "tokenizer": "frozen FSQ",
                    "token_grid": [8, 8],
                    "index_order": "row-major",
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "pixelcnn_prior.pth",
    )
    print(
        f"validation prior: {validation['bits_per_token']:.3f} "
        f"bits/token, {validation['bits_per_image']:.1f} bits/image"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "fsq"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders(args, device)
    path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(
            args, train_loader, validation_loader, device, out_dir
        )
    else:
        if not path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the prior")
        tokenizer = load_tokenizer(path, device)
    if args.stage in {"prior", "all"}:
        train_prior(
            args,
            tokenizer,
            train_loader,
            validation_loader,
            device,
            out_dir,
        )


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
    set_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
