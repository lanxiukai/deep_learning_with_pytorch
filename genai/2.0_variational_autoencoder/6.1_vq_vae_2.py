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
keep hierarchy as the visible increment, this lesson reuses 6.0's gradient-
updated codebooks; the paper tokenizer used EMA codebook updates.
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
from dl_utils.vae.quantization import TokenUsageAccumulator, VQVAE2
from dl_utils.vae.token_prior import PixelCNNPrior


PROJECT_ROOT = infer_project_root()


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
    model: VQVAE2,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    top_vocabulary = model.top_quantizer.codebook_size
    bottom_vocabulary = model.bottom_quantizer.codebook_size
    distortion = 0.0
    top_quantization = 0.0
    bottom_quantization = 0.0
    examples = 0
    top_usage = TokenUsageAccumulator(top_vocabulary)
    bottom_usage = TokenUsageAccumulator(bottom_vocabulary)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, top, bottom, _, diagnostics = model(x)
        distortion += float(F.mse_loss(reconstruction, x)) * x.shape[0]
        top_quantization += (
            float(diagnostics["top_quantization_mse"]) * x.shape[0]
        )
        bottom_quantization += (
            float(diagnostics["bottom_quantization_mse"]) * x.shape[0]
        )
        top_usage.update(top)
        bottom_usage.update(bottom)
        examples += x.shape[0]
    top_statistics = top_usage.statistics()
    bottom_statistics = bottom_usage.statistics()
    top_entropy = (
        float(top_statistics["token_entropy_nats"]) / math.log(2)
    )
    bottom_entropy = (
        float(bottom_statistics["token_entropy_nats"]) / math.log(2)
    )
    return {
        "mse": distortion / examples,
        "top_quantization_mse": top_quantization / examples,
        "bottom_quantization_mse": bottom_quantization / examples,
        "top_perplexity": float(top_statistics["perplexity"]),
        "bottom_perplexity": float(bottom_statistics["perplexity"]),
        "top_active_codes": float(top_statistics["active_codes"]),
        "bottom_active_codes": float(bottom_statistics["active_codes"]),
        "top_entropy_bits_per_token": top_entropy,
        "bottom_entropy_bits_per_token": bottom_entropy,
        "marginal_entropy_bits_per_image": (
            16 * top_entropy + 64 * bottom_entropy
        ),
        "fixed_length_bits_per_image": (
            16 * math.ceil(math.log2(top_vocabulary))
            + 64 * math.ceil(math.log2(bottom_vocabulary))
        ),
    }


@torch.inference_mode()
def evaluate_priors(
    tokenizer: VQVAE2,
    top_prior: PixelCNNPrior,
    bottom_prior: PixelCNNPrior,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    top_prior.eval()
    bottom_prior.eval()
    top_nll = 0.0
    bottom_nll = 0.0
    examples = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, _, top, bottom, _, _ = tokenizer.encode(x)
        condition = tokenizer.top_condition_from_indices(top)
        top_loss = F.cross_entropy(top_prior(top), top)
        bottom_loss = F.cross_entropy(
            bottom_prior(bottom, condition), bottom
        )
        top_nll += float(top_loss) * x.shape[0]
        bottom_nll += float(bottom_loss) * x.shape[0]
        examples += x.shape[0]
    top_nll /= examples
    bottom_nll /= examples
    return {
        "top_nll_nats_per_token": top_nll,
        "bottom_nll_nats_per_token": bottom_nll,
        "top_bits_per_token": top_nll / math.log(2),
        "bottom_bits_per_token": bottom_nll / math.log(2),
        "bits_per_image": (
            16 * top_nll / math.log(2)
            + 64 * bottom_nll / math.log(2)
        ),
    }


def train_tokenizer(
    args: argparse.Namespace,
    train_loader: DataLoader,
    validation_loader: DataLoader,
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
        for x, _ in train_loader:
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
            "model_name": "vq_vae_2_tokenizer",
            "roadmap_role": "historical_anchor",
            "roadmap_step": 7,
            "direct_baseline": "single-level VQ-VAE",
            "visible_increment": "top and bottom discrete latent hierarchy",
            "interface_id": interface_id,
            "state_dict": model.state_dict(),
            "model_config": config,
            "top_grid": (4, 4),
            "bottom_grid": (8, 8),
            "index_order": "row-major",
            "dataset": "CIFAR-10",
            "image_size": 32,
            "value_range": [-1.0, 1.0],
            "uniform_prior_kl_nats_per_image": (
                16 * math.log(args.top_codebook_size)
                + 64 * math.log(args.bottom_codebook_size)
            ),
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"vq_vae_2_tokenizer": model},
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
        f"entropy={validation['marginal_entropy_bits_per_image']:.1f} "
        f"bits/image, active(top,bottom)="
        f"({validation['top_active_codes']:.0f},"
        f"{validation['bottom_active_codes']:.0f})"
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(path: Path, device: torch.device) -> VQVAE2:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "vq_vae_2_tokenizer":
        raise ValueError("checkpoint is not a VQ-VAE-2 tokenizer")
    model = VQVAE2(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("interface_id") != model_state_fingerprint(model):
        raise ValueError(
            "tokenizer checkpoint has no valid interface identity; retrain stage 1"
        )
    return model.eval().requires_grad_(False)


def train_priors(
    args: argparse.Namespace,
    tokenizer: VQVAE2,
    train_loader: DataLoader,
    validation_loader: DataLoader,
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
        for x, _ in train_loader:
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

    validation = evaluate_priors(
        tokenizer,
        top_prior,
        bottom_prior,
        validation_loader,
        device=device,
    )
    tokenizer_interface_id = model_state_fingerprint(tokenizer)
    common = {
        "format_version": 2,
        "roadmap_role": "historical_anchor_stage_two",
        "roadmap_step": 7,
        "direct_baseline": "frozen hierarchical VQ-VAE-2 tokenizer",
        "tokenizer_checkpoint": "tokenizer.pth",
        "tokenizer_interface_id": tokenizer_interface_id,
        "dataset": "CIFAR-10",
        "validation_metrics": validation,
    }
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
            "visible_increment": "autoregressive top-token prior",
            **reproducibility_metadata(
                models={"vq_vae_2_top_prior": top_prior},
                seed=args.seed,
                training_budget={
                    "epochs": args.prior_epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.prior_epochs
                    * len(train_loader),
                },
                data_preprocessing={
                    "tokenizer": "frozen VQ-VAE-2",
                    "token_grid": [4, 4],
                    "index_order": "row-major",
                    "augmentation": "none",
                },
            ),
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
            "visible_increment": "bottom-token prior conditioned on top tokens",
            **reproducibility_metadata(
                models={
                    "vq_vae_2_bottom_conditional_prior": bottom_prior
                },
                seed=args.seed,
                training_budget={
                    "epochs": args.prior_epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.prior_epochs
                    * len(train_loader),
                },
                data_preprocessing={
                    "tokenizer": "frozen VQ-VAE-2",
                    "token_grid": [8, 8],
                    "top_condition_grid": [4, 4],
                    "index_order": "row-major",
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "bottom_prior.pth",
    )
    print(
        f"validation priors: top={validation['top_bits_per_token']:.3f}, "
        f"bottom|top={validation['bottom_bits_per_token']:.3f} "
        f"bits/token, total={validation['bits_per_image']:.1f} bits/image"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vae" / "vq_vae_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders(args, device)
    path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(
            args, train_loader, validation_loader, device, out_dir
        )
    else:
        if not path.is_file():
            raise FileNotFoundError("train --stage tokenizer before the priors")
        tokenizer = load_tokenizer(path, device)
    if args.stage in {"prior", "all"}:
        train_priors(
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
    set_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
