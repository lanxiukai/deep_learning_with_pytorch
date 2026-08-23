"""VQGAN: VQ-VAE tokenizer plus perceptual and patch-adversarial losses.

The first stage adds three ideas to ``6.0_vq_vae.py``:

* frozen pretrained VGG features as a compact perceptual-loss proxy;
* a PatchGAN discriminator with hinge loss and delayed activation;
* a stopped gradient-norm ratio on the decoder's last layer to scale G loss.

The published VQGAN uses LPIPS; VGG features keep this repository dependency-
light while preserving the perceptual-gradient path.  Do not label the logged
value as LPIPS.  Stage 2 freezes the tokenizer and trains a compact causal
Transformer, retaining VQGAN's token-prior interface without its large-scale
sliding-window setup.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.cifar10 import (
    make_cifar10_loader,
    normalized_cifar10_transform,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import model_state_fingerprint
from dl_utils.vae.perceptual_autoencoder import (
    PatchDiscriminator32,
    RandomFeaturePerceptualLoss,
    VGGPerceptualLoss,
    VQPerceptualAutoencoder32,
    adaptive_adversarial_weight,
    discriminator_hinge_loss,
)
from dl_utils.vae.quantization import TokenUsageAccumulator
from dl_utils.vae.token_prior import CausalTransformerPrior


PROJECT_ROOT = infer_project_root()


def vqgan_autoencoder_step(
    model: VQPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    perceptual: nn.Module,
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
    perceptual_weight: float,
    vq_weight: float,
    discriminator_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    reconstruction, indices, vq_loss, diagnostics = model(x)
    pixel_l1 = F.l1_loss(reconstruction, x)
    feature_loss = perceptual(reconstruction, x)
    # VQGAN's adaptive ratio uses the reconstruction scalar, not VQ loss.
    reconstruction_objective = pixel_l1 + float(perceptual_weight) * feature_loss

    discriminator.requires_grad_(False)
    try:
        adversarial = -discriminator(reconstruction).mean()
        if step >= discriminator_start:
            adaptive = adaptive_adversarial_weight(
                reconstruction_objective,
                adversarial,
                model.decoder.last_layer,
                scale=discriminator_weight,
            )
        else:
            adaptive = x.new_zeros(())
        loss = (
            reconstruction_objective
            + float(vq_weight) * vq_loss
            + adaptive * adversarial
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    finally:
        discriminator.requires_grad_(True)
    return reconstruction.detach(), indices.detach(), {
        "autoencoder": loss.detach(),
        "pixel_l1": pixel_l1.detach(),
        "frozen_feature": feature_loss.detach(),
        "vq": vq_loss.detach(),
        "generator_adversarial": adversarial.detach(),
        "adaptive_weight": adaptive.detach(),
        "perplexity": diagnostics["perplexity"],
        "active_codes": diagnostics["active_codes"].float(),
    }


def vqgan_discriminator_step(
    discriminator: PatchDiscriminator32,
    x: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
) -> dict[str, Tensor]:
    if step < discriminator_start:
        zero = x.new_zeros(())
        return {"discriminator": zero, "real_logit": zero, "fake_logit": zero}
    real_logits = discriminator(x)
    fake_logits = discriminator(reconstruction.detach())
    loss = discriminator_hinge_loss(real_logits, fake_logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "discriminator": loss.detach(),
        "real_logit": real_logits.mean().detach(),
        "fake_logit": fake_logits.mean().detach(),
    }


def smoke_test() -> None:
    torch.manual_seed(7)
    model = VQPerceptualAutoencoder32(
        latent_channels=8, codebook_size=32, hidden_channels=32
    )
    discriminator = PatchDiscriminator32(base_channels=16)
    perceptual = RandomFeaturePerceptualLoss(channels=8)
    ae_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    reconstruction, indices, metrics = vqgan_autoencoder_step(
        model,
        discriminator,
        perceptual,
        x,
        ae_optimizer,
        step=1,
        discriminator_start=0,
        perceptual_weight=0.1,
        vq_weight=1.0,
        discriminator_weight=1.0,
    )
    d_metrics = vqgan_discriminator_step(
        discriminator,
        x,
        reconstruction,
        d_optimizer,
        step=1,
        discriminator_start=0,
    )
    prior = CausalTransformerPrior(
        32, 64, model_dim=32, heads=4, layers=2
    )
    logits, targets = prior.teacher_forcing(indices)
    prior_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    prior_loss.backward()
    assert reconstruction.shape == x.shape
    assert indices.shape == (2, 8, 8)
    assert model.decoder.last_layer.grad is not None
    assert discriminator.head[-1].weight.grad is not None
    print(
        f"smoke test passed: AE={metrics['autoencoder'].item():.3f}, "
        f"D={d_metrics['discriminator'].item():.3f}, "
        f"prior={prior_loss.item():.3f}, "
        f"adaptive={metrics['adaptive_weight'].item():.3f}"
    )


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    transform = normalized_cifar10_transform(horizontal_flip=False)
    root = PROJECT_ROOT / "data" / "cifar10"
    return (
        make_cifar10_loader(
            root,
            args.batch_size,
            device,
            train=True,
            transform=transform,
            num_workers=args.workers,
        ),
        make_cifar10_loader(
            root,
            args.batch_size,
            device,
            train=False,
            transform=transform,
            num_workers=args.workers,
        ),
    )


@torch.inference_mode()
def evaluate_tokenizer(
    model: VQPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    perceptual: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    discriminator.eval()
    perceptual.eval()
    vocabulary_size = model.quantizer.codebook_size
    totals = torch.zeros(5, device=device)
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(x)
        totals += torch.stack(
            [
                F.l1_loss(reconstruction, x),
                perceptual(reconstruction, x),
                diagnostics["quantization_mse"],
                discriminator(x).mean(),
                discriminator(reconstruction).mean(),
            ]
        ) * x.shape[0]
        usage.update(indices)
        examples += x.shape[0]
    values = (totals / examples).tolist()
    statistics = usage.statistics()
    entropy_bits = (
        float(statistics["token_entropy_nats"]) / math.log(2)
    )
    return {
        "pixel_l1": values[0],
        "feature_distance": values[1],
        "quantization_mse": values[2],
        "real_logit": values[3],
        "reconstruction_logit": values[4],
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
    tokenizer: VQPerceptualAutoencoder32,
    prior: CausalTransformerPrior,
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
        _, indices, _, _ = tokenizer.encode(x)
        logits, targets = prior.teacher_forcing(indices)
        loss = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        nll += float(loss) * x.shape[0]
        examples += x.shape[0]
    nll /= examples
    return {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": 64 * nll / math.log(2),
    }


def build_perceptual(name: str, device: torch.device) -> nn.Module:
    if name == "vgg":
        return VGGPerceptualLoss().to(device)
    if name == "random":
        print("warning: random features are a gradient-path debug mode, not a perceptual metric")
        return RandomFeaturePerceptualLoss().to(device)
    raise ValueError(f"unknown perceptual network: {name}")


def train_tokenizer(
    args: argparse.Namespace,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> VQPerceptualAutoencoder32:
    config = {
        "latent_channels": args.latent_channels,
        "codebook_size": args.codebook_size,
        "hidden_channels": args.hidden_channels,
        "commitment": args.commitment,
    }
    model = VQPerceptualAutoencoder32(**config).to(device)
    discriminator = PatchDiscriminator32(args.discriminator_channels).to(device)
    perceptual = build_perceptual(args.perceptual, device)
    ae_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.9)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9)
    )
    fixed_bits = 8 * 8 * math.ceil(math.log2(args.codebook_size))
    print(f"fixed-length upper bound={fixed_bits} bits/image")
    global_step = 0
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        discriminator.train()
        sums = torch.zeros(9, device=device)
        examples = 0
        usage = TokenUsageAccumulator(args.codebook_size)
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, metrics = vqgan_autoencoder_step(
                model,
                discriminator,
                perceptual,
                x,
                ae_optimizer,
                step=global_step,
                discriminator_start=args.discriminator_start,
                perceptual_weight=args.perceptual_weight,
                vq_weight=args.vq_weight,
                discriminator_weight=args.discriminator_weight,
            )
            d_metrics = vqgan_discriminator_step(
                discriminator,
                x,
                reconstruction,
                d_optimizer,
                step=global_step,
                discriminator_start=args.discriminator_start,
            )
            values = [
                metrics["autoencoder"],
                metrics["pixel_l1"],
                metrics["frozen_feature"],
                metrics["vq"],
                metrics["generator_adversarial"],
                metrics["adaptive_weight"],
                d_metrics["discriminator"],
                d_metrics["real_logit"],
                d_metrics["fake_logit"],
            ]
            sums += torch.stack(values) * x.shape[0]
            usage.update(indices)
            examples += x.shape[0]
            global_step += 1
        means = (sums / examples).tolist()
        epoch_usage = usage.statistics()
        print(
            f"tokenizer {epoch:03d}: AE={means[0]:.4f}, L1={means[1]:.4f}, "
            f"feature={means[2]:.4f}, VQ={means[3]:.4f}, G={means[4]:.4f}, "
            f"lambda={means[5]:.3f}, PPL/active="
            f"{epoch_usage['perplexity'].item():.1f}/"
            f"{epoch_usage['active_codes'].item()}, D={means[6]:.4f}, "
            f"logits(real,fake)=({means[7]:.3f},{means[8]:.3f})"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            save_image(
                torch.cat((x[:8], reconstruction[:8])).mul(0.5).add(0.5),
                out_dir / f"tokenizer_{epoch:03d}.png",
                nrow=8,
            )
    validation = evaluate_tokenizer(
        model,
        discriminator,
        perceptual,
        validation_loader,
        device=device,
    )
    interface_id = model_state_fingerprint(model)
    torch.save(
        {
            "format_version": 2,
            "model_name": "vqgan_tokenizer",
            "interface_id": interface_id,
            "state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "discriminator_config": {
                "base_channels": args.discriminator_channels,
            },
            "model_config": config,
            "token_grid": (8, 8),
            "index_order": "row-major",
            "perceptual_network": args.perceptual,
            "perceptual_metric_name": (
                "frozen torchvision VGG16 feature L1"
                if args.perceptual == "vgg"
                else "random frozen feature debug distance"
            ),
            "discriminator_start": args.discriminator_start,
            "global_step": global_step,
            "loss_weights": {
                "perceptual": args.perceptual_weight,
                "vq": args.vq_weight,
                "discriminator": args.discriminator_weight,
            },
            "dataset": "CIFAR-10",
            "image_size": 32,
            "value_range": [-1.0, 1.0],
            "validation_metrics": validation,
        },
        out_dir / "tokenizer.pth",
    )
    print(
        f"validation tokenizer: L1={validation['pixel_l1']:.4f}, "
        f"feature={validation['feature_distance']:.4f}, "
        f"entropy={validation['marginal_entropy_bits_per_token']:.3f} "
        f"bits/token, active={validation['active_codes']:.0f}/"
        f"{args.codebook_size}"
    )
    return model.eval().requires_grad_(False)


def load_tokenizer(path: Path, device: torch.device) -> VQPerceptualAutoencoder32:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "vqgan_tokenizer":
        raise ValueError("checkpoint is not a VQGAN tokenizer")
    model = VQPerceptualAutoencoder32(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("interface_id") != model_state_fingerprint(model):
        raise ValueError(
            "tokenizer checkpoint has no valid interface identity; retrain stage 1"
        )
    return model.eval().requires_grad_(False)


def train_prior(
    args: argparse.Namespace,
    tokenizer: VQPerceptualAutoencoder32,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    tokenizer.eval().requires_grad_(False)
    vocabulary_size = tokenizer.quantizer.codebook_size
    prior = CausalTransformerPrior(
        vocabulary_size,
        64,
        model_dim=args.prior_dim,
        heads=args.prior_heads,
        layers=args.prior_layers,
        dropout=args.prior_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.prior_lr)
    for epoch in range(1, args.prior_epochs + 1):
        prior.train()
        nll_sum = 0.0
        examples = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            with torch.inference_mode():
                _, indices, _, _ = tokenizer.encode(x)
            logits, targets = prior.teacher_forcing(indices)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
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
                64, device=device, temperature=args.temperature
            ).reshape(64, 8, 8)
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
            "model_name": "vqgan_transformer_prior",
            "state_dict": prior.state_dict(),
            "model_config": {
                "vocabulary_size": vocabulary_size,
                "sequence_length": 64,
                "model_dim": args.prior_dim,
                "heads": args.prior_heads,
                "layers": args.prior_layers,
                "dropout": args.prior_dropout,
            },
            "tokenizer_checkpoint": "tokenizer.pth",
            "tokenizer_interface_id": tokenizer_interface_id,
            "dataset": "CIFAR-10",
            "validation_metrics": validation,
        },
        out_dir / "transformer_prior.pth",
    )
    print(
        f"validation prior: {validation['bits_per_token']:.3f} "
        f"bits/token, {validation['bits_per_image']:.1f} bits/image"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vqgan"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders(args, device)
    path = out_dir / "tokenizer.pth"
    if args.stage in {"tokenizer", "all"}:
        tokenizer = train_tokenizer(
            args,
            train_loader,
            validation_loader,
            device,
            out_dir,
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
    parser.add_argument("--tokenizer-epochs", type=int, default=30)
    parser.add_argument("--prior-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--perceptual", choices=("vgg", "random"), default="vgg")
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--discriminator-weight", type=float, default=1.0)
    parser.add_argument("--discriminator-start", type=int, default=1000)
    parser.add_argument("--prior-dim", type=int, default=256)
    parser.add_argument("--prior-heads", type=int, default=8)
    parser.add_argument("--prior-layers", type=int, default=4)
    parser.add_argument("--prior-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--prior-lr", type=float, default=3e-4)
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
