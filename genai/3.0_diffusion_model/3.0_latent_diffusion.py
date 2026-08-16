r"""Move the DDPM state space from pixels into a frozen KL-AE latent grid.

Latent Diffusion is a two-stage model, not a new reverse equation:

1. train and validate ``4.2_kl_autoencoder.py``;
2. freeze its encoder and decoder;
3. encode ``z_0 = latent_scale * E(x)`` and train the same VP denoising loss;
4. sample ``z_0`` with DDPM or DDIM, undo the scale, and decode once.

The latent scale is loaded from the first-stage checkpoint and is part of the
interface.  It is neither fitted by this script nor silently absorbed into the
noise schedule.  The autoencoder stays in evaluation mode with gradients off,
making the two-stage optimization boundary executable and inspectable.

At 32x32 this is primarily an algorithm lesson: the 8x8 latent grid makes the
compute reduction obvious, while reconstruction loss from the first stage also
remains visible as an irreducible quality ceiling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.vae.perceptual_autoencoder import KLPerceptualAutoencoder32


PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
DEFAULT_AUTOENCODER = (
    PROJECT_ROOT / "output" / "kl_autoencoder" / "kl_autoencoder.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "diffusion" / "latent_diffusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "sample", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--autoencoder-checkpoint", type=Path, default=DEFAULT_AUTOENCODER
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "latest.pth"
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--beta-schedule", choices=("linear", "cosine"), default="linear"
    )
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(64, 128))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument(
        "--sampler", choices=("ddpm", "ddim"), default="ddim"
    )
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--class-label", type=int)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    transform = transforms.Compose(
        (
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )
    )
    try:
        dataset = datasets.CIFAR10(
            data_dir, train=True, transform=transform, download=False
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"CIFAR-10 was not found under {data_dir}. "
            "Prepare it before training latent diffusion."
        ) from error
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def load_autoencoder(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[KLPerceptualAutoencoder32, float, dict[str, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "kl_perceptual_autoencoder":
        raise ValueError("first-stage checkpoint is not a KL perceptual autoencoder")
    config = checkpoint["model_config"]
    latent_scale = float(checkpoint["latent_scale"])
    if not math.isfinite(latent_scale) or latent_scale <= 0.0:
        raise ValueError("first-stage latent_scale must be finite and positive")
    model = KLPerceptualAutoencoder32(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    return model, latent_scale, config


def latent_diffusion_batch(
    denoiser: DiffusionUNet,
    diffusion: GaussianDiffusion,
    z0: Tensor,
    class_labels: Tensor,
    condition_dropout: float,
) -> Tensor:
    """The original DDPM epsilon objective, now evaluated on frozen latents."""
    batch_size = z0.shape[0]
    timesteps = torch.randint(
        diffusion.num_steps, (batch_size,), device=z0.device
    )
    epsilon = torch.randn_like(z0)
    z_t = diffusion.q_sample(z0, timesteps, epsilon)

    labels: Tensor | None = None
    if denoiser.num_classes is not None:
        labels = class_labels.clone()
        drop = torch.rand(batch_size, device=z0.device) < condition_dropout
        labels[drop] = denoiser.null_class
    predicted_epsilon = denoiser(z_t, timesteps, labels)
    return (predicted_epsilon - epsilon).square().mean()


def checkpoint_payload(
    denoiser: DiffusionUNet,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    autoencoder_config: dict[str, int],
    latent_scale: float,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "algorithm": "latent_vp_ddpm",
        "dataset": "CIFAR-10",
        "prediction_type": "epsilon",
        "latent_shape": [
            autoencoder_config["latent_channels"],
            denoiser.sample_size,
            denoiser.sample_size,
        ],
        "latent_scale": latent_scale,
        "autoencoder_model_name": "kl_perceptual_autoencoder",
        "autoencoder_config": autoencoder_config,
        "model_config": denoiser.config(),
        "diffusion_config": diffusion.config(),
        "epoch": epoch,
        "model_state": denoiser.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def validate_first_stage(
    checkpoint: dict[str, object],
    autoencoder_config: dict[str, int],
    latent_scale: float,
) -> None:
    if checkpoint.get("autoencoder_config") != autoencoder_config:
        raise ValueError("latent checkpoint expects a different autoencoder shape")
    stored_scale = float(checkpoint["latent_scale"])
    if not math.isclose(stored_scale, latent_scale, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("latent checkpoint expects a different latent_scale")


def train_model(
    args: argparse.Namespace,
    autoencoder: KLPerceptualAutoencoder32,
    latent_scale: float,
    autoencoder_config: dict[str, int],
    device: torch.device,
) -> tuple[DiffusionUNet, GaussianDiffusion]:
    loader = make_loader(
        args.data_dir, args.batch_size, args.num_workers, device
    )
    denoiser = DiffusionUNet(
        image_size=8,
        in_channels=autoencoder.latent_channels,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        num_classes=10 if args.conditional else None,
    ).to(device)
    diffusion = GaussianDiffusion(
        num_steps=args.num_steps, beta_schedule=args.beta_schedule
    ).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.learning_rate)

    start_epoch = 1
    if args.resume_from is not None:
        checkpoint = torch.load(
            args.resume_from, map_location=device, weights_only=True
        )
        if checkpoint.get("algorithm") != "latent_vp_ddpm":
            raise ValueError("resume checkpoint is not latent VP diffusion")
        validate_first_stage(checkpoint, autoencoder_config, latent_scale)
        if checkpoint.get("model_config") != denoiser.config():
            raise ValueError("resume checkpoint has a different denoiser config")
        if checkpoint.get("diffusion_config") != diffusion.config():
            raise ValueError("resume checkpoint has a different diffusion config")
        denoiser.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in denoiser.parameters())
    print(
        f"device={device} denoiser={parameter_count / 1e6:.2f}M "
        f"latent=[{autoencoder.latent_channels}, 8, 8] scale={latent_scale:.6f}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        denoiser.train()
        loss_sum = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch:03d}/{args.epochs:03d}")
        for batch_index, (images, class_labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            with torch.no_grad():
                z0 = autoencoder.encode_latent(
                    images, sample=True, latent_scale=latent_scale
                )
            loss = latent_diffusion_batch(
                denoiser,
                diffusion,
                z0,
                class_labels,
                args.condition_dropout,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            progress.set_postfix(loss=f"{loss_sum / batch_index:.4f}")

        torch.save(
            checkpoint_payload(
                denoiser,
                diffusion,
                optimizer,
                epoch,
                autoencoder_config,
                latent_scale,
            ),
            args.output_dir / "latest.pth",
        )
    return denoiser.eval(), diffusion


def load_denoiser(
    checkpoint_path: Path,
    autoencoder_config: dict[str, int],
    latent_scale: float,
    device: torch.device,
) -> tuple[DiffusionUNet, GaussianDiffusion]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "latent_vp_ddpm":
        raise ValueError("checkpoint is not a latent VP diffusion checkpoint")
    validate_first_stage(checkpoint, autoencoder_config, latent_scale)
    denoiser = DiffusionUNet(**checkpoint["model_config"]).to(device)
    denoiser.load_state_dict(checkpoint["model_state"])
    diffusion = GaussianDiffusion(**checkpoint["diffusion_config"]).to(device)
    return denoiser.eval(), diffusion


def sample_labels(
    denoiser: DiffusionUNet,
    sample_count: int,
    class_label: int | None,
    device: torch.device,
) -> Tensor | None:
    if denoiser.num_classes is None:
        if class_label is not None:
            raise ValueError("class_label requires a conditional denoiser")
        return None
    if class_label is not None:
        if not 0 <= class_label < denoiser.num_classes:
            raise ValueError("class_label is outside the CIFAR-10 range")
        return torch.full(
            (sample_count,), class_label, device=device, dtype=torch.long
        )
    return torch.arange(sample_count, device=device) % denoiser.num_classes


@torch.no_grad()
def generate(
    args: argparse.Namespace,
    autoencoder: KLPerceptualAutoencoder32,
    latent_scale: float,
    denoiser: DiffusionUNet,
    diffusion: GaussianDiffusion,
    device: torch.device,
) -> Tensor:
    if denoiser.num_classes is None and args.guidance_scale != 1.0:
        raise ValueError("guidance_scale only changes a conditional denoiser")
    labels = sample_labels(
        denoiser, args.sample_count, args.class_label, device
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (
        args.sample_count,
        autoencoder.latent_channels,
        denoiser.sample_size,
        denoiser.sample_size,
    )
    latents, _ = diffusion.sample(
        denoiser,
        shape,
        sampler=args.sampler,
        prediction_type="epsilon",
        num_inference_steps=(args.ddim_steps if args.sampler == "ddim" else None),
        eta=args.eta,
        labels=labels,
        guidance_scale=args.guidance_scale,
        clip_x0=None,
        generator=generator,
    )
    images = autoencoder.decode_latent(latents, latent_scale=latent_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        images.mul(0.5).add(0.5),
        args.output_dir / f"{args.sampler}_samples.png",
        nrow=max(1, round(math.sqrt(args.sample_count))),
    )
    print(f"saved decoded samples to {args.output_dir}")
    return images


def smoke_test() -> None:
    set_seed(19)
    autoencoder = KLPerceptualAutoencoder32(
        latent_channels=4, hidden_channels=32
    ).eval().requires_grad_(False)
    denoiser = DiffusionUNet(
        image_size=8,
        in_channels=4,
        hidden_dims=(16, 32),
        num_heads=4,
        num_classes=3,
    )
    diffusion = GaussianDiffusion(num_steps=8)
    images = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    labels = torch.tensor((0, 2))
    with torch.no_grad():
        z0 = autoencoder.encode_latent(images, sample=True, latent_scale=0.5)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    loss = latent_diffusion_batch(
        denoiser, diffusion, z0, labels, condition_dropout=0.5
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in autoencoder.parameters())

    latents, _ = diffusion.sample(
        denoiser,
        z0.shape,
        sampler="ddim",
        num_inference_steps=4,
        labels=labels,
        guidance_scale=1.5,
        clip_x0=None,
    )
    decoded = autoencoder.decode_latent(latents, latent_scale=0.5)
    assert z0.shape == (2, 4, 8, 8)
    assert decoded.shape == images.shape
    assert torch.isfinite(loss) and torch.isfinite(decoded).all()
    print(
        "smoke test passed: frozen KL-AE boundary, latent DDPM loss, "
        "DDIM sampling, and decoding"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.sample_count < 1:
        raise ValueError("epochs, batch_size, and sample_count must be positive")
    if not 0.0 <= args.condition_dropout < 1.0:
        raise ValueError("condition_dropout must lie in [0, 1)")
    if args.eta < 0.0 or args.guidance_scale < 0.0:
        raise ValueError("eta and guidance_scale must be non-negative")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    validate_args(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder, latent_scale, autoencoder_config = load_autoencoder(
        args.autoencoder_checkpoint, device
    )
    if args.mode in ("train", "both"):
        denoiser, diffusion = train_model(
            args,
            autoencoder,
            latent_scale,
            autoencoder_config,
            device,
        )
    else:
        denoiser, diffusion = load_denoiser(
            args.checkpoint, autoencoder_config, latent_scale, device
        )
    if args.mode in ("sample", "both"):
        generate(
            args, autoencoder, latent_scale, denoiser, diffusion, device
        )


if __name__ == "__main__":
    main()
