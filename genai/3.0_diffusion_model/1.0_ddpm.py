r"""Train a compact DDPM while keeping the paper equations visible.

This lesson follows one incremental path:

1. sample a timestep and noise in the closed-form forward process,
2. ask a time-conditioned U-Net to predict epsilon (the DDPM default),
3. minimize the simple denoising objective,
4. save the model together with the exact diffusion/model configuration.

The optional ``x0``, ``v``, and ``score`` targets use the same VP path.  For a
score target, multiplying squared error by ``sigma_t^2`` makes the objective
exactly equal to epsilon-MSE:

    sigma_t^2 ||s_theta(x_t,t) + epsilon / sigma_t||^2
        = ||-sigma_t s_theta(x_t,t) - epsilon||^2.

Linear beta scheduling and epsilon prediction reproduce the original DDPM
baseline.  Cosine scheduling is the isolated Improved-DDPM increment.  Optional
class dropout trains the null condition needed by classifier-free guidance;
the reverse-time algorithm itself remains in ``1.1_ddpm_sampling.py``.

The default 32x32 CIFAR-10 model is intentionally compact enough for a single
12 GB GPU.  Run ``--smoke-test`` to check equations, gradients, DDPM sampling,
and DDIM sampling without loading data or writing files.
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
from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion, PredictionType
from dl_utils.diffusion.diffusion_unet import DiffusionUNet


PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "diffusion" / "ddpm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--beta-schedule", choices=("linear", "cosine"), default="linear"
    )
    parser.add_argument(
        "--prediction-type",
        choices=("epsilon", "x0", "v", "score"),
        default="epsilon",
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=(64, 128, 256)
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Train a CIFAR-10 class-conditional model with a learned null class.",
    )
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=0.1,
        help="Probability of replacing a class with the null condition.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Save a 50-step DDIM preview every N epochs; zero disables it.",
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cifar10_loader(
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
            root=data_dir,
            train=True,
            transform=transform,
            download=False,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"CIFAR-10 was not found under {data_dir}. "
            "Prepare the dataset before starting a training run."
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


def diffuse_training_batch(
    model: DiffusionUNet,
    diffusion: GaussianDiffusion,
    x0: Tensor,
    class_labels: Tensor,
    prediction_type: PredictionType,
    condition_dropout: float,
) -> tuple[Tensor, Tensor]:
    """Construct one Monte Carlo estimate of the simple diffusion objective."""
    batch_size = x0.shape[0]
    timesteps = torch.randint(
        diffusion.num_steps, (batch_size,), device=x0.device
    )
    epsilon = torch.randn_like(x0)

    # q(x_t | x_0) = alpha_t x_0 + sigma_t epsilon.
    x_t = diffusion.q_sample(x0, timesteps, epsilon)
    target = diffusion.training_target(
        x0, epsilon, timesteps, prediction_type
    )

    labels: Tensor | None = None
    if model.num_classes is not None:
        labels = class_labels.clone()
        drop = torch.rand(batch_size, device=x0.device) < condition_dropout
        labels[drop] = model.null_class

    prediction = model(x_t, timesteps, labels)
    squared_error = (prediction - target).square()
    if prediction_type == "score":
        # sigma_t^2 weighting keeps score regression equal to epsilon-MSE.
        squared_error = (
            diffusion.noise_scale(timesteps, x0).square() * squared_error
        )
    per_sample = squared_error.flatten(start_dim=1).mean(dim=1)
    return per_sample.mean(), timesteps


def save_forward_process(
    diffusion: GaussianDiffusion,
    clean_image: Tensor,
    output_path: Path,
    num_frames: int = 8,
) -> None:
    """Visualize several marginals q(x_t|x0) using one fixed epsilon draw."""
    timesteps = torch.linspace(
        0,
        diffusion.num_steps - 1,
        num_frames,
        device=clean_image.device,
    ).round().long()
    clean = clean_image.expand(num_frames, -1, -1, -1)
    shared_noise = torch.randn_like(clean_image).expand_as(clean)
    noisy = diffusion.q_sample(clean, timesteps, shared_noise)
    save_image(noisy.mul(0.5).add(0.5), output_path, nrow=num_frames)


def checkpoint_payload(
    model: DiffusionUNet,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    prediction_type: PredictionType,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "algorithm": "vp_ddpm",
        "dataset": "CIFAR-10",
        "data_range": [-1.0, 1.0],
        "epoch": epoch,
        "prediction_type": prediction_type,
        "model_config": model.config(),
        "diffusion_config": diffusion.config(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def load_training_state(
    checkpoint_path: Path,
    model: DiffusionUNet,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    prediction_type: PredictionType,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected = {
        "algorithm": "vp_ddpm",
        "model_config": model.config(),
        "diffusion_config": diffusion.config(),
        "prediction_type": prediction_type,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint has a different {key}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]) + 1


@torch.no_grad()
def save_preview(
    model: DiffusionUNet,
    diffusion: GaussianDiffusion,
    prediction_type: PredictionType,
    output_path: Path,
    sample_count: int,
    seed: int,
) -> None:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(
        sample_count,
        model.in_channels,
        model.sample_size,
        model.sample_size,
        device=device,
        generator=generator,
    )
    labels = None
    if model.num_classes is not None:
        labels = torch.arange(sample_count, device=device) % model.num_classes
    samples, _ = diffusion.sample(
        model,
        initial_noise.shape,
        sampler="ddim",
        prediction_type=prediction_type,
        num_inference_steps=min(50, diffusion.num_steps),
        eta=0.0,
        labels=labels,
        initial_noise=initial_noise,
        generator=generator,
    )
    save_image(
        samples.mul(0.5).add(0.5),
        output_path,
        nrow=max(1, round(math.sqrt(sample_count))),
    )


def smoke_test() -> None:
    """Exercise target algebra, one gradient step, DDPM, DDIM, and CFG paths."""
    set_seed(7)
    device = torch.device("cpu")
    model = DiffusionUNet(
        image_size=8,
        hidden_dims=(16, 32),
        num_heads=4,
        num_classes=3,
    ).to(device)
    diffusion = GaussianDiffusion(num_steps=8).to(device)
    x0 = torch.randn(2, 3, 8, 8).clamp(-1.0, 1.0)
    epsilon = torch.randn_like(x0)
    timesteps = torch.tensor((1, 7), dtype=torch.long)
    x_t = diffusion.q_sample(x0, timesteps, epsilon)

    for prediction_type in ("epsilon", "x0", "v", "score"):
        target = diffusion.training_target(
            x0, epsilon, timesteps, prediction_type
        )
        converted = diffusion.convert_model_output(
            x_t,
            timesteps,
            target,
            prediction_type,
            clip_x0=None,
        )
        torch.testing.assert_close(converted.x0, x0, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            converted.epsilon, epsilon, atol=2e-5, rtol=2e-5
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.tensor((0, 2), dtype=torch.long)
    loss, _ = diffuse_training_batch(
        model, diffusion, x0, labels, "epsilon", condition_dropout=0.5
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise AssertionError("training loss is not finite")

    initial_noise = torch.randn_like(x0)
    generator = torch.Generator(device=device).manual_seed(11)
    ddpm, _ = diffusion.sample(
        model,
        x0.shape,
        sampler="ddpm",
        labels=labels,
        guidance_scale=1.5,
        initial_noise=initial_noise,
        generator=generator,
    )
    ddim, _ = diffusion.sample(
        model,
        x0.shape,
        sampler="ddim",
        num_inference_steps=4,
        eta=0.0,
        labels=labels,
        guidance_scale=1.5,
        initial_noise=initial_noise,
    )
    assert ddpm.shape == ddim.shape == x0.shape
    assert torch.isfinite(ddpm).all() and torch.isfinite(ddim).all()
    print(
        "smoke test passed: parameterizations, backward pass, "
        "DDPM, DDIM, and classifier-free guidance"
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 <= args.condition_dropout < 1.0:
        raise ValueError("condition_dropout must lie in [0, 1)")
    if args.sample_every < 0 or args.sample_count < 1:
        raise ValueError("sample_every must be non-negative and sample_count positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = cifar10_loader(
        args.data_dir, args.batch_size, args.num_workers, device
    )
    model = DiffusionUNet(
        image_size=32,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        num_classes=10 if args.conditional else None,
    ).to(device)
    diffusion = GaussianDiffusion(
        num_steps=args.num_steps,
        beta_schedule=args.beta_schedule,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch = 1
    if args.resume_from is not None:
        start_epoch = load_training_state(
            args.resume_from,
            model,
            diffusion,
            optimizer,
            args.prediction_type,
            device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_clean, _ = next(iter(loader))
    save_forward_process(
        diffusion,
        first_clean[:1].to(device),
        args.output_dir / "forward_process.png",
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count / 1e6:.2f}M "
        f"target={args.prediction_type} schedule={args.beta_schedule}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch:03d}/{args.epochs:03d}")
        for batch_index, (x0, class_labels) in enumerate(progress, start=1):
            x0 = x0.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            loss, _ = diffuse_training_batch(
                model,
                diffusion,
                x0,
                class_labels,
                args.prediction_type,
                args.condition_dropout,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            progress.set_postfix(loss=f"{loss_sum / batch_index:.4f}")

        torch.save(
            checkpoint_payload(
                model,
                diffusion,
                optimizer,
                epoch,
                args.prediction_type,
            ),
            args.output_dir / "latest.pth",
        )
        if args.sample_every and (
            epoch % args.sample_every == 0 or epoch == args.epochs
        ):
            save_preview(
                model,
                diffusion,
                args.prediction_type,
                args.output_dir / f"samples_epoch_{epoch:03d}.png",
                args.sample_count,
                args.seed,
            )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    train(args)


if __name__ == "__main__":
    main()
