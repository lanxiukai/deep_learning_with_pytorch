r"""Teach EDM as four separable design choices around one denoising network.

EDM is not just another scheduler.  This compact lesson makes its four coupled
training/sampling pieces explicit:

1. wrap a base U-Net with input, skip, output, and noise preconditioning;
2. draw a continuous positive noise scale from a log-normal distribution;
3. pair that scale with the prescribed weighted clean-image regression loss;
4. solve ``dx/dsigma = (x - D(x;sigma)) / sigma`` on a rho-shaped grid.

Sampling uses Euler prediction and optional Heun correction.  The zero-noise
endpoint is handled by Euler alone because the ODE quotient is singular there.
Stochastic churn and architecture-specific large-model details are omitted so
that the four core choices can be varied independently.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.diffusion.diffusion_unet import DiffusionUNet


PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "diffusion" / "edm"


class EDMPreconditioner(nn.Module):
    r"""Turn a base network into the EDM denoiser ``D_theta(x; sigma)``."""

    def __init__(
        self,
        network: DiffusionUNet,
        sigma_data: float = 0.5,
        noise_embedding_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if sigma_data <= 0.0 or noise_embedding_scale <= 0.0:
            raise ValueError("EDM scales must be positive")
        if network.num_classes is not None:
            raise ValueError("this EDM lesson uses an unconditional base network")
        self.network = network
        self.sigma_data = sigma_data
        self.noise_embedding_scale = noise_embedding_scale

    def config(self) -> dict[str, object]:
        return {
            "network_config": self.network.config(),
            "sigma_data": self.sigma_data,
            "noise_embedding_scale": self.noise_embedding_scale,
        }

    def forward(self, noisy: Tensor, sigma: Tensor) -> Tensor:
        if sigma.shape != (noisy.shape[0],):
            raise ValueError("sigma must have shape [batch]")
        sigma_image = sigma.reshape(noisy.shape[0], 1, 1, 1)
        sigma_data = self.sigma_data
        denominator = (sigma_image.square() + sigma_data**2).sqrt()
        c_skip = sigma_data**2 / denominator.square()
        c_out = sigma_image * sigma_data / denominator
        c_in = denominator.reciprocal()
        c_noise = 0.25 * sigma.log()

        # Scaling is part of this U-Net's embedding-coordinate convention; the
        # mathematical conditioning variable remains c_noise = log(sigma) / 4.
        residual = self.network(
            c_in * noisy,
            c_noise * self.noise_embedding_scale,
        )
        return c_skip * noisy + c_out * residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "sample", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "latest.pth"
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=(64, 128, 256)
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--log-sigma-mean", type=float, default=-1.2)
    parser.add_argument("--log-sigma-std", type=float, default=1.2)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sampling-steps", type=int, default=18)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--trajectory-frames", type=int, default=8)
    parser.add_argument("--trajectory-count", type=int, default=8)
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
            "Prepare it before starting EDM training."
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


def edm_training_batch(
    model: EDMPreconditioner,
    clean: Tensor,
    log_sigma_mean: float,
    log_sigma_std: float,
) -> tuple[Tensor, Tensor]:
    """One EDM clean-image regression estimate with matched weighting."""
    log_sigma = log_sigma_mean + log_sigma_std * torch.randn(
        clean.shape[0], device=clean.device
    )
    sigma = log_sigma.exp()
    sigma_image = sigma.reshape(clean.shape[0], 1, 1, 1)
    noisy = clean + sigma_image * torch.randn_like(clean)
    denoised = model(noisy, sigma)

    weight = (
        sigma.square() + model.sigma_data**2
    ) / (sigma * model.sigma_data).square()
    per_sample = (denoised - clean).square().flatten(start_dim=1).mean(dim=1)
    return (weight * per_sample).mean(), sigma


@torch.no_grad()
def update_ema(
    averaged: nn.Module,
    current: nn.Module,
    decay: float,
) -> None:
    """Apply the one-line parameter EMA used for sampling checkpoints."""
    for averaged_parameter, parameter in zip(
        averaged.parameters(), current.parameters(), strict=True
    ):
        averaged_parameter.lerp_(parameter, 1.0 - decay)
    for averaged_buffer, buffer in zip(
        averaged.buffers(), current.buffers(), strict=True
    ):
        averaged_buffer.copy_(buffer)


def edm_noise_grid(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> Tensor:
    """Construct the rho-shaped positive grid and append sigma=0."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 0.0 < sigma_min < sigma_max or rho <= 0.0:
        raise ValueError("expected 0 < sigma_min < sigma_max and rho > 0")
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device)
    inverse_rho = 1.0 / rho
    positive = (
        sigma_max**inverse_rho
        + ramp * (sigma_min**inverse_rho - sigma_max**inverse_rho)
    ).pow(rho)
    return torch.cat((positive, positive.new_zeros(1)))


@torch.no_grad()
def sample_edm(
    model: EDMPreconditioner,
    shape: tuple[int, int, int, int],
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    solver: Literal["euler", "heun"] = "heun",
    initial_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    trajectory_frames: int = 8,
) -> tuple[Tensor, list[Tensor], int]:
    """Integrate the EDM probability-flow ODE on a decreasing sigma grid."""
    if solver not in ("euler", "heun"):
        raise ValueError(f"unknown solver: {solver}")
    device = next(model.parameters()).device
    sigmas = edm_noise_grid(num_steps, sigma_min, sigma_max, rho, device)
    if initial_noise is None:
        initial_noise = torch.randn(
            shape, device=device, generator=generator
        )
    elif tuple(initial_noise.shape) != shape:
        raise ValueError("initial_noise does not match requested shape")
    state = initial_noise.to(device).clone() * sigmas[0]

    trajectory: list[Tensor] = []
    capture_positions: set[int] = set()
    if return_trajectory:
        if trajectory_frames < 2:
            raise ValueError("trajectory_frames must be at least 2")
        trajectory.append(state.detach().cpu())
        capture_count = min(trajectory_frames - 1, num_steps)
        capture_positions = set(
            torch.linspace(0, num_steps - 1, capture_count)
            .round()
            .long()
            .tolist()
        )

    was_training = model.training
    model.eval()
    function_evaluations = 0
    try:
        for index in range(num_steps):
            sigma = sigmas[index]
            sigma_next = sigmas[index + 1]
            sigma_batch = sigma.expand(shape[0])
            denoised = model(state, sigma_batch)
            function_evaluations += 1
            derivative = (state - denoised) / sigma
            step = sigma_next - sigma
            euler = state + step * derivative

            if solver == "heun" and sigma_next > 0:
                next_batch = sigma_next.expand(shape[0])
                denoised_next = model(euler, next_batch)
                function_evaluations += 1
                next_derivative = (euler - denoised_next) / sigma_next
                state = state + 0.5 * step * (derivative + next_derivative)
            else:
                # Never evaluate (x-D)/sigma at the singular sigma=0 endpoint.
                state = euler
            if return_trajectory and index in capture_positions:
                trajectory.append(state.detach().cpu())
    finally:
        model.train(was_training)
    return state.clamp(-1.0, 1.0), trajectory, function_evaluations


def checkpoint_payload(
    model: EDMPreconditioner,
    averaged: EDMPreconditioner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "algorithm": "edm",
        "dataset": "CIFAR-10",
        "data_range": [-1.0, 1.0],
        "epoch": epoch,
        "model_config": model.config(),
        "training_noise": {
            "distribution": "log_normal",
            "log_sigma_mean": args.log_sigma_mean,
            "log_sigma_std": args.log_sigma_std,
        },
        "ema_decay": args.ema_decay,
        "model_state": model.state_dict(),
        "ema_state": averaged.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def build_model(
    hidden_dims: tuple[int, ...],
    dropout: float,
    sigma_data: float,
) -> EDMPreconditioner:
    network = DiffusionUNet(
        image_size=32,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return EDMPreconditioner(network, sigma_data=sigma_data)


def model_from_config(config: dict[str, object]) -> EDMPreconditioner:
    network = DiffusionUNet(**config["network_config"])
    return EDMPreconditioner(
        network,
        sigma_data=float(config["sigma_data"]),
        noise_embedding_scale=float(config["noise_embedding_scale"]),
    )


def train_model(
    args: argparse.Namespace,
    device: torch.device,
) -> EDMPreconditioner:
    loader = make_loader(
        args.data_dir, args.batch_size, args.num_workers, device
    )
    model = build_model(
        tuple(args.hidden_dims), args.dropout, args.sigma_data
    ).to(device)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    if args.resume_from is not None:
        checkpoint = torch.load(
            args.resume_from, map_location=device, weights_only=True
        )
        expected_noise = {
            "distribution": "log_normal",
            "log_sigma_mean": args.log_sigma_mean,
            "log_sigma_std": args.log_sigma_std,
        }
        if checkpoint.get("algorithm") != "edm":
            raise ValueError("resume checkpoint is not EDM")
        if checkpoint.get("model_config") != model.config():
            raise ValueError("resume checkpoint has a different model config")
        if checkpoint.get("training_noise") != expected_noise:
            raise ValueError("resume checkpoint has a different noise distribution")
        if checkpoint.get("ema_decay") != args.ema_decay:
            raise ValueError("resume checkpoint has a different EMA decay")
        model.load_state_dict(checkpoint["model_state"])
        averaged.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count / 1e6:.2f}M "
        f"sigma_data={model.sigma_data:g}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch:03d}/{args.epochs:03d}")
        for batch_index, (clean, _) in enumerate(progress, start=1):
            clean = clean.to(device, non_blocking=True)
            loss, _ = edm_training_batch(
                model,
                clean,
                args.log_sigma_mean,
                args.log_sigma_std,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)

            loss_sum += loss.item()
            progress.set_postfix(loss=f"{loss_sum / batch_index:.4f}")
        torch.save(
            checkpoint_payload(model, averaged, optimizer, epoch, args),
            args.output_dir / "latest.pth",
        )
    return averaged


def load_averaged_model(
    checkpoint_path: Path,
    device: torch.device,
) -> EDMPreconditioner:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "edm":
        raise ValueError("checkpoint is not an EDM teaching checkpoint")
    model = model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["ema_state"])
    return model.eval()


def save_trajectory(
    trajectory: list[Tensor],
    output_path: Path,
    image_count: int,
) -> None:
    if not trajectory:
        return
    count = min(image_count, trajectory[0].shape[0])
    frames = torch.cat(
        [state[:count].clamp(-1.0, 1.0) for state in trajectory], dim=0
    )
    save_image(frames.mul(0.5).add(0.5), output_path, nrow=count)


@torch.no_grad()
def generate(
    model: EDMPreconditioner,
    args: argparse.Namespace,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (
        args.sample_count,
        model.network.in_channels,
        model.network.sample_size,
        model.network.sample_size,
    )
    samples, trajectory, nfe = sample_edm(
        model,
        shape,
        num_steps=args.sampling_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        solver=args.solver,
        generator=generator,
        return_trajectory=True,
        trajectory_frames=args.trajectory_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        samples.mul(0.5).add(0.5),
        args.output_dir / f"{args.solver}_samples.png",
        nrow=max(1, round(math.sqrt(args.sample_count))),
    )
    save_trajectory(
        trajectory,
        args.output_dir / f"{args.solver}_trajectory.png",
        args.trajectory_count,
    )
    print(f"saved samples to {args.output_dir}; NFE={nfe}")
    return samples


def smoke_test() -> None:
    set_seed(23)
    model = EDMPreconditioner(
        DiffusionUNet(
            image_size=8,
            hidden_dims=(16, 32),
            num_heads=4,
        ),
        sigma_data=0.5,
    )
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    clean = torch.randn(2, 3, 8, 8).clamp(-1.0, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, sigmas = edm_training_batch(model, clean, -1.2, 1.2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    update_ema(averaged, model, decay=0.9)
    assert torch.isfinite(loss) and (sigmas > 0).all()

    initial_noise = torch.randn_like(clean)
    first, path, nfe = sample_edm(
        averaged,
        clean.shape,
        num_steps=4,
        sigma_min=0.01,
        sigma_max=2.0,
        rho=3.0,
        solver="heun",
        initial_noise=initial_noise,
        return_trajectory=True,
        trajectory_frames=4,
    )
    second, _, _ = sample_edm(
        averaged,
        clean.shape,
        num_steps=4,
        sigma_min=0.01,
        sigma_max=2.0,
        rho=3.0,
        solver="heun",
        initial_noise=initial_noise,
    )
    torch.testing.assert_close(first, second)
    assert first.shape == clean.shape and len(path) == 4
    assert nfe == 7 and torch.isfinite(first).all()
    print(
        "smoke test passed: EDM preconditioning, weighted loss, EMA, "
        "rho schedule, and deterministic Heun sampling"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.sample_count < 1:
        raise ValueError("epochs, batch_size, and sample_count must be positive")
    if args.log_sigma_std <= 0.0 or args.sigma_data <= 0.0:
        raise ValueError("noise standard deviation and sigma_data must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must lie in [0, 1)")
    if args.trajectory_frames < 2 or args.trajectory_count < 1:
        raise ValueError("trajectory dimensions are invalid")
    edm_noise_grid(
        args.sampling_steps,
        args.sigma_min,
        args.sigma_max,
        args.rho,
        torch.device("cpu"),
    )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    validate_args(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode in ("train", "both"):
        model = train_model(args, device)
    else:
        model = load_averaged_model(args.checkpoint, device)
    if args.mode in ("sample", "both"):
        generate(model, args, device)


if __name__ == "__main__":
    main()
