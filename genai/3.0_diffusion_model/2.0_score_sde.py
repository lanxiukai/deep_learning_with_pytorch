r"""Train and sample a continuous-time VP score model.

This lesson is the continuous-time increment after discrete DDPM.  The forward
SDE is

    dx = -1/2 beta(t) x dt + sqrt(beta(t)) dw,

whose perturbation kernel remains analytic:

    x_t = alpha(t) x_0 + sigma(t) epsilon.

The network regresses the conditional score ``-epsilon / sigma(t)``.  Averaging
that conditional target over clean examples gives the marginal score needed by
the reverse process.  The implemented denoising-score-matching loss uses
``sigma(t)^2`` weighting, so it can be read as stable epsilon regression while
the network output is still a score.

Two reverse-time views reuse the same trained network:

* the reverse SDE uses Euler--Maruyama and injects Brownian noise;
* the probability-flow ODE removes that noise and uses Euler or Heun updates.

The ODE shares the SDE's time marginals in the ideal-score limit but not its
individual trajectories.  This script deliberately keeps the solver explicit
instead of depending on a general-purpose SDE/ODE library.
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
from dl_utils.diffusion.diffusion_score_sde import VPSDE, sample_vp_sde
from dl_utils.diffusion.diffusion_unet import DiffusionUNet


PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "diffusion" / "score_sde"
TIME_EMBEDDING_SCALE = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "sample", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "latest.pth",
        help="Checkpoint loaded in sample mode.",
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
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--time-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--sampler",
        choices=("reverse_sde", "probability_flow", "both"),
        default="both",
    )
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--ode-solver", choices=("euler", "heun"), default="heun")
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


def score_matching_batch(
    model: DiffusionUNet,
    sde: VPSDE,
    x0: Tensor,
    time_epsilon: float,
) -> tuple[Tensor, Tensor]:
    """One Monte Carlo estimate of weighted denoising score matching."""
    batch_size = x0.shape[0]
    time = time_epsilon + (1.0 - time_epsilon) * torch.rand(
        batch_size, device=x0.device
    )
    epsilon = torch.randn_like(x0)
    x_t, _, sigma = sde.marginal_sample(x0, time, epsilon)
    target_score = sde.score_target(epsilon, sigma)
    predicted_score = model(x_t, time * TIME_EMBEDDING_SCALE)

    # sigma^2 ||score_theta - (-epsilon/sigma)||^2
    # is exactly ||sigma score_theta + epsilon||^2.
    residual = sigma * (predicted_score - target_score)
    per_sample = residual.square().flatten(start_dim=1).mean(dim=1)
    return per_sample.mean(), time


def save_forward_process(
    sde: VPSDE,
    clean_image: Tensor,
    output_path: Path,
    time_epsilon: float,
    num_frames: int = 8,
) -> None:
    times = torch.linspace(
        time_epsilon, 1.0, num_frames, device=clean_image.device
    )
    clean = clean_image.expand(num_frames, -1, -1, -1)
    shared_noise = torch.randn_like(clean_image).expand_as(clean)
    noisy, _, _ = sde.marginal_sample(clean, times, shared_noise)
    save_image(noisy.mul(0.5).add(0.5), output_path, nrow=num_frames)


def checkpoint_payload(
    model: DiffusionUNet,
    sde: VPSDE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    time_epsilon: float,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "algorithm": "vp_score_sde",
        "dataset": "CIFAR-10",
        "data_range": [-1.0, 1.0],
        "epoch": epoch,
        "model_config": model.config(),
        "sde_config": {"beta_min": sde.beta_min, "beta_max": sde.beta_max},
        "time_epsilon": time_epsilon,
        "time_embedding_scale": TIME_EMBEDDING_SCALE,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DiffusionUNet, VPSDE, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "vp_score_sde":
        raise ValueError("checkpoint is not a vp_score_sde teaching checkpoint")
    if checkpoint.get("time_embedding_scale") != TIME_EMBEDDING_SCALE:
        raise ValueError("checkpoint uses a different time embedding scale")
    model = DiffusionUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    sde = VPSDE(**checkpoint["sde_config"])
    return model.eval(), sde, float(checkpoint["time_epsilon"])


def load_training_state(
    checkpoint_path: Path,
    model: DiffusionUNet,
    sde: VPSDE,
    optimizer: torch.optim.Optimizer,
    time_epsilon: float,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected = {
        "algorithm": "vp_score_sde",
        "model_config": model.config(),
        "sde_config": {"beta_min": sde.beta_min, "beta_max": sde.beta_max},
        "time_epsilon": time_epsilon,
        "time_embedding_scale": TIME_EMBEDDING_SCALE,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint has a different {key}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]) + 1


def save_sample_grid(samples: Tensor, output_path: Path) -> None:
    save_image(
        samples.mul(0.5).add(0.5),
        output_path,
        nrow=max(1, round(math.sqrt(samples.shape[0]))),
    )


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


def train_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DiffusionUNet, VPSDE, float]:
    loader = cifar10_loader(
        args.data_dir, args.batch_size, args.num_workers, device
    )
    model = DiffusionUNet(
        image_size=32,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch = 1
    if args.resume_from is not None:
        start_epoch = load_training_state(
            args.resume_from,
            model,
            sde,
            optimizer,
            args.time_epsilon,
            device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_clean, _ = next(iter(loader))
    save_forward_process(
        sde,
        first_clean[:1].to(device),
        args.output_dir / "forward_sde.png",
        args.time_epsilon,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count / 1e6:.2f}M "
        f"VP-SDE beta=[{sde.beta_min:g}, {sde.beta_max:g}]"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch:03d}/{args.epochs:03d}")
        for batch_index, (x0, _) in enumerate(progress, start=1):
            x0 = x0.to(device, non_blocking=True)
            loss, _ = score_matching_batch(
                model, sde, x0, args.time_epsilon
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            progress.set_postfix(loss=f"{loss_sum / batch_index:.4f}")

        torch.save(
            checkpoint_payload(
                model, sde, optimizer, epoch, args.time_epsilon
            ),
            args.output_dir / "latest.pth",
        )
    return model.eval(), sde, args.time_epsilon


@torch.no_grad()
def generate_samples(
    model: DiffusionUNet,
    sde: VPSDE,
    time_epsilon: float,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.sample_count,
        model.in_channels,
        model.sample_size,
        model.sample_size,
        device=device,
        generator=initial_generator,
    )
    requested = (
        ("reverse_sde", "probability_flow")
        if args.sampler == "both"
        else (args.sampler,)
    )
    outputs: dict[str, Tensor] = {}
    for index, sampler in enumerate(requested):
        print(
            f"{sampler}: {args.sampling_steps} updates"
            + (
                f", {args.ode_solver} solver"
                if sampler == "probability_flow"
                else ", Euler-Maruyama"
            )
        )
        generator = torch.Generator(device=device).manual_seed(
            args.seed + index + 1
        )
        samples, trajectory = sample_vp_sde(
            model,
            sde,
            initial_noise.shape,
            sampler=sampler,
            num_steps=args.sampling_steps,
            time_epsilon=time_epsilon,
            ode_solver=args.ode_solver,
            time_embedding_scale=TIME_EMBEDDING_SCALE,
            initial_noise=initial_noise,
            generator=generator,
            return_trajectory=True,
            trajectory_frames=args.trajectory_frames,
        )
        outputs[sampler] = samples.cpu()
        save_sample_grid(
            samples.cpu(), args.output_dir / f"{sampler}_samples.png"
        )
        save_trajectory(
            trajectory,
            args.output_dir / f"{sampler}_trajectory.png",
            args.trajectory_count,
        )

    if len(outputs) == 2:
        comparison = torch.cat(
            (outputs["reverse_sde"], outputs["probability_flow"]), dim=0
        )
        save_image(
            comparison.mul(0.5).add(0.5),
            args.output_dir / "reverse_sde_vs_probability_flow.png",
            nrow=args.sample_count,
        )
    print(f"saved samples to {args.output_dir}")


def smoke_test() -> None:
    set_seed(13)
    model = DiffusionUNet(
        image_size=8,
        hidden_dims=(16, 32),
        num_heads=4,
    )
    sde = VPSDE(beta_min=0.1, beta_max=5.0)
    x0 = torch.randn(2, 3, 8, 8).clamp(-1.0, 1.0)
    time = torch.tensor((0.2, 0.8))
    noise = torch.randn_like(x0)
    _, _, sigma = sde.marginal_sample(x0, time, noise)
    target = sde.score_target(noise, sigma)
    torch.testing.assert_close(sigma * target, -noise)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, _ = score_matching_batch(model, sde, x0, time_epsilon=1e-2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss):
        raise AssertionError("score-matching loss is not finite")

    initial_noise = torch.randn_like(x0)
    reverse, reverse_path = sample_vp_sde(
        model,
        sde,
        x0.shape,
        sampler="reverse_sde",
        num_steps=4,
        time_epsilon=1e-2,
        initial_noise=initial_noise,
        generator=torch.Generator().manual_seed(17),
        return_trajectory=True,
        trajectory_frames=4,
    )
    flow_a, flow_path = sample_vp_sde(
        model,
        sde,
        x0.shape,
        sampler="probability_flow",
        num_steps=4,
        time_epsilon=1e-2,
        ode_solver="heun",
        initial_noise=initial_noise,
        return_trajectory=True,
        trajectory_frames=4,
    )
    flow_b, _ = sample_vp_sde(
        model,
        sde,
        x0.shape,
        sampler="probability_flow",
        num_steps=4,
        time_epsilon=1e-2,
        ode_solver="heun",
        initial_noise=initial_noise,
    )
    torch.testing.assert_close(flow_a, flow_b)
    assert reverse.shape == flow_a.shape == x0.shape
    assert len(reverse_path) == len(flow_path) == 4
    assert torch.isfinite(reverse).all() and torch.isfinite(flow_a).all()
    print(
        "smoke test passed: weighted score matching, reverse SDE, "
        "and deterministic probability-flow ODE"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.sample_count < 1:
        raise ValueError("epochs, batch_size, and sample_count must be positive")
    if args.sampling_steps < 2 or args.trajectory_frames < 2:
        raise ValueError("sampling_steps and trajectory_frames must be at least 2")
    if args.trajectory_count < 1:
        raise ValueError("trajectory_count must be positive")
    if not 0.0 < args.time_epsilon < 1.0:
        raise ValueError("time_epsilon must lie in (0, 1)")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    validate_args(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("train", "both"):
        model, sde, time_epsilon = train_model(args, device)
    else:
        model, sde, time_epsilon = load_model(args.checkpoint, device)

    if args.mode in ("sample", "both"):
        generate_samples(model, sde, time_epsilon, args, device)


if __name__ == "__main__":
    main()
