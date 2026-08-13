r"""Compare DDPM ancestral sampling with generalized DDIM sampling.

Both samplers consume the same trained denoiser and the same initial ``x_T``.
Their difference is entirely in the reverse transition:

DDPM draws from every adjacent learned Gaussian transition,

    x_{t-1} = mu_theta(x_t, t) + sqrt(beta_tilde) z,

whereas DDIM may skip training steps and separates the update into a predicted
clean image, an epsilon direction, and optional stochasticity ``eta``:

    x_s = sqrt(alpha_bar_s) x0_hat
          + sqrt(1 - alpha_bar_s - sigma_eta^2) epsilon_hat
          + sigma_eta z.

``eta=0`` is deterministic for a fixed initial noise tensor.  DDPM deliberately
uses all training steps because merely subsampling its adjacent posterior is not
the DDIM algorithm.  If the checkpoint is class conditional, this script also
applies classifier-free guidance from the conditional and null predictions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.diffusion_ddpm import GaussianDiffusion, PredictionType
from dl_utils.genai.diffusion_unet import DiffusionUNet


PROJECT_ROOT = infer_project_root()
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "output" / "diffusion" / "ddpm" / "latest.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "diffusion" / "ddpm" / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sampler", choices=("ddpm", "ddim", "both"), default="both"
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity: zero is deterministic; larger values add noise.",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        help="Repeat one class; by default a conditional model cycles all classes.",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--trajectory-frames", type=int, default=8)
    parser.add_argument("--trajectory-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DiffusionUNet, GaussianDiffusion, PredictionType]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "vp_ddpm":
        raise ValueError("checkpoint is not a vp_ddpm teaching checkpoint")
    model = DiffusionUNet(**checkpoint["model_config"]).to(device)
    diffusion = GaussianDiffusion(**checkpoint["diffusion_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    prediction_type = checkpoint["prediction_type"]
    if prediction_type not in ("epsilon", "x0", "v", "score"):
        raise ValueError("checkpoint contains an unknown prediction type")
    return model.eval(), diffusion, prediction_type


def sampling_labels(
    model: DiffusionUNet,
    num_samples: int,
    class_label: int | None,
    device: torch.device,
) -> Tensor | None:
    if model.num_classes is None:
        if class_label is not None:
            raise ValueError("class_label requires a class-conditional checkpoint")
        return None
    if class_label is not None:
        if not 0 <= class_label < model.num_classes:
            raise ValueError("class_label is outside the checkpoint class range")
        return torch.full(
            (num_samples,), class_label, device=device, dtype=torch.long
        )
    return torch.arange(num_samples, device=device) % model.num_classes


def save_sample_grid(samples: Tensor, output_path: Path) -> None:
    sample_count = samples.shape[0]
    save_image(
        samples.mul(0.5).add(0.5),
        output_path,
        nrow=max(1, round(math.sqrt(sample_count))),
    )


def save_trajectory(
    trajectory: list[Tensor],
    output_path: Path,
    image_count: int,
) -> None:
    if not trajectory:
        return
    count = min(image_count, trajectory[0].shape[0])
    # Each row is one reverse-time state; columns track the same initial noises.
    grid = torch.cat(
        [frame[:count].clamp(-1.0, 1.0) for frame in trajectory], dim=0
    )
    save_image(grid.mul(0.5).add(0.5), output_path, nrow=count)


@torch.no_grad()
def generate(
    model: DiffusionUNet,
    diffusion: GaussianDiffusion,
    prediction_type: PredictionType,
    initial_noise: Tensor,
    labels: Tensor | None,
    sampler: str,
    ddim_steps: int,
    eta: float,
    guidance_scale: float,
    seed: int,
    trajectory_frames: int,
) -> tuple[Tensor, list[Tensor]]:
    device = initial_noise.device
    generator = torch.Generator(device=device).manual_seed(seed)
    if sampler == "ddpm":
        print(f"DDPM: {diffusion.num_steps} adjacent ancestral updates")
        return diffusion.sample(
            model,
            initial_noise.shape,
            sampler="ddpm",
            prediction_type=prediction_type,
            labels=labels,
            guidance_scale=guidance_scale,
            initial_noise=initial_noise,
            generator=generator,
            return_trajectory=True,
            trajectory_frames=trajectory_frames,
        )
    print(f"DDIM: {ddim_steps} updates, eta={eta:g}")
    return diffusion.sample(
        model,
        initial_noise.shape,
        sampler="ddim",
        prediction_type=prediction_type,
        num_inference_steps=ddim_steps,
        eta=eta,
        labels=labels,
        guidance_scale=guidance_scale,
        initial_noise=initial_noise,
        generator=generator,
        return_trajectory=True,
        trajectory_frames=trajectory_frames,
    )


def smoke_test() -> None:
    torch.manual_seed(5)
    model = DiffusionUNet(
        image_size=8,
        hidden_dims=(16, 32),
        num_heads=4,
    )
    diffusion = GaussianDiffusion(num_steps=8)
    initial_noise = torch.randn(2, 3, 8, 8)
    ddpm, ddpm_path = generate(
        model,
        diffusion,
        "epsilon",
        initial_noise,
        None,
        "ddpm",
        ddim_steps=4,
        eta=0.0,
        guidance_scale=1.0,
        seed=9,
        trajectory_frames=4,
    )
    ddim_a, ddim_path = generate(
        model,
        diffusion,
        "epsilon",
        initial_noise,
        None,
        "ddim",
        ddim_steps=4,
        eta=0.0,
        guidance_scale=1.0,
        seed=9,
        trajectory_frames=4,
    )
    ddim_b, _ = generate(
        model,
        diffusion,
        "epsilon",
        initial_noise,
        None,
        "ddim",
        ddim_steps=4,
        eta=0.0,
        guidance_scale=1.0,
        seed=99,
        trajectory_frames=4,
    )
    torch.testing.assert_close(ddim_a, ddim_b)
    assert ddpm.shape == ddim_a.shape == initial_noise.shape
    assert len(ddpm_path) == len(ddim_path) == 4
    print("smoke test passed: DDPM path and deterministic DDIM path")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.num_samples < 1 or args.trajectory_count < 1:
        raise ValueError("sample and trajectory counts must be positive")
    if args.trajectory_frames < 2:
        raise ValueError("trajectory_frames must be at least 2")
    if args.eta < 0.0 or args.guidance_scale < 0.0:
        raise ValueError("eta and guidance_scale must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, prediction_type = load_checkpoint(args.checkpoint, device)
    if not 2 <= args.ddim_steps <= diffusion.num_steps:
        raise ValueError("ddim_steps must lie between 2 and the training step count")
    if model.num_classes is None and args.guidance_scale != 1.0:
        raise ValueError("guidance_scale only changes a conditional checkpoint")

    labels = sampling_labels(
        model, args.num_samples, args.class_label, device
    )
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        model.in_channels,
        model.sample_size,
        model.sample_size,
        device=device,
        generator=initial_generator,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested = ("ddpm", "ddim") if args.sampler == "both" else (args.sampler,)
    outputs: dict[str, Tensor] = {}
    for index, sampler in enumerate(requested):
        samples, trajectory = generate(
            model,
            diffusion,
            prediction_type,
            initial_noise,
            labels,
            sampler,
            args.ddim_steps,
            args.eta,
            args.guidance_scale,
            args.seed + index + 1,
            args.trajectory_frames,
        )
        outputs[sampler] = samples.cpu()
        save_sample_grid(samples.cpu(), args.output_dir / f"{sampler}_samples.png")
        save_trajectory(
            trajectory,
            args.output_dir / f"{sampler}_trajectory.png",
            args.trajectory_count,
        )

    if len(outputs) == 2:
        comparison = torch.cat((outputs["ddpm"], outputs["ddim"]), dim=0)
        save_image(
            comparison.mul(0.5).add(0.5),
            args.output_dir / "ddpm_vs_ddim.png",
            nrow=args.num_samples,
        )
    print(f"saved samples to {args.output_dir}")


if __name__ == "__main__":
    main()
