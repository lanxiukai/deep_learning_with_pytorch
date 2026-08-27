"""Benchmark a short 128x128 training burst for one compact GAN lesson."""

from __future__ import annotations

import argparse
import math
import runpy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from dl_utils.gan.progan import ProGANDiscriminator, ProGANGenerator
from dl_utils.gan.stylegan import StyleGANDiscriminator, StyleGANGenerator
from dl_utils.gan.stylegan2 import StyleDiscriminator, StyleGenerator
from dl_utils.gan.stylegan_common import ProgressivePhase
from dl_utils.gan.training import GANRun, initialize_gan_models, prepare_gan_run
from dl_utils.training.accelerator import make_fused_adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LESSON_DIR = PROJECT_ROOT / "genai" / "1.0_generative_adversarial_network"
LESSON_FILES = {
    "progan": "7.0_progan.py",
    "stylegan": "7.1_stylegan.py",
    "stylegan2": "7.2_stylegan2.py",
}
RESOLUTION = 128


@dataclass
class BenchmarkState:
    """Objects needed to run repeated training bursts for one lesson."""

    model_name: str
    lesson: dict[str, Any]
    run: GANRun
    generator: nn.Module
    discriminator: nn.Module
    averaged_generator: nn.Module
    optimizer_g: Optimizer
    optimizer_d: Optimizer
    batch_size: int
    path_mean: torch.Tensor
    global_step: int


@dataclass(frozen=True)
class BackwardOnlyPrecision:
    """Run the real BF16 backward path without changing model parameters."""

    base: Any

    @property
    def name(self) -> str:
        return self.base.name

    def autocast(self):
        return self.base.autocast()

    def backward_step(self, loss, optimizer) -> None:
        del optimizer
        loss.backward()


def positive_int(value: str) -> int:
    """Parse one positive CLI integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse one non-negative CLI integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_state(args: argparse.Namespace) -> BenchmarkState:
    """Build the real lesson models, optimizers, BF16 runtime, and data stream."""
    lesson = runpy.run_path(str(LESSON_DIR / LESSON_FILES[args.model]))
    run = prepare_gan_run(
        f"benchmark-{args.model}",
        seed=int(lesson["SEED"]),
        data_pipeline=args.data_pipeline,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        project_root=PROJECT_ROOT,
    )
    if args.backward_only:
        run.precision = BackwardOnlyPrecision(run.precision)
    base_batch_size = (
        int(lesson["BATCH_SIZES"][RESOLUTION])
        if args.model in {"progan", "stylegan"}
        else int(lesson["BATCH_SIZE"])
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else base_batch_size * args.batch_scale
    )
    if run.dataset_size < batch_size:
        raise ValueError("CelebA train split is smaller than the benchmark batch.")

    if args.model == "progan":
        generator = ProGANGenerator(**lesson["MODEL_CONFIG"])
        discriminator = ProGANDiscriminator(**lesson["DISCRIMINATOR_CONFIG"])
    elif args.model == "stylegan":
        generator = StyleGANGenerator(**lesson["MODEL_CONFIG"])
        discriminator = StyleGANDiscriminator(**lesson["DISCRIMINATOR_CONFIG"])
    else:
        generator = StyleGenerator(**lesson["MODEL_CONFIG"])
        discriminator = StyleDiscriminator(**lesson["DISCRIMINATOR_CONFIG"])
    generator, discriminator, averaged_generator = initialize_gan_models(
        generator,
        discriminator,
        run.device,
    )

    if args.model == "stylegan2":
        g_ratio = lesson["G_REG_EVERY"] / (lesson["G_REG_EVERY"] + 1)
        d_ratio = lesson["D_REG_EVERY"] / (lesson["D_REG_EVERY"] + 1)
    else:
        g_ratio = d_ratio = 1.0
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=run.device,
        lr=lesson["LEARNING_RATE"] * g_ratio,
        betas=(0.0, 0.99**g_ratio),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=run.device,
        lr=lesson["LEARNING_RATE"] * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )
    return BenchmarkState(
        model_name=args.model,
        lesson=lesson,
        run=run,
        generator=generator,
        discriminator=discriminator,
        averaged_generator=averaged_generator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        batch_size=batch_size,
        path_mean=torch.zeros((), device=run.device),
        global_step=0,
    )


def train_batches(
    state: BenchmarkState,
    num_batches: int,
) -> dict[str, float]:
    """Run one representative regularization cycle at 128x128."""
    starting_step = state.global_step
    if state.model_name in {"progan", "stylegan"}:
        phase = ProgressivePhase(
            resolution=RESOLUTION,
            name="stabilization",
            batch_size=state.batch_size,
            num_batches=num_batches,
        )
        metrics, state.global_step = state.lesson["train_phase"](
            state.generator,
            state.discriminator,
            state.run.data,
            state.optimizer_g,
            state.optimizer_d,
            state.averaged_generator,
            phase,
            state.run.precision,
            state.global_step,
            state.lesson["D_REG_EVERY"],
            state.lesson["REG_BATCH_SHRINK"],
        )
    else:
        epoch = state.lesson["TrainingEpoch"](
            num_batches=num_batches,
            num_images=num_batches * state.batch_size,
            final_batch_size=state.batch_size,
        )
        metrics, state.path_mean, state.global_step = state.lesson["train_epoch"](
            state.generator,
            state.discriminator,
            state.run.data,
            state.optimizer_g,
            state.optimizer_d,
            state.averaged_generator,
            state.path_mean,
            state.global_step,
            epoch,
            state.batch_size,
            state.run.precision,
            state.lesson["R1_BATCH_SHRINK"],
            state.lesson["PATH_BATCH_SHRINK"],
        )
    if state.global_step != starting_step + num_batches:
        raise RuntimeError("benchmark completed an unexpected number of steps.")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"benchmark produced non-finite metrics: {metrics}")
    return metrics


def write_result(
    path: Path,
    state: BenchmarkState,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    """Write one machine-readable TSV row for the shell orchestrator."""
    timed_images = state.batch_size * args.batches
    images_per_second = timed_images / elapsed_seconds
    peak_allocated_gib = torch.cuda.max_memory_allocated(state.run.device) / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved(state.run.device) / 2**30
    header = (
        "model",
        "resolution",
        "batch_size",
        "pipeline",
        "precision",
        "warmup_batches",
        "timed_batches",
        "timed_images",
        "train_seconds",
        "images_per_second",
        "peak_allocated_gib",
        "peak_reserved_gib",
    )
    values = (
        state.model_name,
        str(RESOLUTION),
        str(state.batch_size),
        state.run.pipeline,
        state.run.precision.name,
        str(args.warmup_batches),
        str(args.batches),
        str(timed_images),
        f"{elapsed_seconds:.6f}",
        f"{images_per_second:.3f}",
        f"{peak_allocated_gib:.3f}",
        f"{peak_reserved_gib:.3f}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\t".join(header) + "\n" + "\t".join(values) + "\n",
        encoding="utf-8",
    )


def main(args: argparse.Namespace) -> None:
    """Warm the real training path, time it, and report throughput and memory."""
    state = build_state(args)
    print(
        f"model={state.model_name} resolution={RESOLUTION} "
        f"batch={state.batch_size} precision={state.run.precision.name} "
        f"pipeline={state.run.pipeline}"
    )
    if args.warmup_batches:
        print(f"warmup_batches={args.warmup_batches}")
        train_batches(state, args.warmup_batches)

    torch.cuda.synchronize(state.run.device)
    torch.cuda.reset_peak_memory_stats(state.run.device)
    start_time = time.perf_counter()
    metrics = train_batches(state, args.batches)
    torch.cuda.synchronize(state.run.device)
    elapsed_seconds = time.perf_counter() - start_time
    if elapsed_seconds <= 0:
        raise RuntimeError("benchmark timer returned a non-positive duration.")
    write_result(args.result_file, state, args, elapsed_seconds)

    metric_summary = " ".join(f"{name}={value:.6g}" for name, value in metrics.items())
    print(
        f"timed_batches={args.batches} "
        f"train_seconds={elapsed_seconds:.3f} "
        f"images_per_second={state.batch_size * args.batches / elapsed_seconds:.3f}"
    )
    print(f"metrics {metric_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short, full-training-load 128x128 BF16 benchmark for one GAN."
        )
    )
    parser.add_argument("model", choices=tuple(LESSON_FILES))
    parser.add_argument("--batches", type=positive_int, default=16)
    parser.add_argument("--warmup-batches", type=non_negative_int, default=1)
    parser.add_argument("--backward-only", action="store_true")
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument("--batch-scale", type=positive_int, default=1)
    batch_group.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    parser.add_argument("--prefetch-factor", type=positive_int, default=2)
    parser.add_argument(
        "--data-pipeline",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--result-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
