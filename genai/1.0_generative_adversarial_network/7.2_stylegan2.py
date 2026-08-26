"""Train a compact 128x128 StyleGAN2 on aligned CelebA.

The defaults target an NVIDIA B200 or B300: a 2,500-kimg budget, BF16 AMP, and
twice the teaching batch size while retaining lazy R1 and path-length
regularization.  Higher-order penalties stay in FP32 on reduced regularization
batches.  Fused Adam, channels-last tensors, and batched CUDA JPEG decoding
avoid requiring custom CUDA kernels.

Fresh runs reset ``output/stylegan2/training`` and ``checkpoints``.  A latest
full-state checkpoint is written at every data-epoch boundary, matching the
reference lesson, together with a fixed-latent/noise EMA sample grid.

Run:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.2_stylegan2.py

Resume:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.2_stylegan2.py \
        --resume-from output/stylegan2/checkpoints/latest.pth

Default schedule:
    Real images shown to D:  2.5 million
    Main D/G updates:        39,063 / 39,063

Model size:
    Generator:               4.05 M parameters (plus one EMA copy)
    Discriminator:           4.76 M parameters
"""

import argparse
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.data.celeba import (
    CelebAAlignedDataset,
    make_celeba_training_loader,
    prepare_encoded_celeba_batch,
    resolve_celeba_pipeline,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.inference import generate_in_batches
from dl_utils.gan.stylegan2 import StyleDiscriminator, StyleGenerator
from dl_utils.gan.stylegan_common import (
    denormalize,
    path_length_penalty,
    r1_penalty,
    sample_mixing_latents,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.accelerator import (
    configure_device,
    make_fused_adam,
    resolve_bf16_precision,
)
from dl_utils.training.checkpoints import TrainingCheckpoint, save_model_weights
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.optimization import update_ema_by_images

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
OUTPUT_ROOT = Path(os.environ.get("DL_OUTPUT_ROOT", str(PROJECT_ROOT / "output")))
OUT_DIR = OUTPUT_ROOT / "stylegan2"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZE = 64
TOTAL_KIMG = 2_500
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
D_REG_EVERY = 16
PATH_WEIGHT = 2.0
G_REG_EVERY = 8
R1_BATCH_SHRINK = 2
PATH_BATCH_SHRINK = 4
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 16
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
}
DISCRIMINATOR_CONFIG = {"base_channels": BASE_CHANNELS}
METRIC_NAMES = (
    "loss_d_main",
    "loss_g_main",
    "r1_penalty",
    "path_penalty",
)


@dataclass(frozen=True)
class TrainingEpoch:
    """One data epoch in a fixed-resolution image budget."""

    num_batches: int
    num_images: int
    final_batch_size: int

    def batch_size_at(self, batch_index, batch_size):
        if not 0 <= batch_index < self.num_batches:
            raise ValueError("batch_index is outside this training epoch.")
        if batch_index == self.num_batches - 1:
            return self.final_batch_size
        return batch_size


def build_training_schedule(total_kimg, batch_size, dataset_size):
    """Build data epochs with an exact final image and batch count."""
    if min(total_kimg, batch_size, dataset_size) < 1:
        raise ValueError("total_kimg, batch_size, and dataset_size must be positive.")
    if dataset_size < batch_size:
        raise ValueError("dataset_size must contain at least one complete batch.")
    total_images = total_kimg * 1_000
    total_batches = math.ceil(total_images / batch_size)
    final_training_batch = total_images - batch_size * (total_batches - 1)
    batches_per_epoch = dataset_size // batch_size
    schedule = []
    for batch_start in range(0, total_batches, batches_per_epoch):
        num_batches = min(batches_per_epoch, total_batches - batch_start)
        is_last = batch_start + num_batches == total_batches
        final_batch_size = final_training_batch if is_last else batch_size
        num_images = num_batches * batch_size
        if is_last:
            num_images -= batch_size - final_training_batch
        schedule.append(
            TrainingEpoch(
                num_batches=num_batches,
                num_images=num_images,
                final_batch_size=final_batch_size,
            )
        )
    return tuple(schedule)


def _next_real_batch(
    batches,
    loader,
    epoch,
    batch_index,
    batch_size,
    device,
    pipeline,
):
    try:
        raw_images, _ = next(batches)
    except StopIteration:
        batches = iter(loader)
        raw_images, _ = next(batches)
    if pipeline == "cuda":
        real_images = prepare_encoded_celeba_batch(raw_images, 128, device)
    else:
        real_images = raw_images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
    real_images = real_images[: epoch.batch_size_at(batch_index, batch_size)]
    return real_images, batches


def train_epoch(
    generator,
    discriminator,
    loader,
    batches,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    path_mean,
    global_step,
    epoch,
    batch_size,
    device,
    pipeline,
    precision,
    r1_batch_shrink,
    path_batch_shrink,
):
    """Train one data epoch with separate lazy penalties."""
    metrics = MetricAccumulator(METRIC_NAMES, device=device)
    progress = tqdm(
        range(epoch.num_batches),
        desc="StyleGAN2 128x",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for batch_index in progress:
        real, batches = _next_real_batch(
            batches,
            loader,
            epoch,
            batch_index,
            batch_size,
            device,
            pipeline,
        )
        current_batch = len(real)

        optimizer_d.zero_grad(set_to_none=True)
        z, mixing_z = sample_mixing_latents(
            current_batch,
            generator.z_dim,
            device,
            STYLE_MIXING_PROBABILITY,
        )
        with torch.no_grad(), precision.autocast():
            fake = generator(z, mixing_z=mixing_z)
        with precision.autocast():
            real_scores = discriminator(real)
            fake_scores = discriminator(fake)
            loss_d_main = (
                F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()
            )
        precision.backward_step(loss_d_main, optimizer_d)

        weighted_r1 = real.new_zeros(())
        if global_step % D_REG_EVERY == 0:
            optimizer_d.zero_grad(set_to_none=True)
            reg_batch = max(1, current_batch // r1_batch_shrink)
            real_for_r1 = real[:reg_batch].detach().float().requires_grad_(True)
            penalty_r1 = r1_penalty(
                discriminator(real_for_r1),
                real_for_r1,
            )
            weighted_r1 = 0.5 * R1_GAMMA * D_REG_EVERY * penalty_r1
            precision.backward_step(weighted_r1, optimizer_d)

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z, mixing_z = sample_mixing_latents(
                current_batch,
                generator.z_dim,
                device,
                STYLE_MIXING_PROBABILITY,
            )
            with precision.autocast():
                fake = generator(
                    z,
                    mixing_z=mixing_z,
                    update_w_avg=True,
                )
                loss_g_main = F.softplus(-discriminator(fake)).mean()
            precision.backward_step(loss_g_main, optimizer_g)

            weighted_path = real.new_zeros(())
            if global_step % G_REG_EVERY == 0:
                optimizer_g.zero_grad(set_to_none=True)
                path_batch = max(1, current_batch // path_batch_shrink)
                path_z = torch.randn(
                    path_batch,
                    generator.z_dim,
                    device=device,
                )
                path_images, path_ws = generator(path_z, return_ws=True)
                penalty_path, path_mean = path_length_penalty(
                    path_images,
                    path_ws,
                    path_mean,
                )
                weighted_path = PATH_WEIGHT * G_REG_EVERY * penalty_path
                precision.backward_step(weighted_path, optimizer_g)
            update_ema_by_images(
                averaged_generator,
                generator,
                current_batch,
                EMA_HALF_LIFE_KIMG,
            )
        finally:
            discriminator.requires_grad_(True)

        metrics.update(
            (loss_d_main, loss_g_main, weighted_r1, weighted_path),
            num_examples=current_batch,
        )
        global_step += 1

    return metrics.compute(), path_mean, global_step, batches


def save_training_samples(
    generator,
    fixed_z,
    output_path,
    seen_kimg,
    precision,
):
    """Save fixed-latent and fixed-noise EMA samples."""

    def generate(z_batch):
        with precision.autocast():
            return generator(z_batch, noise_mode="fixed").float()

    samples = generate_in_batches(
        fixed_z,
        SAMPLE_BATCH_SIZE,
        generate,
        module=generator,
    )
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=f"StyleGAN2 EMA fixed samples - {seen_kimg:.0f} kimg",
    )


def _resolved_run_options(args):
    total_kimg = args.total_kimg if args.total_kimg is not None else TOTAL_KIMG
    batch_scale = args.batch_scale if args.batch_scale is not None else 1
    r1_batch_shrink = (
        args.r1_batch_shrink if args.r1_batch_shrink is not None else R1_BATCH_SHRINK
    )
    path_batch_shrink = (
        args.path_batch_shrink
        if args.path_batch_shrink is not None
        else PATH_BATCH_SHRINK
    )
    default_workers = min(8, os.cpu_count() or NUM_WORKERS)
    num_workers = default_workers if args.num_workers is None else args.num_workers
    if (
        min(
            total_kimg,
            batch_scale,
            r1_batch_shrink,
            path_batch_shrink,
            args.prefetch_factor,
        )
        < 1
    ):
        raise ValueError("training counts and scales must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    return {
        "total_kimg": total_kimg,
        "batch_size": BATCH_SIZE * batch_scale,
        "r1_batch_shrink": r1_batch_shrink,
        "path_batch_shrink": path_batch_shrink,
        "num_workers": num_workers,
        "prefetch_factor": args.prefetch_factor,
    }


def main(args):
    if not (DATA_DIR / "list_eval_partition.csv").is_file():
        raise FileNotFoundError(
            f"CelebA data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    options = _resolved_run_options(args)
    set_seed(SEED)
    device = try_gpu()
    configure_device(device)
    precision = resolve_bf16_precision(device)
    pipeline = resolve_celeba_pipeline(
        DATA_DIR,
        device,
        requested=args.data_pipeline,
    )
    dataset_size = len(CelebAAlignedDataset(DATA_DIR, split="train"))
    if dataset_size < options["batch_size"]:
        raise ValueError("CelebA train split is smaller than a training batch.")
    training_schedule = build_training_schedule(
        options["total_kimg"],
        options["batch_size"],
        dataset_size,
    )
    total_epochs = len(training_schedule)
    total_batches = sum(epoch.num_batches for epoch in training_schedule)
    total_images = sum(epoch.num_images for epoch in training_schedule)

    if args.resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    loader = make_celeba_training_loader(
        DATA_DIR,
        128,
        options["batch_size"],
        device,
        pipeline=pipeline,
        num_workers=options["num_workers"],
        prefetch_factor=options["prefetch_factor"],
    )
    generator = StyleGenerator(**MODEL_CONFIG).to(
        device,
        memory_format=torch.channels_last,
    )
    discriminator = StyleDiscriminator(**DISCRIMINATOR_CONFIG).to(
        device,
        memory_format=torch.channels_last,
    )
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)

    g_ratio = G_REG_EVERY / (G_REG_EVERY + 1)
    d_ratio = D_REG_EVERY / (D_REG_EVERY + 1)
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=device,
        lr=LEARNING_RATE * g_ratio,
        betas=(0.0, 0.99**g_ratio),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=device,
        lr=LEARNING_RATE * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )
    print(
        f"precision={precision.name} pipeline={pipeline} "
        f"workers={options['num_workers']} "
        f"total_kimg={options['total_kimg']} "
        f"batch={options['batch_size']}"
    )

    models = {
        "online_generator": generator,
        "stylegan2_generator": averaged_generator,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    checkpoint = TrainingCheckpoint(
        CHECKPOINT_DIR / "latest.pth",
        unit="fixed-resolution-epoch-main-metrics-v2",
        models=models,
        optimizers=optimizers,
    )
    run_config = {
        "precision": precision.name,
        "data_pipeline": pipeline,
        "dataset_size": dataset_size,
        "total_kimg": options["total_kimg"],
        "batch_size": options["batch_size"],
        "r1_batch_shrink": options["r1_batch_shrink"],
        "path_batch_shrink": options["path_batch_shrink"],
    }
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    completed_epochs, state = checkpoint.resume(
        args.resume_from,
        initial_state={
            "loss_history": {
                "kimg": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
            "path_mean": torch.zeros(()),
            "global_step": 0,
            "seen_images": 0,
            "run_config": run_config,
        },
    )
    if completed_epochs > total_epochs:
        raise ValueError("Checkpoint epoch exceeds the training schedule.")
    if state.get("run_config") != run_config:
        raise ValueError(
            "Checkpoint runtime options differ from this run; reuse the "
            "original batch, budget, regularization, and data-pipeline "
            "options."
        )
    if args.resume_from is not None:
        print(f"Resumed after epoch {completed_epochs}: {args.resume_from}")

    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)
    path_mean = state["path_mean"].to(device)
    global_step = int(state["global_step"])
    seen_images = int(state["seen_images"])
    completed_schedule = training_schedule[:completed_epochs]
    if global_step != sum(epoch.num_batches for epoch in completed_schedule):
        raise ValueError("Checkpoint global step does not match its epoch.")
    if seen_images != sum(epoch.num_images for epoch in completed_schedule):
        raise ValueError("Checkpoint image count does not match its epoch.")

    batches = iter(loader)
    for epoch_index, epoch in enumerate(
        training_schedule[completed_epochs:],
        start=completed_epochs + 1,
    ):
        print(f"Epoch {epoch_index}/{total_epochs}")
        metrics, path_mean, global_step, batches = train_epoch(
            generator,
            discriminator,
            loader,
            batches,
            optimizer_g,
            optimizer_d,
            averaged_generator,
            path_mean,
            global_step,
            epoch,
            options["batch_size"],
            device,
            pipeline,
            precision,
            options["r1_batch_shrink"],
            options["path_batch_shrink"],
        )
        seen_images += epoch.num_images
        seen_kimg = seen_images / 1_000
        loss_history["kimg"].append(seen_kimg)
        for name, value in metrics.items():
            loss_history[name].append(value)

        state["path_mean"] = path_mean.detach().cpu()
        state["global_step"] = global_step
        state["seen_images"] = seen_images
        checkpoint.save(epoch_index, state)
        save_training_samples(
            averaged_generator,
            fixed_z,
            TRAINING_DIR / f"kimg_{round(seen_kimg):05d}.png",
            seen_kimg,
            precision,
        )

    if global_step != total_batches or seen_images != total_images:
        raise RuntimeError("Training ended with an unexpected image budget.")
    save_model_weights(
        averaged_generator,
        OUT_DIR / "stylegan2_generator.pth",
        metadata={"model_name": "stylegan2", "model_config": MODEL_CONFIG},
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D logistic loss": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G non-saturating loss": loss_history["loss_g_main"],
            },
            "R1 regularization": {
                "Expected weighted contribution": loss_history["r1_penalty"],
            },
            "Path length regularization": {
                "Expected weighted contribution": loss_history["path_penalty"],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the full-resolution 128x128 CelebA StyleGAN2."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--total-kimg", type=int)
    parser.add_argument("--batch-scale", type=int)
    parser.add_argument("--r1-batch-shrink", type=int)
    parser.add_argument("--path-batch-shrink", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--data-pipeline",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
