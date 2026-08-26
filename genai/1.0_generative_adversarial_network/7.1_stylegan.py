"""Train a compact progressive StyleGAN on 128x128 aligned CelebA.

The defaults target an NVIDIA B200 or B300: BF16 AMP, twice the teaching batch
size, a 2.2-million-image schedule, lazy full-FP32 R1, channels-last tensors,
fused Adam, and automatic CUDA JPEG preprocessing.  They aim for recognizable
reference samples rather than paper-level fidelity.

Fresh runs reset ``output/stylegan/training`` and ``checkpoints``.  The latest
full-state checkpoint is replaced atomically at each progressive phase
boundary, matching the reference lesson, together with a fixed-latent/noise
EMA sample grid.

Run:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.1_stylegan.py

Resume:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.1_stylegan.py \
        --resume-from output/stylegan/checkpoints/latest.pth

Default schedule:
    Progressive phases:      11 x 200 kimg
    Real images shown to D:  2.2 million
    Main D/G updates:        24,222 / 24,222

Model size:
    Generator:               4.18 M parameters (plus one EMA copy)
    Discriminator:           4.59 M parameters
"""

import argparse
import os
from copy import deepcopy
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
from dl_utils.gan.stylegan import StyleGANDiscriminator, StyleGANGenerator
from dl_utils.gan.stylegan_common import (
    RESOLUTIONS,
    build_progressive_schedule,
    denormalize,
    phase_alpha,
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
OUT_DIR = OUTPUT_ROOT / "stylegan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZES = {4: 256, 8: 256, 16: 128, 32: 64, 64: 64, 128: 64}
PHASE_KIMG = 200
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 16
SEED = 42
D_REG_EVERY = 16
REG_BATCH_SHRINK = 2

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
}
DISCRIMINATOR_CONFIG = {"base_channels": BASE_CHANNELS}
METRIC_NAMES = ("loss_d_main", "loss_g_main", "r1_penalty")


def build_training_schedule(
    dataset_size,
    *,
    batch_sizes=None,
    phase_kimg=PHASE_KIMG,
):
    """Build one fade-in or stabilization entry per progressive phase."""
    batch_sizes = dict(BATCH_SIZES if batch_sizes is None else batch_sizes)
    if dataset_size < max(batch_sizes.values()):
        raise ValueError("CelebA train split is smaller than a training batch.")
    return build_progressive_schedule(
        resolutions=RESOLUTIONS,
        batch_sizes=batch_sizes,
        phase_kimg=phase_kimg,
    )


def _next_real_batch(batches, loader, phase, device, pipeline):
    try:
        raw_images, _ = next(batches)
    except StopIteration:
        batches = iter(loader)
        raw_images, _ = next(batches)

    if pipeline == "cuda":
        real_images = prepare_encoded_celeba_batch(
            raw_images,
            phase.resolution,
            device,
        )
    else:
        real_images = raw_images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
    return real_images, batches


def train_phase(
    generator,
    discriminator,
    loader,
    batches,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    phase,
    device,
    pipeline,
    precision,
    global_step,
    d_reg_every,
    reg_batch_shrink,
):
    """Train one progressive phase with BF16 main passes and FP32 R1."""
    metrics = MetricAccumulator(METRIC_NAMES, device=device)
    progress = tqdm(
        range(phase.num_batches),
        desc=f"StyleGAN {phase.resolution}x {phase.name}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for batch_index in progress:
        real_images, batches = _next_real_batch(
            batches,
            loader,
            phase,
            device,
            pipeline,
        )
        alpha = phase_alpha(phase, batch_index)
        batch_size = len(real_images)

        optimizer_d.zero_grad(set_to_none=True)
        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
            STYLE_MIXING_PROBABILITY,
        )
        with torch.no_grad(), precision.autocast():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
            )
        with precision.autocast():
            real_scores = discriminator(
                real_images,
                resolution=phase.resolution,
                alpha=alpha,
            )
            fake_scores = discriminator(
                fake_images,
                resolution=phase.resolution,
                alpha=alpha,
            )
            loss_d_main = (
                F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()
            )
        weighted_r1 = real_images.new_zeros(())
        if global_step % d_reg_every == 0:
            reg_batch = max(1, batch_size // reg_batch_shrink)
            real_for_r1 = real_images[:reg_batch].detach().float().requires_grad_(True)
            penalty_r1 = r1_penalty(
                discriminator(
                    real_for_r1,
                    resolution=phase.resolution,
                    alpha=alpha,
                ),
                real_for_r1,
            )
            weighted_r1 = 0.5 * R1_GAMMA * d_reg_every * penalty_r1
        precision.backward_step(
            loss_d_main.float() + weighted_r1,
            optimizer_d,
        )

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                device,
                STYLE_MIXING_PROBABILITY,
            )
            with precision.autocast():
                fake_images = generator(
                    z,
                    resolution=phase.resolution,
                    alpha=alpha,
                    mixing_z=mixing_z,
                    update_w_avg=True,
                )
                loss_g_main = F.softplus(
                    -discriminator(
                        fake_images,
                        resolution=phase.resolution,
                        alpha=alpha,
                    )
                ).mean()
            precision.backward_step(loss_g_main, optimizer_g)
            update_ema_by_images(
                averaged_generator,
                generator,
                batch_size,
                EMA_HALF_LIFE_KIMG,
            )
        finally:
            discriminator.requires_grad_(True)

        metrics.update(
            (loss_d_main, loss_g_main, weighted_r1),
            num_examples=batch_size,
        )
        global_step += 1

    return metrics.compute(), global_step, batches


def save_training_samples(
    generator,
    fixed_z,
    output_path,
    phase,
    seen_kimg,
    precision,
):
    """Save fixed-latent and fixed-noise EMA samples."""

    def generate(z_batch):
        with precision.autocast():
            return generator(
                z_batch,
                resolution=phase.resolution,
                alpha=phase_alpha(phase, phase.num_batches - 1),
                noise_mode="fixed",
            ).float()

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
        title=(
            f"StyleGAN EMA fixed samples - {phase.resolution}x"
            f"{phase.resolution} {phase.name} - {seen_kimg:.0f} kimg"
        ),
    )


def _resolved_run_options(args):
    phase_kimg = args.phase_kimg if args.phase_kimg is not None else PHASE_KIMG
    batch_scale = args.batch_scale if args.batch_scale is not None else 1
    d_reg_every = args.d_reg_every if args.d_reg_every is not None else D_REG_EVERY
    reg_batch_shrink = (
        args.reg_batch_shrink if args.reg_batch_shrink is not None else REG_BATCH_SHRINK
    )
    default_workers = min(8, os.cpu_count() or NUM_WORKERS)
    num_workers = default_workers if args.num_workers is None else args.num_workers
    if (
        min(
            phase_kimg,
            batch_scale,
            d_reg_every,
            reg_batch_shrink,
            args.prefetch_factor,
        )
        < 1
    ):
        raise ValueError("training counts and scales must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    batch_sizes = {
        resolution: batch_size * batch_scale
        for resolution, batch_size in BATCH_SIZES.items()
    }
    return {
        "phase_kimg": phase_kimg,
        "batch_sizes": batch_sizes,
        "d_reg_every": d_reg_every,
        "reg_batch_shrink": reg_batch_shrink,
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
    training_schedule = build_training_schedule(
        dataset_size,
        batch_sizes=options["batch_sizes"],
        phase_kimg=options["phase_kimg"],
    )
    total_phases = len(training_schedule)

    if args.resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    generator = StyleGANGenerator(**MODEL_CONFIG).to(
        device,
        memory_format=torch.channels_last,
    )
    discriminator = StyleGANDiscriminator(**DISCRIMINATOR_CONFIG).to(
        device,
        memory_format=torch.channels_last,
    )
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=device,
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=device,
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    print(
        f"precision={precision.name} pipeline={pipeline} "
        f"workers={options['num_workers']} "
        f"phase_kimg={options['phase_kimg']} "
        f"D_reg_every={options['d_reg_every']}"
    )

    models = {
        "online_generator": generator,
        "stylegan_generator": averaged_generator,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    checkpoint = TrainingCheckpoint(
        CHECKPOINT_DIR / "latest.pth",
        unit="progressive-phase-main-metrics-v2",
        models=models,
        optimizers=optimizers,
    )
    run_config = {
        "precision": precision.name,
        "data_pipeline": pipeline,
        "phase_kimg": options["phase_kimg"],
        "batch_sizes": options["batch_sizes"],
        "d_reg_every": options["d_reg_every"],
        "reg_batch_shrink": options["reg_batch_shrink"],
    }
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    completed_phases, state = checkpoint.resume(
        args.resume_from,
        initial_state={
            "loss_history": {
                "kimg": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
            "global_step": 0,
            "run_config": run_config,
        },
    )
    if completed_phases > total_phases:
        raise ValueError("Checkpoint phase exceeds the training schedule.")
    if state.get("run_config") != run_config:
        raise ValueError(
            "Checkpoint runtime options differ from this run; reuse the "
            "original batch, budget, regularization, and data-pipeline "
            "options."
        )
    if args.resume_from is not None:
        print(f"Resumed after phase {completed_phases}: {args.resume_from}")

    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)
    global_step = int(state["global_step"])
    completed_schedule = training_schedule[:completed_phases]
    seen_images = sum(phase.num_images for phase in completed_schedule)
    expected_steps = sum(phase.num_batches for phase in completed_schedule)
    if global_step != expected_steps:
        raise ValueError("Checkpoint global step does not match its phase.")

    loader = None
    batches = None
    loader_key = None
    for phase_index, phase in enumerate(
        training_schedule[completed_phases:],
        start=completed_phases + 1,
    ):
        print(
            f"Phase {phase_index}/{total_phases}: "
            f"{phase.resolution}x{phase.resolution} {phase.name}"
        )
        key = (
            phase.batch_size,
            None if pipeline == "cuda" else phase.resolution,
        )
        if key != loader_key:
            loader = make_celeba_training_loader(
                DATA_DIR,
                phase.resolution,
                phase.batch_size,
                device,
                pipeline=pipeline,
                num_workers=options["num_workers"],
                prefetch_factor=options["prefetch_factor"],
            )
            batches = iter(loader)
            loader_key = key

        metrics, global_step, batches = train_phase(
            generator,
            discriminator,
            loader,
            batches,
            optimizer_g,
            optimizer_d,
            averaged_generator,
            phase,
            device,
            pipeline,
            precision,
            global_step,
            options["d_reg_every"],
            options["reg_batch_shrink"],
        )
        seen_images += phase.num_images
        seen_kimg = seen_images / 1_000
        loss_history["kimg"].append(seen_kimg)
        for name, value in metrics.items():
            loss_history[name].append(value)

        state["global_step"] = global_step
        checkpoint.save(phase_index, state)
        save_training_samples(
            averaged_generator,
            fixed_z,
            TRAINING_DIR / f"kimg_{round(seen_kimg):05d}.png",
            phase,
            seen_kimg,
            precision,
        )

    save_model_weights(
        averaged_generator,
        OUT_DIR / "stylegan_generator.pth",
        metadata={"model_name": "stylegan", "model_config": MODEL_CONFIG},
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D logistic": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G non-saturating": loss_history["loss_g_main"],
            },
            "R1 regularization": {
                "Expected weighted contribution": loss_history["r1_penalty"],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive 128x128 CelebA StyleGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--phase-kimg", type=int)
    parser.add_argument("--batch-scale", type=int)
    parser.add_argument("--d-reg-every", type=int)
    parser.add_argument("--reg-batch-shrink", type=int)
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
