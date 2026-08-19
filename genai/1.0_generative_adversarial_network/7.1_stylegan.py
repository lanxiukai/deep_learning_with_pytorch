"""Train a compact, paper-oriented StyleGAN on 128x128 aligned CelebA.

This lesson keeps ProGAN's 4x4-to-128x128 progressive schedule and adds the
original StyleGAN generator ideas:
    - a normalized Z-to-W mapping network and learned constant input;
    - per-layer AdaIN styles and independent stochastic noise;
    - style-mixing regularization and a moving W center for truncation;
    - non-saturating logistic loss with R1 regularization;
    - an image-count-based exponential moving average of G for sampling.

The original model reaches 1024x1024 with wider features and a much longer
schedule.  This single-GPU teaching profile keeps its eight-layer mapping
depth and every algorithmic mechanism, but stops at 128x128 and measures each
phase in thousands of real images (kimg).  Lazy regularization remains
intentionally reserved for StyleGAN2.

Data:
    data/celeba aligned images and official train split, prepared by
    tool_scripts/download_dataset.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/stylegan/training/kimg_*.png: phase-boundary fixed-z/noise samples
    output/stylegan/checkpoints/latest.pth: latest phase-boundary training state
    output/stylegan/stylegan_generator.pth: final EMA generator with w_avg
    output/stylegan/loss_curves.png: logistic objectives and R1 loss

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.1_stylegan.py \
        --resume-from output/stylegan/checkpoints/latest.pth
    Checkpoints with a different phase metric schema are unsupported.

Training data -- aligned CelebA train split:
Training images:         162,770
Progressive phases:      11 x 400 kimg
Real images shown to D:  4.4 million
Optimizer updates:       96,875 D / 96,875 G (1:1)

Generator:                4.18 M params (plus one EMA copy)
Discriminator:            4.59 M params
Trainable total:          8.77 M params
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.data.celeba import make_aligned_celeba_loader
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.stylegan import (
    StyleGANDiscriminator,
    StyleGANGenerator,
)
from dl_utils.gan.inference import generate_in_batches
from dl_utils.gan.progressive_training import (
    build_progressive_schedule,
    phase_alpha,
)
from dl_utils.gan.stylegan_common import RESOLUTIONS, denormalize
from dl_utils.gan.stylegan_training import (
    r1_penalty,
    sample_mixing_latents,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.metrics import WeightedMetricAccumulator
from dl_utils.training.optimization import update_ema_by_images
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    save_model_weights,
)


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32, 64: 32, 128: 32}
PHASE_KIMG = 400
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
# Express smoothing in images rather than optimizer steps: progressive stages
# use different batches, but every 10 kimg always halves old EMA information.
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 4
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
}

DISCRIMINATOR_CONFIG = {
    "base_channels": BASE_CHANNELS,
}

METRIC_NAMES = (
    "loss_d_main",
    "loss_g_main",
    "r1_penalty",
)


def build_training_schedule():
    """Build the lesson's fade-in and stabilization phases."""
    return build_progressive_schedule(
        resolutions=RESOLUTIONS,
        batch_sizes=BATCH_SIZES,
        phase_kimg=PHASE_KIMG,
    )


def make_loader(resolution, batch_size, device):
    """Create one aligned CelebA loader for the active resolution."""
    return make_aligned_celeba_loader(
        DATA_DIR,
        resolution,
        batch_size=batch_size,
        device=device,
        num_workers=NUM_WORKERS,
    )


def train_phase(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    phase,
    device,
):
    """Train one progressive phase with logistic and R1 objectives."""
    metrics = WeightedMetricAccumulator(METRIC_NAMES, device=device)
    batches = iter(loader)
    for batch_index in tqdm(
        range(phase.num_batches),
        desc="Training",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        try:
            real_images, _ = next(batches)
        except StopIteration:
            batches = iter(loader)
            real_images, _ = next(batches)
        alpha = phase_alpha(phase, batch_index)
        real_images = real_images.to(device, non_blocking=True)
        real_images.requires_grad_(True)
        batch_size = real_images.shape[0]

        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
            STYLE_MIXING_PROBABILITY,
        )
        with torch.no_grad():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
            )
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
        loss_d_main = F.softplus(-real_scores).mean()
        loss_d_main = loss_d_main + F.softplus(fake_scores).mean()
        weighted_r1 = 0.5 * R1_GAMMA * r1_penalty(
            real_scores,
            real_images,
        )
        loss_d = loss_d_main + weighted_r1
        optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        optimizer_d.step()
        real_images.requires_grad_(False)

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                device,
                STYLE_MIXING_PROBABILITY,
            )
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
            loss_g_main.backward()
            optimizer_g.step()
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
            weight=batch_size,
        )

    return metrics.compute()


def save_training_samples(
    generator,
    fixed_z,
    output_path,
    phase,
    seen_kimg,
    alpha,
):
    """Save fixed-z and fixed-noise samples at the active stage."""
    samples = generate_in_batches(
        fixed_z,
        SAMPLE_BATCH_SIZE,
        lambda z_batch: generator(
            z_batch,
            resolution=phase.resolution,
            alpha=alpha,
            noise_mode="fixed",
        ),
        module=generator,
    )
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=(
            f"StyleGAN EMA fixed samples - {phase.resolution}x"
            f"{phase.resolution} {phase.name} - {seen_kimg:.0f} kimg - "
            f"alpha={alpha:.2f}"
        ),
    )


def main(resume_from=None):
    if resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    if not (DATA_DIR / "list_eval_partition.csv").is_file():
        raise FileNotFoundError(
            f"CelebA data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    set_seed(SEED)
    device = try_gpu()
    training_schedule = build_training_schedule()
    total_phases = len(training_schedule)

    generator = StyleGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = StyleGANDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )

    models = {
        "online_generator": generator,
        "stylegan_generator": averaged_generator,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    checkpoint = TrainingCheckpoint(
        CHECKPOINT_DIR / "latest.pth",
        unit="phase-main-metrics",
        models=models,
        optimizers=optimizers,
    )
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    completed_phases, state = checkpoint.resume(
        resume_from,
        initial_state={
            "loss_history": {
                "kimg": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
        },
    )
    if resume_from is not None:
        print(f"Resumed after phase {completed_phases}: {resume_from}")
    if completed_phases > total_phases:
        raise ValueError("Checkpoint phase exceeds the training schedule.")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)

    completed_schedule = training_schedule[:completed_phases]
    seen_images = sum(phase.num_images for phase in completed_schedule)
    for phase_index, phase in enumerate(
        training_schedule[completed_phases:],
        start=completed_phases + 1,
    ):
        print(
            f"Phase {phase_index}/{total_phases}: "
            f"{phase.resolution}x{phase.resolution} {phase.name}"
        )
        loader = make_loader(phase.resolution, phase.batch_size, device)
        metrics = train_phase(
            generator,
            discriminator,
            loader,
            optimizer_g,
            optimizer_d,
            averaged_generator,
            phase,
            device,
        )
        seen_images += phase.num_images
        seen_kimg = seen_images / 1_000
        loss_history["kimg"].append(seen_kimg)
        for name, value in metrics.items():
            loss_history[name].append(value)

        save_training_samples(
            averaged_generator,
            fixed_z,
            TRAINING_DIR / f"kimg_{round(seen_kimg):05d}.png",
            phase,
            seen_kimg,
            phase_alpha(phase, phase.num_batches - 1),
        )
        checkpoint.save(phase_index, state)

    save_model_weights(
        averaged_generator,
        OUT_DIR / "stylegan_generator.pth",
        metadata={
            "model_name": "stylegan",
            "model_config": MODEL_CONFIG,
        },
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
                "Weighted R1 contribution": loss_history["r1_penalty"],
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
