"""Train a compact, paper-oriented ProGAN on 128x128 aligned CelebA.

This lesson presents the original progressively growing GAN ingredients:
    - G and D grow together from 4x4 to 128x128;
    - each new resolution fades in before a stabilization phase;
    - equalized learning rates, PixelNorm, and minibatch standard deviation;
    - Wasserstein objectives with gradient and critic-drift penalties;
    - an image-count-based exponential moving average of G for sampling.

The original work reaches 1024x1024.  This single-GPU teaching profile stops
at 128x128, measures every phase in thousands of real images (kimg), and keeps
the complete progressive algorithm explicit.  Multi-GPU execution, metric
suites, and fused high-resolution kernels remain intentionally omitted.

Data:
    data/celeba aligned images and official train split, prepared by
    tool_scripts/download_dataset.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/progan/training/kimg_*.png: phase-boundary fixed-z EMA samples
    output/progan/checkpoints/latest.pth: latest phase-boundary training state
    output/progan/progan_generator.pth: final EMA generator and configuration
    output/progan/loss_curves.png: objectives and regularization losses

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.0_progan.py \
        --resume-from output/progan/checkpoints/latest.pth
    Checkpoints with a different phase metric schema are unsupported.

Training data -- aligned CelebA train split:
Training images:         162,770
Progressive phases:      11 x 400 kimg
Real images shown to D:  4.4 million
Optimizer updates:       96,875 D / 96,875 G (1:1)

Generator:                4.06 M params (plus one EMA copy)
Discriminator:            4.59 M params
Trainable total:          8.65 M params
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

from dl_utils.data.celeba import make_aligned_celeba_loader
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.progan import (
    ProGANDiscriminator,
    ProGANGenerator,
)
from dl_utils.gan.inference import generate_in_batches
from dl_utils.gan.progressive_training import (
    build_progressive_schedule,
    phase_alpha,
)
from dl_utils.gan.stylegan_common import RESOLUTIONS, denormalize
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.optimization import update_ema_by_images
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    save_model_weights,
)


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "progan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32, 64: 32, 128: 32}
PHASE_KIMG = 400
NUM_WORKERS = 4
Z_DIM = 128
BASE_CHANNELS = 32
LEARNING_RATE = 1e-3
GRADIENT_PENALTY_WEIGHT = 10.0
DRIFT_PENALTY_WEIGHT = 1e-3
# A half-life measured in real images keeps EMA equally meaningful when the
# progressive schedule changes batch size.  After 10 kimg, the initial EMA
# value's remaining contribution has halved.
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 4
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "base_channels": BASE_CHANNELS,
}

DISCRIMINATOR_CONFIG = {
    "base_channels": BASE_CHANNELS,
}

METRIC_NAMES = (
    "loss_d_main",
    "loss_g_main",
    "gradient_penalty",
)


def gradient_penalty(
    discriminator,
    real_images,
    fake_images,
    resolution,
    alpha,
):
    """Penalize critic gradients away from unit norm on interpolated images."""
    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    interpolated = torch.lerp(fake_images, real_images, epsilon)
    interpolated.requires_grad_(True)
    scores = discriminator(
        interpolated,
        resolution=resolution,
        alpha=alpha,
    )
    gradients = torch.autograd.grad(
        scores.sum(),
        interpolated,
        create_graph=True,
    )[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return (norms - 1.0).square().mean()


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
    """Train one progressive phase with the WGAN-GP objectives visible."""
    metrics = MetricAccumulator(METRIC_NAMES, device=device)
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
        batch_size = real_images.shape[0]

        z = torch.randn(batch_size, generator.z_dim, device=device)  # (B, z_dim)
        with torch.no_grad():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
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
        loss_d_main = fake_scores.mean() - real_scores.mean()
        penalty_gp = gradient_penalty(
            discriminator,
            real_images,
            fake_images,
            phase.resolution,
            alpha,
        )
        weighted_gp = GRADIENT_PENALTY_WEIGHT * penalty_gp
        weighted_drift = DRIFT_PENALTY_WEIGHT * real_scores.square().mean()
        loss_d = loss_d_main + weighted_gp + weighted_drift
        optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        optimizer_d.step()

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, generator.z_dim, device=device)
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
            )
            loss_g_main = -discriminator(
                fake_images,
                resolution=phase.resolution,
                alpha=alpha,
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
            (
                loss_d_main,
                loss_g_main,
                weighted_gp,
            ),
            num_examples=batch_size,
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
    """Save fixed latent samples with the active progressive stage."""
    samples = generate_in_batches(
        fixed_z,
        SAMPLE_BATCH_SIZE,
        lambda z_batch: generator(
            z_batch,
            resolution=phase.resolution,
            alpha=alpha,
        ),
        module=generator,
    )
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=(
            f"ProGAN EMA fixed-z samples - {phase.resolution}x"
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
    training_schedule = build_progressive_schedule(
        resolutions=RESOLUTIONS,
        batch_sizes=BATCH_SIZES,
        phase_kimg=PHASE_KIMG,
    )
    total_phases = len(training_schedule)

    generator = ProGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = ProGANDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
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
        "progan_generator": averaged_generator,
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
        loader = make_aligned_celeba_loader(
            DATA_DIR,
            phase.resolution,
            batch_size=phase.batch_size,
            device=device,
            num_workers=NUM_WORKERS,
        )
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
        OUT_DIR / "progan_generator.pth",
        metadata={
            "model_name": "progan",
            "model_config": MODEL_CONFIG,
        },
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D Wasserstein": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G Wasserstein": loss_history["loss_g_main"],
            },
            "Gradient penalty": {
                "Weighted gradient penalty": loss_history[
                    "gradient_penalty"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive 128x128 CelebA ProGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
