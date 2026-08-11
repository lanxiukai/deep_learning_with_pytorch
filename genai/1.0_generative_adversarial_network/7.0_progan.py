"""Train a compact, paper-oriented ProGAN on CIFAR-10.

This lesson presents the original progressively growing GAN ingredients:
    - G and D grow together from 4x4 to 32x32;
    - each new resolution fades in before a stabilization phase;
    - equalized learning rates, PixelNorm, and minibatch standard deviation;
    - Wasserstein objectives with gradient and critic-drift penalties;
    - an exponential moving average of G for monitoring and final sampling.

The original work targets much larger images and measures stages in thousands
of images.  Here every stage uses short epoch-based phases over CIFAR-10.  The
progressive algorithm remains explicit while multi-GPU execution, metric
suites, and high-resolution engineering are intentionally omitted.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/progan/training/epoch_*.png: fixed-z EMA samples by global epoch
    output/progan/checkpoints/latest.pth: full recoverable training state
    output/progan/checkpoints/epoch_*.pth: phase-boundary state archives
    output/progan/progan_generator.pth: final EMA generator and configuration
    output/progan/online_generator.pth: final non-averaged generator
    output/progan/discriminator.pth: final discriminator and configuration
    output/progan/loss_curves.png: objectives and regularization losses

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.0_progan.py \
        --resume-from output/progan/checkpoints/latest.pth

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,920 at 4x4/8x8; 49,984 at 16x16/32x32
Progressive phases:      7 x 12 epochs (84 global epochs)
Optimizer updates:       70,272 D / 70,272 G (1:1)

Generator:                2.85 M params (plus one EMA copy)
Discriminator:            3.38 M params
Trainable total:          6.23 M params
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.progan import (
    ProGANDiscriminator,
    ProGANGenerator,
    denormalize,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "progan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

RESOLUTIONS = (4, 8, 16, 32)
BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32}
EPOCHS_PER_PHASE = 12
NUM_WORKERS = 4
Z_DIM = 128
BASE_CHANNELS = 32
LEARNING_RATE = 1e-3
GRADIENT_PENALTY_WEIGHT = 10.0
DRIFT_PENALTY_WEIGHT = 1e-3
EMA_DECAY = 0.999
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY_EPOCHS = 4
CHECKPOINT_EVERY_EPOCHS = 4
ARCHIVE_EVERY_EPOCHS = EPOCHS_PER_PHASE
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "base_channels": BASE_CHANNELS,
}

DISCRIMINATOR_CONFIG = {
    "base_channels": BASE_CHANNELS,
}

METRIC_NAMES = (
    "loss_d",
    "wasserstein_d",
    "loss_g",
    "weighted_gp",
    "weighted_drift",
)


@dataclass(frozen=True)
class TrainingPhase:
    """One fixed-resolution segment of the progressive schedule."""

    resolution: int
    name: str
    num_epochs: int
    batch_size: int
    num_batches: int


def build_training_schedule(num_examples):
    """Describe every phase before training so progress and resume are exact."""
    phases = []
    for resolution in RESOLUTIONS:
        names = (
            ("stabilization",)
            if resolution == 4
            else ("fade-in", "stabilization")
        )
        batch_size = BATCH_SIZES[resolution]
        num_batches = num_examples // batch_size
        if num_batches == 0:
            raise ValueError(
                f"batch size {batch_size} exceeds dataset size {num_examples}"
            )
        for name in names:
            phases.append(
                TrainingPhase(
                    resolution=resolution,
                    name=name,
                    num_epochs=EPOCHS_PER_PHASE,
                    batch_size=batch_size,
                    num_batches=num_batches,
                )
            )
    return tuple(phases)


def phase_alpha(phase, epoch_index, batch_index):
    """Increase alpha linearly across one complete fade-in phase."""
    if phase.name != "fade-in":
        return 1.0
    total_steps = phase.num_epochs * phase.num_batches
    if total_steps == 1:
        return 1.0
    current_step = epoch_index * phase.num_batches + batch_index
    return current_step / (total_steps - 1)


def make_loader(resolution, batch_size, device):
    """Create a resolution-specific CIFAR-10 loader."""
    transform_steps = []
    if resolution != 32:
        transform_steps.append(
            transforms.Resize(
                (resolution, resolution),
                antialias=True,
            )
        )
    transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
        transform=transforms.Compose(transform_steps),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
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
    epsilon = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=real_images.device,
    )
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


@torch.no_grad()
def update_ema(averaged_generator, generator, decay=EMA_DECAY):
    """Move ProGAN's sampling weights toward the online generator."""
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1).")
    for averaged_parameter, parameter in zip(
        averaged_generator.parameters(),
        generator.parameters(),
        strict=True,
    ):
        averaged_parameter.lerp_(parameter, 1 - decay)
    for averaged_buffer, buffer in zip(
        averaged_generator.buffers(),
        generator.buffers(),
        strict=True,
    ):
        averaged_buffer.copy_(buffer)


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    phase,
    epoch_index,
    device,
    progress_bar=None,
):
    """Train one progressive epoch with the WGAN-GP objectives visible."""
    if len(loader) != phase.num_batches:
        raise ValueError("loader length does not match the training schedule.")

    metric_sums = torch.zeros(len(METRIC_NAMES), device=device)
    num_examples = 0

    for batch_index, (real_images, _) in enumerate(loader):
        alpha = phase_alpha(phase, epoch_index, batch_index)
        real_images = real_images.to(device, non_blocking=True)
        batch_size = real_images.shape[0]

        z = torch.randn(batch_size, generator.z_dim, device=device)
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
        wasserstein_d = fake_scores.mean() - real_scores.mean()
        penalty_gp = gradient_penalty(
            discriminator,
            real_images,
            fake_images,
            phase.resolution,
            alpha,
        )
        weighted_gp = GRADIENT_PENALTY_WEIGHT * penalty_gp
        weighted_drift = DRIFT_PENALTY_WEIGHT * real_scores.square().mean()
        loss_d = wasserstein_d + weighted_gp + weighted_drift
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
            loss_g = -discriminator(
                fake_images,
                resolution=phase.resolution,
                alpha=alpha,
            ).mean()
            loss_g.backward()
            optimizer_g.step()
            update_ema(averaged_generator, generator)
        finally:
            discriminator.requires_grad_(True)

        metrics = torch.stack(
            [
                loss_d,
                wasserstein_d,
                loss_g,
                weighted_gp,
                weighted_drift,
            ]
        ).detach()
        metric_sums += metrics * batch_size
        num_examples += batch_size

        if progress_bar is not None:
            progress_bar.update(1)

    return dict(
        zip(
            METRIC_NAMES,
            (value / num_examples for value in metric_sums.tolist()),
            strict=True,
        )
    )


def save_training_samples(
    generator,
    fixed_z,
    output_path,
    phase,
    global_epoch,
    alpha,
):
    """Save fixed latent samples with the active progressive stage."""
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            samples = generator(
                fixed_z,
                resolution=phase.resolution,
                alpha=alpha,
            ).cpu()
    finally:
        generator.train(was_training)
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=(
            f"ProGAN EMA fixed-z samples - {phase.resolution}x"
            f"{phase.resolution} {phase.name} - epoch {global_epoch:03d} - "
            f"alpha={alpha:.2f}"
        ),
    )


def main(resume_from=None):
    if resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    set_seed(SEED)
    device = try_gpu()
    dataset_info = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
    )
    training_schedule = build_training_schedule(len(dataset_info))
    scheduled_epochs = tuple(
        (phase, epoch_index)
        for phase in training_schedule
        for epoch_index in range(phase.num_epochs)
    )
    total_epochs = len(scheduled_epochs)
    total_steps = sum(phase.num_batches for phase, _ in scheduled_epochs)

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

    schedule_metadata = [asdict(phase) for phase in training_schedule]
    session = TrainingSession(
        OUT_DIR,
        total_epochs=total_epochs,
        models={
            "online_generator": generator,
            "progan_generator": averaged_generator,
            "discriminator": discriminator,
        },
        optimizers={"generator": optimizer_g, "discriminator": optimizer_d},
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata={
            "format_version": 5,
            "conditioning": "unconditional",
            "objective": "wasserstein_gp",
            "progressive_training": True,
            "training_schedule": schedule_metadata,
            "ema_decay": EMA_DECAY,
        },
        model_metadata={
            "online_generator": {
                "model_name": "progan_online_generator",
                "model_config": MODEL_CONFIG,
                "weights": "online",
            },
            "progan_generator": {
                "model_name": "progan",
                "model_config": MODEL_CONFIG,
                "weights": "ema",
            },
            "discriminator": {
                "model_name": "progan_discriminator",
                "model_config": DISCRIMINATOR_CONFIG,
            },
        },
    )
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    start_epoch, state = session.start(
        resume_from,
        reset_output_dir=False,
        initial_state={
            "loss_history": {
                "epoch": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
        },
    )
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)

    if loss_history["epoch"] != list(range(1, start_epoch)):
        raise ValueError("Checkpoint loss history does not match its epoch.")
    if tuple(fixed_z.shape) != (SAMPLES_TO_DISPLAY, Z_DIM):
        raise ValueError("Checkpoint fixed z has an unexpected shape.")
    if resume_from is not None:
        print(f"Resumed training from epoch {start_epoch - 1}: {resume_from}")

    completed_steps = sum(
        phase.num_batches
        for phase, _ in scheduled_epochs[: start_epoch - 1]
    )
    active_phase = None
    loader = None
    with tqdm(
        total=total_steps,
        initial=completed_steps,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for global_epoch, (phase, epoch_index) in enumerate(
            scheduled_epochs,
            start=1,
        ):
            if global_epoch < start_epoch:
                continue
            if phase != active_phase:
                loader = make_loader(
                    phase.resolution,
                    phase.batch_size,
                    device,
                )
                active_phase = phase
            assert loader is not None

            progress_bar.set_description(
                f"ProGAN | {phase.resolution}x{phase.resolution} "
                f"{phase.name} | Epoch {global_epoch}/{total_epochs}",
                refresh=False,
            )
            metrics = train_epoch(
                generator,
                discriminator,
                loader,
                optimizer_g,
                optimizer_d,
                averaged_generator,
                phase,
                epoch_index,
                device,
                progress_bar=progress_bar,
            )
            loss_history["epoch"].append(global_epoch)
            for name, value in metrics.items():
                loss_history[name].append(value)

            alpha = phase_alpha(
                phase,
                epoch_index,
                phase.num_batches - 1,
            )
            should_sample = (
                epoch_index == 0
                or global_epoch % SAMPLE_EVERY_EPOCHS == 0
                or global_epoch == total_epochs
            )
            if should_sample:
                save_training_samples(
                    averaged_generator,
                    fixed_z,
                    TRAINING_DIR / f"epoch_{global_epoch:03d}.png",
                    phase,
                    global_epoch,
                    alpha,
                )

            session.checkpoint(global_epoch, state)

    session.finish()
    save_loss_panels(
        loss_history["epoch"],
        {
            "Wasserstein discriminator objective": {
                "D total": loss_history["loss_d"],
                "D Wasserstein": loss_history["wasserstein_d"],
            },
            "Wasserstein generator objective": {
                "G Wasserstein": loss_history["loss_g"],
            },
            "Gradient penalty": {
                "Weighted gradient penalty": loss_history["weighted_gp"],
            },
            "Drift penalty": {
                "Weighted drift penalty": loss_history["weighted_drift"],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="global epoch across progressive stages",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive CIFAR-10 ProGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
