"""Train a compact, paper-oriented StyleGAN on CIFAR-10.

This lesson keeps ProGAN's 4x4-to-32x32 progressive schedule and adds the
original StyleGAN generator ideas:
    - a normalized Z-to-W mapping network and learned constant input;
    - per-layer AdaIN styles and independent stochastic noise;
    - style-mixing regularization and a moving W center for truncation;
    - non-saturating logistic loss with R1 regularization;
    - an exponential moving average of G for monitoring and final sampling.

The original model targets high-resolution faces with a deeper mapping
network and a much longer image-count schedule.  This CIFAR-10 adaptation
keeps every algorithmic mechanism but uses smaller widths and epoch-based
progressive phases.  Lazy regularization is intentionally left for StyleGAN2.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/stylegan/training/epoch_*.png: fixed-z/noise EMA samples
    output/stylegan/checkpoints/latest.pth: full recoverable training state
    output/stylegan/checkpoints/epoch_*.pth: phase-boundary state archives
    output/stylegan/stylegan_generator.pth: final EMA generator with w_avg
    output/stylegan/online_generator.pth: final non-averaged generator
    output/stylegan/discriminator.pth: final discriminator and configuration
    output/stylegan/loss_curves.png: logistic objectives and R1 loss

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.1_stylegan.py \
        --resume-from output/stylegan/checkpoints/latest.pth

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,920 at 4x4/8x8; 49,984 at 16x16/32x32
Progressive phases:      7 x 12 epochs (84 global epochs)
Optimizer updates:       70,272 D / 70,272 G (1:1)

Generator:                2.76 M params (plus one EMA copy)
Discriminator:            3.38 M params
Trainable total:          6.14 M params
"""

import argparse
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.stylegan import (
    StyleGANDiscriminator,
    StyleGANGenerator,
    denormalize,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan"
TRAINING_DIR = OUT_DIR / "training"

RESOLUTIONS = (4, 8, 16, 32)
BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32}
EPOCHS_PER_PHASE = 12
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 4
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
EMA_DECAY = 0.999
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY_EPOCHS = 4
CHECKPOINT_EVERY_EPOCHS = 4
ARCHIVE_EVERY_EPOCHS = EPOCHS_PER_PHASE
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
    "loss_d",
    "loss_d_main",
    "loss_g",
    "weighted_r1",
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
    """Increase alpha from zero to one over a complete fade-in phase."""
    if phase.name != "fade-in":
        return 1.0
    total_steps = phase.num_epochs * phase.num_batches
    if total_steps == 1:
        return 1.0
    current_step = epoch_index * phase.num_batches + batch_index
    return current_step / (total_steps - 1)


def make_loader(resolution, batch_size, device):
    """Create one CIFAR-10 loader for the active resolution."""
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


def sample_mixing_latents(batch_size, z_dim, device):
    """Usually return two z batches to decorrelate adjacent styles."""
    z = torch.randn(batch_size, z_dim, device=device)
    mixing_z = (
        torch.randn_like(z)
        if random.random() < STYLE_MIXING_PROBABILITY
        else None
    )
    return z, mixing_z


def r1_penalty(real_scores, real_images):
    """Measure squared discriminator gradients at real images."""
    gradients = torch.autograd.grad(
        real_scores.sum(),
        real_images,
        create_graph=True,
    )[0]
    return gradients.square().flatten(1).sum(dim=1).mean()


@torch.no_grad()
def update_ema(averaged_generator, generator, decay=EMA_DECAY):
    """Move StyleGAN's sampling weights toward the online generator."""
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
    """Train one progressive StyleGAN epoch with logistic/R1 objectives."""
    if len(loader) != phase.num_batches:
        raise ValueError("loader length does not match the training schedule.")

    metric_sums = torch.zeros(len(METRIC_NAMES), device=device)
    num_examples = 0

    for batch_index, (real_images, _) in enumerate(loader):
        alpha = phase_alpha(phase, epoch_index, batch_index)
        real_images = real_images.to(device, non_blocking=True)
        real_images.requires_grad_(True)
        batch_size = real_images.shape[0]

        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
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
            )
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
                update_w_avg=True,
            )
            loss_g = F.softplus(
                -discriminator(
                    fake_images,
                    resolution=phase.resolution,
                    alpha=alpha,
                )
            ).mean()
            loss_g.backward()
            optimizer_g.step()
            update_ema(averaged_generator, generator)
        finally:
            discriminator.requires_grad_(True)

        metrics = torch.stack(
            [loss_d, loss_d_main, loss_g, weighted_r1]
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
    """Save fixed-z and fixed-noise samples at the active stage."""
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            samples = generator(
                fixed_z,
                resolution=phase.resolution,
                alpha=alpha,
                noise_mode="fixed",
            ).cpu()
    finally:
        generator.train(was_training)
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=(
            f"StyleGAN EMA fixed samples - {phase.resolution}x"
            f"{phase.resolution} {phase.name} - epoch {global_epoch:03d} - "
            f"alpha={alpha:.2f}"
        ),
    )


def main(resume_from=None):
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

    schedule_metadata = [asdict(phase) for phase in training_schedule]
    session = TrainingSession(
        OUT_DIR,
        total_epochs=total_epochs,
        models={
            "online_generator": generator,
            "stylegan_generator": averaged_generator,
            "discriminator": discriminator,
        },
        optimizers={"generator": optimizer_g, "discriminator": optimizer_d},
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata={
            "format_version": 6,
            "conditioning": "unconditional",
            "objective": "non_saturating_logistic_r1",
            "progressive_training": True,
            "training_schedule": schedule_metadata,
            "style_mixing_probability": STYLE_MIXING_PROBABILITY,
            "ema_decay": EMA_DECAY,
        },
        model_metadata={
            "online_generator": {
                "model_name": "stylegan_online_generator",
                "model_config": MODEL_CONFIG,
                "weights": "online",
            },
            "stylegan_generator": {
                "model_name": "stylegan",
                "model_config": MODEL_CONFIG,
                "weights": "ema",
            },
            "discriminator": {
                "model_name": "stylegan_discriminator",
                "model_config": DISCRIMINATOR_CONFIG,
            },
        },
    )
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    start_epoch, state = session.start(
        resume_from,
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
                f"StyleGAN | {phase.resolution}x{phase.resolution} "
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
            "Logistic discriminator objective": {
                "D total": loss_history["loss_d"],
                "D logistic": loss_history["loss_d_main"],
            },
            "Non-saturating generator objective": {
                "G non-saturating": loss_history["loss_g"],
            },
            "R1 regularization": {
                "Weighted R1 contribution": loss_history["weighted_r1"],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="global epoch across progressive stages",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive CIFAR-10 StyleGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
