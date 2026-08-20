"""Train a compact, paper-oriented class-conditional SAGAN on CIFAR-10.

This lesson starts from the ResNet hinge-GAN training in ``6.0_sn_gan.py``
and presents SAGAN's algorithmic additions:
    - class-indexed conditional BatchNorm in G;
    - projection conditioning in D;
    - non-local self-attention with pooled key/value branches and a
      zero-initialized residual gate in G/D;
    - spectral normalization in both G and D;
    - TTUR with a configurable number of D updates per G update.

The original paper trains 128x128 ImageNet models.  Here the same ideas are
kept in a three-block 32x32 CIFAR-10 model with ordinary single-device
BatchNorm.  G tapers channels from 512 at 4x4 to 64 at 32x32, while D keeps
128 channels and applies attention after its first 16x16 block.  Random
horizontal flips are retained as a small, useful image augmentation.  Loss
curves, fixed class samples, and full checkpoints are training conveniences
and do not alter the adversarial updates.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/sagan/training/epoch_*.png: fixed-z samples grouped by class
    output/sagan/checkpoints/latest.pth: full recoverable training state
    output/sagan/checkpoints/epoch_*.pth: sparse full-state archives
    output/sagan/generator.pth: final generator and configuration
    output/sagan/discriminator.pth: final discriminator and configuration
    output/sagan/loss_curves.png: separate D and G loss panels

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/6.1_sagan.py \
        --resume-from output/sagan/checkpoints/latest.pth

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)
Training epochs:         100
Optimizer updates:        78,100 D / 78,100 G (n_D:1, n_D=1)

Generator:                3.60 M params
Discriminator:            1.08 M params
Total:                    4.68 M params
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dl_utils.data.cifar10 import (
    make_cifar10_loader,
    normalized_cifar10_transform,
)
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.sagan import (
    CIFAR10_CLASS_NAMES,
    SAGANDiscriminator,
    SAGANGenerator,
    make_fixed_class_latent_grid,
)
from dl_utils.gan.sn_gan import (
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.gan.update_schedule import UpdateRatioSchedule
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sagan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

NUM_EPOCHS = 100
BATCH_SIZE = 64
# If BATCH_SIZE is raised to 128, raise NUM_EPOCHS to 200 as well.
NUM_WORKERS = 8
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 128
GENERATOR_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 128
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 4e-4
DISCRIMINATOR_UPDATES_PER_GENERATOR = 1
SAMPLES_TO_DISPLAY = NUM_CLASSES * SAMPLES_PER_CLASS
SAMPLE_EVERY_EPOCHS = 5
CHECKPOINT_EVERY_EPOCHS = 5
ARCHIVE_EVERY_EPOCHS = 10
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": GENERATOR_BASE_CHANNELS,
}

DISCRIMINATOR_CONFIG = {
    "num_classes": NUM_CLASSES,
    "base_channels": DISCRIMINATOR_BASE_CHANNELS,
}


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    device,
    discriminator_steps,
    progress_bar=None,
    update_schedule=None,
):
    """Train one epoch while preserving SAGAN's global n_D-to-one phase."""
    if discriminator_steps < 0:
        raise ValueError("discriminator_steps must be non-negative.")
    if update_schedule is None:
        update_schedule = UpdateRatioSchedule(
            DISCRIMINATOR_UPDATES_PER_GENERATOR,
            len(loader),
            NUM_EPOCHS,
        )
    discriminator_metrics = MetricAccumulator(
        ("loss",),
        device=device,
    )
    generator_metrics = MetricAccumulator(
        ("loss",),
        device=device,
    )

    for real, labels in loader:
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = real.shape[0]

        # D learns both realism and whether each image matches its label.
        noise = torch.randn(batch_size, Z_DIM, device=device)
        with torch.no_grad():
            fake = generator(noise, labels)
        real_scores = discriminator(real, labels)
        fake_scores = discriminator(fake, labels)
        loss_d = discriminator_hinge_loss(real_scores, fake_scores)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        discriminator_metrics.update((loss_d,), num_examples=batch_size)
        discriminator_steps += 1

        should_update_generator = update_schedule.generator_due(
            discriminator_steps
        )
        if should_update_generator:
            # G receives the class signal through conditional BatchNorm and
            # D's projection term. Freezing D avoids unused D gradients.
            sampled_labels = torch.randint(
                NUM_CLASSES,
                (batch_size,),
                device=device,
            )
            noise = torch.randn(batch_size, Z_DIM, device=device)
            discriminator.requires_grad_(False)
            try:
                opt_g.zero_grad(set_to_none=True)
                fake = generator(noise, sampled_labels)
                loss_g = generator_hinge_loss(
                    discriminator(fake, sampled_labels)
                )
                loss_g.backward()
                opt_g.step()
            finally:
                discriminator.requires_grad_(True)

            generator_metrics.update((loss_g,), num_examples=batch_size)

        if progress_bar is not None:
            progress_bar.update(1)

    return (
        discriminator_metrics.compute()["loss"],
        generator_metrics.compute()["loss"],
        discriminator_steps,
    )


def main(resume_from=None):
    if resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    set_seed(SEED)
    device = try_gpu()
    loader = make_cifar10_loader(
        DATA_DIR,
        BATCH_SIZE,
        device,
        train=True,
        transform=normalized_cifar10_transform(horizontal_flip=True),
        num_workers=NUM_WORKERS,
    )
    update_schedule = UpdateRatioSchedule(
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
        len(loader),
        NUM_EPOCHS,
    )

    generator = SAGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = SAGANDiscriminator(**DISCRIMINATOR_CONFIG).to(device)

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=GENERATOR_LR,
        betas=(0.0, 0.9),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=DISCRIMINATOR_LR,
        betas=(0.0, 0.9),
    )

    session = TrainingSession(
        OUT_DIR,
        total_epochs=NUM_EPOCHS,
        models={"generator": generator, "discriminator": discriminator},
        optimizers={"generator": opt_g, "discriminator": opt_d},
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata={
            "format_version": 8,
            "conditioning": "class_conditional",
            "discriminator_conditioning": "projection",
            "update_ratio": DISCRIMINATOR_UPDATES_PER_GENERATOR,
        },
        model_metadata={
            "generator": {
                "model_name": "sagan",
                "model_config": MODEL_CONFIG,
            },
            "discriminator": {
                "model_name": "sagan_discriminator",
                "model_config": DISCRIMINATOR_CONFIG,
            },
        },
    )
    fixed_noise, fixed_labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        SAMPLES_PER_CLASS,
        Z_DIM,
        device,
    )
    start_epoch, state = session.start(
        resume_from,
        reset_output_dir=False,
        initial_state={
            "loss_history": {
                "epoch": [],
                "discriminator": [],
                "generator": [],
            },
            "fixed_noise": fixed_noise.cpu(),
            "fixed_labels": fixed_labels.cpu(),
            "discriminator_steps": 0,
        },
    )
    loss_history = state["loss_history"]
    fixed_noise = state["fixed_noise"].to(device)
    fixed_labels = state["fixed_labels"].to(device)
    discriminator_steps = state["discriminator_steps"]

    planned_discriminator_updates = (
        update_schedule.total_discriminator_updates
    )
    planned_generator_updates = update_schedule.total_generator_updates
    expected_steps = update_schedule.completed_discriminator_updates(
        start_epoch - 1
    )
    if discriminator_steps != expected_steps:
        raise ValueError("Checkpoint discriminator step count is inconsistent.")
    if loss_history["epoch"] != list(range(1, start_epoch)):
        raise ValueError("Checkpoint loss history does not match its epoch.")
    expected_shape = (SAMPLES_TO_DISPLAY, Z_DIM)
    if tuple(fixed_noise.shape) != expected_shape:
        raise ValueError("Checkpoint fixed noise has an unexpected shape.")
    if tuple(fixed_labels.shape) != (SAMPLES_TO_DISPLAY,):
        raise ValueError("Checkpoint fixed labels have an unexpected shape.")
    if fixed_labels.dtype != torch.long:
        raise ValueError("Checkpoint fixed labels must use torch.long.")
    if resume_from is not None:
        print(f"Resumed training from epoch {start_epoch - 1}: {resume_from}")

    print(
        "Planned optimizer updates: "
        f"D={planned_discriminator_updates:,}, "
        f"G={planned_generator_updates:,} "
        f"(exactly {DISCRIMINATOR_UPDATES_PER_GENERATOR}:1)"
    )
    with tqdm(
        total=planned_discriminator_updates,
        initial=discriminator_steps,
        desc=f"Epoch {min(start_epoch, NUM_EPOCHS)}/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{NUM_EPOCHS}",
                refresh=False,
            )

            loss_d, loss_g, discriminator_steps = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                device,
                discriminator_steps,
                progress_bar=progress_bar,
                update_schedule=update_schedule,
            )
            loss_history["epoch"].append(epoch)
            loss_history["discriminator"].append(loss_d)
            loss_history["generator"].append(loss_g)

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    generator,
                    fixed_noise,
                    fixed_labels,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    class_names=CIFAR10_CLASS_NAMES[:NUM_CLASSES],
                    title=f"SAGAN fixed class samples - epoch {epoch:03d}",
                    shared_latents_across_classes=True,
                )

            state["discriminator_steps"] = discriminator_steps
            session.checkpoint(epoch, state)

    if discriminator_steps != planned_discriminator_updates:
        raise RuntimeError("Training ended with an unexpected D update count.")
    session.finish()
    save_loss_panels(
        loss_history["epoch"],
        {
            "Discriminator hinge loss": {
                "D hinge loss": loss_history["discriminator"],
            },
            "Generator adversarial loss": {
                "G adversarial loss": loss_history["generator"],
            },
        },
        OUT_DIR / "loss_curves.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the class-conditional CIFAR-10 SAGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
