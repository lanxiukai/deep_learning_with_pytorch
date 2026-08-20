"""Train an unconditional, paper-oriented SN-GAN on CIFAR-10.

This lesson follows the CIFAR ResNet experiment in *Spectral Normalization for
Generative Adversarial Networks*:
    - CIFAR-10 is modeled without using its class labels;
    - z has 128 dimensions and initializes a 4x4x256 feature map;
    - G contains three constant-width upsampling residual blocks;
    - D contains two downsampling and two same-resolution residual blocks;
    - spectral normalization is applied to every learned map in D, but not G;
    - the hinge objective and an exact five D updates per G update use the
      paper's ResNet training setting;
    - D uses batches of 64 real/fake images while G uses batches of 128;
    - real uint8 pixels use uniform dequantization before entering D.

Projection discrimination and conditional BatchNorm are intentionally absent.
They first appear in ``6.1_sagan.py``, whose original paper explicitly uses
both mechanisms.  Fixed-noise samples, loss curves, and full checkpoints are
teaching conveniences and do not alter the adversarial updates.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/sn_gan/training/epoch_*.png: fixed-z unconditional samples
    output/sn_gan/checkpoints/latest.pth: full recoverable training state
    output/sn_gan/checkpoints/epoch_*.pth: sparse full-state archives
    output/sn_gan/generator.pth: final generator and configuration
    output/sn_gan/discriminator.pth: final discriminator and configuration
    output/sn_gan/loss_curves.png: separate D and G loss panels

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/6.0_sn_gan.py \
        --resume-from output/sn_gan/checkpoints/latest.pth

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)
Training epochs:         650
Optimizer updates:       507,650 D / 101,530 G (exactly 5:1)

Generator:                4.28 M params
Discriminator:            1.05 M params
Total:                    5.33 M params
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from tqdm import tqdm

from dl_utils.data.cifar10 import make_cifar10_loader
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.sn_gan import (
    SNDiscriminator,
    SNGenerator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    uniform_dequantize_uint8,
)
from dl_utils.gan.update_schedule import UpdateRatioSchedule
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_fixed_noise_samples
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sn_gan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

NUM_EPOCHS = 650
DISCRIMINATOR_BATCH_SIZE = 64
GENERATOR_BATCH_SIZE = 128
NUM_WORKERS = 8
Z_DIM = 128
GENERATOR_BASE_CHANNELS = 256
DISCRIMINATOR_BASE_CHANNELS = 128
LEARNING_RATE = 2e-4
DISCRIMINATOR_UPDATES_PER_GENERATOR = 5
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY_EPOCHS = 50
CHECKPOINT_EVERY_EPOCHS = 10
ARCHIVE_EVERY_EPOCHS = 50
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "base_channels": GENERATOR_BASE_CHANNELS,
}

DISCRIMINATOR_CONFIG = {
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
    """Train one epoch while preserving the exact five-to-one update phase."""
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

    for real, _ in loader:
        real = real.to(device, non_blocking=True)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, Z_DIM, device=device)
        with torch.no_grad():
            fake = generator(noise)
        real_scores = discriminator(real)
        fake_scores = discriminator(fake)
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
            noise = torch.randn(
                GENERATOR_BATCH_SIZE,
                Z_DIM,
                device=device,
            )
            discriminator.requires_grad_(False)
            try:
                opt_g.zero_grad(set_to_none=True)
                fake = generator(noise)
                loss_g = generator_hinge_loss(discriminator(fake))
                loss_g.backward()
                opt_g.step()
            finally:
                discriminator.requires_grad_(True)

            generator_metrics.update(
                (loss_g,),
                num_examples=GENERATOR_BATCH_SIZE,
            )

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
        DISCRIMINATOR_BATCH_SIZE,
        device,
        train=True,
        transform=transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(uniform_dequantize_uint8),
            ]
        ),
        num_workers=NUM_WORKERS,
    )
    update_schedule = UpdateRatioSchedule(
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
        len(loader),
        NUM_EPOCHS,
    )

    generator = SNGenerator(**MODEL_CONFIG).to(device)
    discriminator = SNDiscriminator(**DISCRIMINATOR_CONFIG).to(device)

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
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
            "format_version": 5,
            "conditioning": "unconditional",
            "update_ratio": DISCRIMINATOR_UPDATES_PER_GENERATOR,
        },
        model_metadata={
            "generator": {"model_name": "sn_gan", "model_config": MODEL_CONFIG},
            "discriminator": {
                "model_name": "sn_gan_discriminator",
                "model_config": DISCRIMINATOR_CONFIG,
            },
        },
    )
    fixed_noise = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
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
            "discriminator_steps": 0,
        },
    )
    loss_history = state["loss_history"]
    fixed_noise = state["fixed_noise"].to(device)
    discriminator_steps = state["discriminator_steps"]

    planned_discriminator_updates = (
        update_schedule.total_discriminator_updates
    )
    planned_generator_updates = update_schedule.total_generator_updates
    print(
        "Planned optimizer updates: "
        f"D={planned_discriminator_updates:,}, "
        f"G={planned_generator_updates:,} "
        f"(exactly {DISCRIMINATOR_UPDATES_PER_GENERATOR}:1)"
    )
    expected_steps = update_schedule.completed_discriminator_updates(
        start_epoch - 1
    )
    if discriminator_steps != expected_steps:
        raise ValueError("Checkpoint discriminator step count is inconsistent.")
    if loss_history["epoch"] != list(range(1, start_epoch)):
        raise ValueError("Checkpoint loss history does not match its epoch.")
    if fixed_noise.shape != (SAMPLES_TO_DISPLAY, Z_DIM):
        raise ValueError("Checkpoint fixed noise has an unexpected shape.")
    if resume_from is not None:
        print(f"Resumed training from epoch {start_epoch - 1}: {resume_from}")

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
                save_fixed_noise_samples(
                    generator,
                    fixed_noise,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    columns=SAMPLE_GRID_COLUMNS,
                    title="Unconditional SN-GAN fixed-z samples",
                    epoch=epoch,
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
        description="Train the unconditional CIFAR-10 SN-GAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
