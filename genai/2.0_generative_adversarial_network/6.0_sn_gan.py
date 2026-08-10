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
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/sn_gan/training/epoch_*.png: fixed-z unconditional samples
    output/sn_gan/checkpoints/latest.pth: full recoverable training state
    output/sn_gan/checkpoints/epoch_*.pth: sparse full-state archives
    output/sn_gan/generator.pth: final generator and configuration
    output/sn_gan/discriminator.pth: final discriminator and configuration
    output/sn_gan/loss_curves.png: separate D and G loss panels

Resume an interrupted run:
    python genai/2.0_generative_adversarial_network/6.0_sn_gan.py \
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.sn_gan import (
    SNDiscriminator,
    SNGenerator,
    count_spectral_norm_layers,
    discriminator_hinge_loss,
    generator_hinge_loss,
    uniform_dequantize_uint8,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_fixed_noise_samples
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sn_gan"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 650
DISCRIMINATOR_BATCH_SIZE = 64
GENERATOR_BATCH_SIZE = 128
NUM_WORKERS = 8
Z_DIM = 128
GENERATOR_BASE_CHANNELS = 256
DISCRIMINATOR_BASE_CHANNELS = 128
IMAGE_SIZE = 32
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
    "image_size": IMAGE_SIZE,
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
):
    """Train one epoch while preserving the exact five-to-one update phase."""
    if discriminator_steps < 0:
        raise ValueError("discriminator_steps must be non-negative.")
    discriminator_loss_sum = torch.zeros((), device=device)
    generator_loss_sum = torch.zeros((), device=device)
    discriminator_examples = 0
    generator_examples = 0

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

        discriminator_loss_sum += loss_d.detach() * batch_size
        discriminator_examples += batch_size
        discriminator_steps += 1

        should_update_generator = (
            discriminator_steps % DISCRIMINATOR_UPDATES_PER_GENERATOR == 0
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

            generator_loss_sum += loss_g.detach() * GENERATOR_BATCH_SIZE
            generator_examples += GENERATOR_BATCH_SIZE

        if progress_bar is not None:
            progress_bar.update(1)

    if generator_examples == 0:
        raise RuntimeError("No generator update occurred in this epoch.")

    return (
        (discriminator_loss_sum / discriminator_examples).item(),
        (generator_loss_sum / generator_examples).item(),
        discriminator_steps,
    )


def main(resume_from=None):
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    set_seed(SEED)
    device = try_gpu()
    dataset = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(uniform_dequantize_uint8),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=DISCRIMINATOR_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    generator = SNGenerator(**MODEL_CONFIG).to(device)
    discriminator = SNDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    sn_layer_counts = (
        count_spectral_norm_layers(generator),
        count_spectral_norm_layers(discriminator),
    )
    if sn_layer_counts[0] != 0 or sn_layer_counts[1] == 0:
        raise RuntimeError("SN-GAN requires spectral norm in D but not G.")

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
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_noise = state["fixed_noise"].to(device)
    discriminator_steps = state["discriminator_steps"]

    planned_discriminator_updates = NUM_EPOCHS * len(loader)
    planned_generator_updates, incomplete_update_cycle = divmod(
        planned_discriminator_updates,
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
    )
    if incomplete_update_cycle:
        raise RuntimeError(
            "The configured epoch count does not end on an exact 5D:1G "
            "update boundary."
        )
    print(
        "Planned optimizer updates: "
        f"D={planned_discriminator_updates:,}, "
        f"G={planned_generator_updates:,} "
        f"(exactly {DISCRIMINATOR_UPDATES_PER_GENERATOR}:1)"
    )
    expected_steps = (start_epoch - 1) * len(loader)
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
