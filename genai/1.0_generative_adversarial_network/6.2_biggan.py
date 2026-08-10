"""Train a compact, paper-oriented BigGAN on CIFAR-10.

This lesson keeps SAGAN's class-conditional ResNet, self-attention, spectral
normalization, projection discriminator, and hinge objective.  It then makes
the main BigGAN additions that remain meaningful at 32x32:
    - one class embedding shared by every generator block;
    - hierarchical latent chunks (skip-z) in conditional BatchNorm;
    - orthogonal initialization of G/D and modified orthogonal regularization;
    - two D updates per G update with BigGAN's Adam/TTUR settings;
    - an exponential moving average of G for monitoring and final sampling.

The ImageNet-scale batch, channel widths, cross-replica BatchNorm, and standing
statistics are deliberately omitted.  Normal CIFAR-10 BatchNorm running
statistics are copied into the EMA model instead.  The truncation trick is
kept in ``6.3_sn_sagan_biggan_evaluation.py`` because it is a sampling choice,
not a training loss.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/biggan/training/epoch_*.png: fixed-z EMA samples by class
    output/biggan/checkpoints/latest.pth: full recoverable training state
    output/biggan/checkpoints/epoch_*.pth: sparse full-state archives
    output/biggan/generator.pth: final EMA generator and configuration
    output/biggan/online_generator.pth: final non-averaged generator
    output/biggan/discriminator.pth: final discriminator and configuration
    output/biggan/loss_curves.png: adversarial and regularization losses

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/6.2_biggan.py \
        --resume-from output/biggan/checkpoints/latest.pth

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)
Training epochs:         100
Optimizer updates:       78,100 D / 39,050 G (exactly 2:1)

Generator:                0.84 M params (plus one EMA copy)
Discriminator:            1.06 M params
Trainable total:          1.90 M params
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.biggan import (
    BigGANDiscriminator,
    CompactBigGANGenerator,
    modified_orthogonal_regularization,
)
from dl_utils.genai.sagan import (
    CIFAR10_CLASS_NAMES,
    make_fixed_class_latent_grid,
)
from dl_utils.genai.sn_gan import (
    count_spectral_norm_layers,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "biggan"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 120
BASE_CHANNELS = 32
CLASS_EMBEDDING_DIM = 32
IMAGE_SIZE = 32
GENERATOR_LR = 5e-5
DISCRIMINATOR_LR = 2e-4
DISCRIMINATOR_UPDATES_PER_GENERATOR = 2
ORTHOGONAL_REGULARIZATION_STRENGTH = 1e-4
EMA_DECAY = 0.9999
SAMPLES_TO_DISPLAY = NUM_CLASSES * SAMPLES_PER_CLASS
SAMPLE_EVERY_EPOCHS = 5
CHECKPOINT_EVERY_EPOCHS = 5
ARCHIVE_EVERY_EPOCHS = 25
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "class_embedding_dim": CLASS_EMBEDDING_DIM,
    "image_size": IMAGE_SIZE,
}

DISCRIMINATOR_CONFIG = {
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
}


def initialize_orthogonal_weights(module):
    """Apply BigGAN's orthogonal initialization to learned affine maps."""
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        weight = getattr(module, "weight_orig", None)
        if weight is None:
            weight = module.weight
        nn.init.orthogonal_(weight)
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.zeros_(bias)


@torch.no_grad()
def update_ema(averaged_generator, generator, decay=EMA_DECAY):
    """Move BigGAN's sampling weights toward the online generator."""
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
        # This compact implementation uses ordinary BN running statistics
        # instead of the paper's cross-replica/standing-statistics pipeline.
        averaged_buffer.copy_(buffer)


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    averaged_generator,
    device,
    discriminator_steps,
    progress_bar=None,
):
    """Train one epoch while preserving BigGAN's exact two-to-one phase."""
    if discriminator_steps < 0:
        raise ValueError("discriminator_steps must be non-negative.")
    discriminator_loss_sum = torch.zeros((), device=device)
    generator_loss_sums = torch.zeros(3, device=device)
    discriminator_examples = 0
    generator_examples = 0

    for real, labels in loader:
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, Z_DIM, device=device)
        with torch.no_grad():
            fake = generator(noise, labels)
        real_scores = discriminator(real, labels)
        fake_scores = discriminator(fake, labels)
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
                loss_g_adversarial = generator_hinge_loss(
                    discriminator(fake, sampled_labels)
                )
                loss_g_regularization = modified_orthogonal_regularization(
                    generator,
                    strength=ORTHOGONAL_REGULARIZATION_STRENGTH,
                )
                loss_g_total = loss_g_adversarial + loss_g_regularization
                loss_g_total.backward()
                opt_g.step()
                update_ema(averaged_generator, generator)
            finally:
                discriminator.requires_grad_(True)

            generator_loss_sums += (
                torch.stack(
                    [
                        loss_g_total,
                        loss_g_adversarial,
                        loss_g_regularization,
                    ]
                ).detach()
                * batch_size
            )
            generator_examples += batch_size

        if progress_bar is not None:
            progress_bar.update(1)

    if generator_examples == 0:
        raise RuntimeError("No generator update occurred in this epoch.")

    generator_losses = (generator_loss_sums / generator_examples).tolist()
    return (
        (discriminator_loss_sum / discriminator_examples).item(),
        *generator_losses,
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
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    generator = CompactBigGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = BigGANDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    generator.apply(initialize_orthogonal_weights)
    discriminator.apply(initialize_orthogonal_weights)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    sn_layer_counts = (
        count_spectral_norm_layers(generator),
        count_spectral_norm_layers(discriminator),
    )
    if sn_layer_counts[0] == 0 or sn_layer_counts[1] == 0:
        raise RuntimeError("BigGAN requires spectral norm in both G and D.")

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=GENERATOR_LR,
        betas=(0.0, 0.999),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=DISCRIMINATOR_LR,
        betas=(0.0, 0.999),
    )

    session = TrainingSession(
        OUT_DIR,
        total_epochs=NUM_EPOCHS,
        models={
            "online_generator": generator,
            "generator": averaged_generator,
            "discriminator": discriminator,
        },
        optimizers={"generator": opt_g, "discriminator": opt_d},
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata={
            "format_version": 5,
            "conditioning": "class_conditional",
            "discriminator_conditioning": "projection",
            "update_ratio": DISCRIMINATOR_UPDATES_PER_GENERATOR,
            "ema_decay": EMA_DECAY,
            "orthogonal_regularization_strength": (
                ORTHOGONAL_REGULARIZATION_STRENGTH
            ),
        },
        model_metadata={
            "online_generator": {
                "model_name": "biggan_online_generator",
                "model_config": MODEL_CONFIG,
                "weights": "online",
            },
            "generator": {
                "model_name": "biggan",
                "model_config": MODEL_CONFIG,
                "weights": "ema",
            },
            "discriminator": {
                "model_name": "biggan_discriminator",
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
        initial_state={
            "loss_history": {
                "epoch": [],
                "discriminator": [],
                "generator_total": [],
                "generator_adversarial": [],
                "generator_regularization": [],
            },
            "fixed_noise": fixed_noise.cpu(),
            "fixed_labels": fixed_labels.cpu(),
            "discriminator_steps": 0,
        },
    )
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_noise = state["fixed_noise"].to(device)
    fixed_labels = state["fixed_labels"].to(device)
    discriminator_steps = state["discriminator_steps"]

    planned_discriminator_updates = NUM_EPOCHS * len(loader)
    planned_generator_updates, incomplete_update_cycle = divmod(
        planned_discriminator_updates,
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
    )
    if incomplete_update_cycle:
        raise RuntimeError(
            "The configured epoch count does not end on an exact 2D:1G "
            "update boundary."
        )
    expected_steps = (start_epoch - 1) * len(loader)
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

            (
                loss_d,
                loss_g_total,
                loss_g_adversarial,
                loss_g_regularization,
                discriminator_steps,
            ) = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                averaged_generator,
                device,
                discriminator_steps,
                progress_bar=progress_bar,
            )
            loss_history["epoch"].append(epoch)
            loss_history["discriminator"].append(loss_d)
            loss_history["generator_total"].append(loss_g_total)
            loss_history["generator_adversarial"].append(
                loss_g_adversarial
            )
            loss_history["generator_regularization"].append(
                loss_g_regularization
            )

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    averaged_generator,
                    fixed_noise,
                    fixed_labels,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    class_names=CIFAR10_CLASS_NAMES[:NUM_CLASSES],
                    title=(
                        "Compact BigGAN EMA fixed class samples - "
                        f"epoch {epoch:03d}"
                    ),
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
            "Generator objective": {
                "G total loss": loss_history["generator_total"],
                "G adversarial loss": loss_history["generator_adversarial"],
            },
            "Modified orthogonal regularization": {
                "G orthogonal penalty": loss_history[
                    "generator_regularization"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the class-conditional CIFAR-10 BigGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
