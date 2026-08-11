"""Train a compact, paper-oriented class-conditional SAGAN on CIFAR-10.

This lesson starts from the ResNet hinge-GAN training in ``6.0_sn_gan.py``
and presents SAGAN's algorithmic additions:
    - class-indexed conditional BatchNorm in G;
    - projection conditioning in D;
    - non-local self-attention with pooled key/value branches and a
      zero-initialized residual gate in G/D;
    - spectral normalization in both G and D;
    - TTUR with one D update per G update.

The original paper trains 128x128 ImageNet models.  Here the same ideas are
kept in a three-block 32x32 CIFAR-10 model with ordinary single-device
BatchNorm.  Random horizontal flips are retained as a small, useful image
augmentation.  Loss curves, fixed class samples, and full checkpoints are
training conveniences and do not alter the adversarial updates.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
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
Optimizer updates:       78,100 D / 78,100 G (1:1)

Generator:                1.14 M params
Discriminator:            1.08 M params
Total:                    2.22 M params
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
from dl_utils.genai.sagan import (
    CIFAR10_CLASS_NAMES,
    SAGANDiscriminator,
    SAGANGenerator,
    make_fixed_class_latent_grid,
)
from dl_utils.genai.sn_gan import (
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sagan"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 120
BASE_CHANNELS = 32
IMAGE_SIZE = 32
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 4e-4
SAMPLES_TO_DISPLAY = NUM_CLASSES * SAMPLES_PER_CLASS
SAMPLE_EVERY_EPOCHS = 5
CHECKPOINT_EVERY_EPOCHS = 5
ARCHIVE_EVERY_EPOCHS = 25
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "image_size": IMAGE_SIZE,
}

DISCRIMINATOR_CONFIG = {
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
}


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    device,
    progress_bar=None,
):
    """Train one epoch with SAGAN's balanced D/G hinge updates."""
    discriminator_loss_sum = torch.zeros((), device=device)
    generator_loss_sum = torch.zeros((), device=device)
    num_examples = 0

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

        # G receives the class signal through conditional BatchNorm and D's
        # projection term.  Freezing D avoids computing unused D gradients.
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

        discriminator_loss_sum += loss_d.detach() * batch_size
        generator_loss_sum += loss_g.detach() * batch_size
        num_examples += batch_size

        if progress_bar is not None:
            progress_bar.update(1)

    return (
        (discriminator_loss_sum / num_examples).item(),
        (generator_loss_sum / num_examples).item(),
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
            "format_version": 6,
            "conditioning": "class_conditional",
            "discriminator_conditioning": "projection",
            "update_ratio": 1,
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
        initial_state={
            "loss_history": {
                "epoch": [],
                "discriminator": [],
                "generator": [],
            },
            "fixed_noise": fixed_noise.cpu(),
            "fixed_labels": fixed_labels.cpu(),
        },
    )
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_noise = state["fixed_noise"].to(device)
    fixed_labels = state["fixed_labels"].to(device)

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

    planned_updates = NUM_EPOCHS * len(loader)
    completed_updates = (start_epoch - 1) * len(loader)
    print(f"Planned optimizer updates: D={planned_updates:,}, G={planned_updates:,}")
    with tqdm(
        total=planned_updates,
        initial=completed_updates,
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

            loss_d, loss_g = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                device,
                progress_bar=progress_bar,
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

            session.checkpoint(epoch, state)

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
