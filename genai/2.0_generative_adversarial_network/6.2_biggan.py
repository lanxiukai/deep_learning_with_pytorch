"""Train a compact class-conditional BigGAN-style model on CIFAR-10.

This third lesson keeps the SAGAN discriminator and adds the main generator
ideas that fit a small teaching model:
    - one class embedding shared across all generator blocks;
    - conditional BatchNorm in every generator residual block;
    - hierarchical latent chunks delivered to successive blocks;
    - modified orthogonal regularization;
    - two-time-scale learning rates.

The truncation trick is intentionally reserved for evaluation in 6.3. This is
a compact BigGAN-style model, not a reproduction of large-scale ImageNet
training.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/biggan/training/epoch_*.png: titled fixed class samples
    output/biggan/generator.pth
    output/biggan/loss_curves.png: separate total and component panels

Training data — CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)

Generator:                0.98 M params
Discriminator:            1.23 M params
Total:                    2.21 M params
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.biggan import (
    BigGANDiscriminator,
    CompactBigGANGenerator,
    modified_orthogonal_regularization,
)
from dl_utils.genai.sn_gan import (
    CIFAR10_CLASS_NAMES,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "biggan"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 120
BASE_CHANNELS = 32
CLASS_EMBEDDING_DIM = 128
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 4e-4
ORTHOGONAL_REGULARIZATION_STRENGTH = 1e-4
LOSS_UPDATES_PER_EPOCH = 1

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "class_embedding_dim": CLASS_EMBEDDING_DIM,
}


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    device,
    loss_callback=None,
    loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
    progress_bar=None,
):
    """Train one epoch and optionally report windowed losses and progress."""
    if loss_updates_per_epoch <= 0:
        raise ValueError("loss_updates_per_epoch must be positive.")

    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    loss_sums = torch.zeros(4, device=device)
    window_loss_sums = [0.0] * 4
    num_examples = 0
    window_examples = 0
    num_batches = len(loader)
    update_interval = max(
        1,
        (num_batches + loss_updates_per_epoch - 1)
        // loss_updates_per_epoch,
    )

    for batch_index, (real, labels) in enumerate(loop, start=1):
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

        sampled_labels = torch.randint(
            NUM_CLASSES, (batch_size,), device=device
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
        finally:
            discriminator.requires_grad_(True)

        batch_losses = torch.stack(
            [
                loss_d,
                loss_g_total,
                loss_g_adversarial,
                loss_g_regularization,
            ]
        ).detach()
        loss_sums += batch_losses * batch_size
        num_examples += batch_size
        batch_loss_values = batch_losses.tolist()
        (
            loss_d_value,
            loss_g_value,
            _,
            loss_g_regularization_value,
        ) = batch_loss_values

        if loss_callback is not None:
            for index, loss_value in enumerate(batch_loss_values):
                window_loss_sums[index] += loss_value * batch_size
            window_examples += batch_size
            if (
                batch_index % update_interval == 0
                or batch_index == num_batches
            ):
                window_losses = [
                    loss_sum / window_examples
                    for loss_sum in window_loss_sums
                ]
                loss_callback(batch_index / num_batches, *window_losses)
                window_loss_sums = [0.0] * 4
                window_examples = 0

        postfix = {
            "loss_D": loss_d_value,
            "loss_G": loss_g_value,
            "ortho": loss_g_regularization_value,
        }
        if progress_bar is None:
            loop.set_postfix(**postfix, refresh=False)
        else:
            progress_bar.set_postfix(**postfix, refresh=False)
            progress_bar.update(1)

    return tuple(value / num_examples for value in loss_sums.tolist())


def main():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    reset_dir(str(TRAINING_DIR))
    set_seed(42)
    device = try_gpu()

    dataset = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(64),
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
    discriminator = BigGANDiscriminator(
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)
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

    fixed_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    fixed_noise = torch.randn(fixed_labels.shape[0], Z_DIM, device=device)
    loss_steps = []
    discriminator_losses = []
    generator_losses = []
    generator_adversarial_losses = []
    generator_regularization_losses = []

    timer = Timer()
    with tqdm(
        total=NUM_EPOCHS * len(loader),
        desc=f"Epoch 1/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, NUM_EPOCHS + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{NUM_EPOCHS}",
                refresh=False,
            )

            def update_loss_curve(
                progress,
                window_loss_d,
                window_loss_g,
                window_loss_g_adversarial,
                window_loss_g_regularization,
            ):
                loss_steps.append(epoch - 1 + progress)
                discriminator_losses.append(window_loss_d)
                generator_losses.append(window_loss_g)
                generator_adversarial_losses.append(
                    window_loss_g_adversarial
                )
                generator_regularization_losses.append(
                    window_loss_g_regularization
                )

            (
                loss_d,
                loss_g,
                loss_g_adversarial,
                loss_g_regularization,
            ) = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                device,
                loss_callback=update_loss_curve,
                loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
                progress_bar=progress_bar,
            )
            save_training_samples(
                generator,
                fixed_noise,
                fixed_labels,
                TRAINING_DIR / f"epoch_{epoch:03d}.png",
                class_names=CIFAR10_CLASS_NAMES[:NUM_CLASSES],
                title=(
                    "Compact BigGAN fixed class samples - "
                    f"epoch {epoch:03d}"
                ),
            )
            progress_bar.write(
                f"epoch {epoch:03d}: D={loss_d:.3f}, G={loss_g:.3f}, "
                f"GAN={loss_g_adversarial:.3f}, "
                f"ortho={loss_g_regularization:.5f}, "
                f"gamma_G={generator.attention.gamma.item():.3f}, "
                f"gamma_D={discriminator.attention.gamma.item():.3f}"
            )

    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(
        {
            "model_name": "biggan",
            "model_config": MODEL_CONFIG,
            "state_dict": generator.state_dict(),
        },
        OUT_DIR / "generator.pth",
    )
    save_loss_panels(
        loss_steps,
        {
            "Discriminator hinge loss": {
                "D hinge loss": discriminator_losses,
            },
            "Total generator loss": {
                "G total loss": generator_losses,
            },
            "Generator adversarial loss": {
                "G adversarial loss": generator_adversarial_losses,
            },
            "Generator orthogonal regularization": {
                "G orthogonal penalty": generator_regularization_losses,
            },
        },
        OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
