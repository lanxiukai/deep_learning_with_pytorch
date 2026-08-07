"""Train a compact class-conditional SAGAN on CIFAR-10.

This second lesson keeps the SN-GAN data pipeline, conditioning, losses, and
optimizer settings unchanged. Its only architectural change is self-attention
on the generator and discriminator 16x16 feature maps. The learnable residual
scales start at zero, so training decides how strongly to use non-local cues.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/sagan/training/epoch_*.png: titled fixed class samples
    output/sagan/generator.pth
    output/sagan/loss_curves.png: separate D and G loss panels
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.sagan import SAGANDiscriminator, SAGANGenerator
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
OUT_DIR = PROJECT_ROOT / "output" / "sagan"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 120
BASE_CHANNELS = 32
CONDITION_DIM = 128
LEARNING_RATE = 2e-4
LOSS_UPDATES_PER_EPOCH = 1

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "condition_dim": CONDITION_DIM,
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
    loss_sums = torch.zeros(2, device=device)
    window_loss_sums = [0.0] * 2
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
            loss_g = generator_hinge_loss(
                discriminator(fake, sampled_labels)
            )
            loss_g.backward()
            opt_g.step()
        finally:
            discriminator.requires_grad_(True)

        batch_losses = torch.stack([loss_d, loss_g]).detach()
        loss_sums += batch_losses * batch_size
        num_examples += batch_size
        batch_loss_values = batch_losses.tolist()
        loss_d_value, loss_g_value = batch_loss_values

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
                window_loss_sums = [0.0] * 2
                window_examples = 0

        if progress_bar is None:
            loop.set_postfix(
                loss_D=loss_d_value,
                loss_G=loss_g_value,
                refresh=False,
            )
        else:
            progress_bar.set_postfix(
                loss_D=loss_d_value,
                loss_G=loss_g_value,
                refresh=False,
            )
            progress_bar.update(1)

    return tuple(value / num_examples for value in loss_sums.tolist())


def count_parameters(module):
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in module.parameters())


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

    generator = SAGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = SAGANDiscriminator(
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)
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

    print(
        f"Generator parameters: {count_parameters(generator):,}; "
        f"discriminator parameters: {count_parameters(discriminator):,}."
    )

    fixed_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    fixed_noise = torch.randn(fixed_labels.shape[0], Z_DIM, device=device)
    loss_steps = []
    discriminator_losses = []
    generator_losses = []

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

            def update_loss_curve(progress, window_loss_d, window_loss_g):
                loss_steps.append(epoch - 1 + progress)
                discriminator_losses.append(window_loss_d)
                generator_losses.append(window_loss_g)

            loss_d, loss_g = train_epoch(
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
                title=f"SAGAN fixed class samples - epoch {epoch:03d}",
            )
            progress_bar.write(
                f"epoch {epoch:03d}: D={loss_d:.3f}, G={loss_g:.3f}, "
                f"gamma_G={generator.attention.gamma.item():.3f}, "
                f"gamma_D={discriminator.attention.gamma.item():.3f}"
            )

    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(
        {
            "model_name": "sagan",
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
            "Generator adversarial loss": {
                "G adversarial loss": generator_losses,
            },
        },
        OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
