"""Train a compact class-conditional SN-GAN/SAGAN/BigGAN model.

Compared with the earlier GAN lessons, the important additions are:
    - spectral normalization on G and D;
    - self-attention for non-local image structure;
    - shared class embeddings + conditional BatchNorm in G;
    - a projection discriminator and hinge adversarial losses;
    - BigGAN's truncated-normal sampling trade-off.

This is a 64x64 CIFAR-10 teaching model, not a full BigGAN reproduction.  The
complete residual/attention model is in dl_utils/genai/sagan_biggan.py; this
file keeps the new training and sampling behavior visible.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/sagan/training/epoch_*.png: fixed samples, one row per class
    output/sagan/conditional_generator.pth
    output/sagan/loss_curves.png

Training images: 50,000 (49,984 used per epoch with drop_last=True)
Generator: 1.9 M params; discriminator: 1.2 M; total: 3.1 M
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.sagan_biggan import (
    ConditionalGenerator,
    ProjectionDiscriminator,
    denormalize,
    truncated_normal,
)
from dl_utils.plot.figures import Animator
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
Z_DIM = 128
BASE_CHANNELS = 32
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 4e-4  # TTUR: D learns faster than G.
TRUNCATION = 1.0


def train_epoch(generator, discriminator, loader, opt_G, opt_D, device):
    """Train one epoch with the SAGAN/BigGAN hinge objectives."""
    loop = tqdm(loader, leave=True)
    loss_sums = torch.zeros(2, device=device)
    num_examples = 0

    for real, labels in loop:
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = real.shape[0]

        # D hinge loss: real scores >= 1 and fake scores <= -1.
        noise = torch.randn(batch_size, Z_DIM, device=device)
        with torch.no_grad():
            fake = generator(noise, labels)
        real_scores = discriminator(real, labels)
        fake_scores = discriminator(fake, labels)
        loss_D = F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()
        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()

        # G uses fresh classes/noise and maximizes D(G(z, y), y).
        sampled_labels = torch.randint(NUM_CLASSES, (batch_size,), device=device)
        noise = torch.randn(batch_size, Z_DIM, device=device)
        discriminator.requires_grad_(False)
        fake = generator(noise, sampled_labels)
        loss_G = -discriminator(fake, sampled_labels).mean()
        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()
        discriminator.requires_grad_(True)

        loss_sums += torch.stack([loss_D, loss_G]).detach() * batch_size
        num_examples += batch_size
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

    return tuple(value / num_examples for value in loss_sums.tolist())


def save_training_samples(generator, noise, labels, output_path):
    """Use the same truncated noise/classes after every epoch."""
    generator.eval()
    with torch.inference_mode():
        samples = generator(noise, labels)
    generator.train()
    save_image(
        denormalize(samples),
        output_path,
        nrow=SAMPLES_PER_CLASS,
    )


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

    generator = ConditionalGenerator(
        z_dim=Z_DIM,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)
    opt_G = torch.optim.Adam(
        generator.parameters(), lr=GENERATOR_LR, betas=(0.0, 0.999)
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(0.0, 0.999)
    )

    fixed_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    fixed_noise = truncated_normal(
        (fixed_labels.shape[0], Z_DIM), TRUNCATION, device
    )
    animator = Animator(
        xlabel="epoch",
        ylabel="hinge loss",
        xlim=[1, NUM_EPOCHS],
        legend=["discriminator", "generator"],
        figsize=(5, 3.5),
    )

    timer = Timer()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss_D, loss_G = train_epoch(
            generator, discriminator, loader, opt_G, opt_D, device
        )
        animator.add(epoch, (loss_D, loss_G))
        save_training_samples(
            generator,
            fixed_noise,
            fixed_labels,
            TRAINING_DIR / f"epoch_{epoch:03d}.png",
        )
        print(
            f"epoch {epoch:03d}: D={loss_D:.3f}, G={loss_G:.3f}, "
            f"gamma_G={generator.attention.gamma.item():.3f}, "
            f"gamma_D={discriminator.attention.gamma.item():.3f}"
        )

    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(generator.state_dict(), OUT_DIR / "conditional_generator.pth")
    animator.fig.savefig(OUT_DIR / "loss_curves.png", dpi=300)
    plt.close(animator.fig)


if __name__ == "__main__":
    main()
