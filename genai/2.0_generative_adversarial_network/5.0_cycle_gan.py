"""Train a CycleGAN for unpaired black-hair <-> blond-hair translation.

Adapted from the companion code for "Generative Deep Learning", 2nd ed.
(book_repos/DGAI/ch06CycleGAN.ipynb). Models and shared training helpers live
in dl_utils/genai/cyclegan.py.

Data:
    data/celeba/{black,blond}, prepared by
    tool_scripts/download_dataset_test.py.

Outputs:
    output/cyclegan/training/, reset at the start of every run. The directory
    contains training snapshots, the live loss curve, and these checkpoints:
    - gen_black.pth: blond -> black generator
    - gen_blond.pth: black -> blond generator

Related scripts:
    - 5.1_cycle_gan_evaluation.py evaluates the saved hair-color generators.
    - 5.2_cycle_gan_glasses.py trains the same architecture for glasses.

Training data — hair-color task:
Black hair:              48,472 images
Blond hair:              29,980 images
Available total:         78,452 unique images
Samples per epoch:       48,472 unpaired image pairs
Note: LoadData uses the larger domain as its length, so the smaller blond-hair
domain cycles and 18,492 of its images are reused once per epoch.

Generator (each):      11.4 M params
Discriminator (each):   2.8 M params
Total (2 of each):     28.3 M params
"""

import random

import albumentations
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.cyclegan import (
    Discriminator,
    Generator,
    LoadData,
    train_epoch,
    weights_init,
)
from dl_utils.plot.figures import Animator
from dl_utils.training.timing import Timer

# installation
# !pip install pandas albumentations


# Data lives under <project_root>/data; generated images and checkpoints
# are written under <project_root>/output/cyclegan/training
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
OUT_DIR = PROJECT_ROOT / 'output' / 'cyclegan' / 'training'


CELEBA_DIR = DATA_DIR / "celeba"
BLACK_DIR = CELEBA_DIR / "black"
BLOND_DIR = CELEBA_DIR / "blond"


def save_dataset_samples(image_dirs, output_path, samples_per_dir=8, seed=42):
    """Save a reproducible sample grid from each image directory."""
    rng = random.Random(seed)
    sampled_paths = []
    for image_dir in image_dirs:
        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
        if len(image_paths) < samples_per_dir:
            raise ValueError(
                f"{image_dir} contains {len(image_paths)} images; "
                f"at least {samples_per_dir} are required."
            )
        sampled_paths.append(rng.sample(image_paths, samples_per_dir))

    fig, axes = plt.subplots(
        len(image_dirs),
        samples_per_dir,
        dpi=100,
        figsize=(1.78 * samples_per_dir, 2.18 * len(image_dirs)),
        squeeze=False,
    )
    for row, paths in enumerate(sampled_paths):
        for column, image_path in enumerate(paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.subplots_adjust(wspace=-0.01, hspace=-0.1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Train the hair-color CycleGAN and save its artifacts."""
    reset_dir(str(OUT_DIR))

    if not BLACK_DIR.is_dir() or not BLOND_DIR.is_dir():
        raise FileNotFoundError(
            "CycleGAN requires the local CelebA splits at data/celeba/black "
            "and data/celeba/blond. Run: "
            "python tool_scripts/download_dataset_test.py"
        )

    save_dataset_samples(
        [BLACK_DIR, BLOND_DIR],
        OUT_DIR / "celeba_hair_samples.png",
    )

    transforms = albumentations.Compose(
        [
            albumentations.Resize(width=256, height=256),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=255,
            ),
            ToTensorV2(),  # -> PyTorch Tensor: (C, H, W)
        ],
        additional_targets={"image0": "image"},
    )  # Apply the same transforms to image0.
    dataset = LoadData(
        root_A=[str(BLACK_DIR)],
        root_B=[str(BLOND_DIR)],
        transform=transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )

    device = try_gpu()
    disc_A = Discriminator().to(device)
    disc_B = Discriminator().to(device)
    weights_init(disc_A)
    weights_init(disc_B)

    gen_A = Generator(img_channels=3, num_residuals=9).to(device)
    gen_B = Generator(img_channels=3, num_residuals=9).to(device)
    weights_init(gen_A)
    weights_init(gen_B)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    # Mixed-precision (AMP) gradient scalers prevent FP16 gradient underflow.
    # One scaler is used per optimizer because their update patterns differ.
    g_scaler = torch.amp.GradScaler(device.type)
    d_scaler = torch.amp.GradScaler(device.type)

    lr = 0.00001
    opt_disc = torch.optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )
    opt_gen = torch.optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )

    num_epochs = 1
    animator = Animator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, num_epochs],
        legend=["discriminator", "generator"],
        figsize=(5, 3.5),
    )
    timer = Timer()
    for epoch in range(1, num_epochs + 1):
        loss_D, loss_G = train_epoch(
            disc_A,
            disc_B,
            gen_A,
            gen_B,
            loader,
            opt_disc,
            opt_gen,
            l1,
            mse,
            d_scaler,
            g_scaler,
            device,
            OUT_DIR,
        )
        animator.add(epoch, (loss_D, loss_G))

    total_time = timer.stop()
    total_tenths = round(total_time * 10)
    hours, remainder = divmod(total_tenths, 36000)
    minutes, second_tenths = divmod(remainder, 600)
    seconds = second_tenths / 10
    avg_tenths = round(total_time / num_epochs * 10)
    avg_minutes, avg_second_tenths = divmod(avg_tenths, 600)
    avg_seconds = avg_second_tenths / 10
    print(
        f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, "
        f"training time {hours} h {minutes} min {seconds:.1f} s, "
        f"avg {avg_minutes} min {avg_seconds:.1f} s/epoch on {device}"
    )

    torch.save(gen_A.state_dict(), OUT_DIR / "gen_black.pth")
    torch.save(gen_B.state_dict(), OUT_DIR / "gen_blond.pth")
    animator.fig.savefig(OUT_DIR / "loss_curves.png", dpi=300)
    plt.close(animator.fig)


if __name__ == "__main__":
    main()
