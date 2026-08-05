"""Train a CycleGAN for unpaired black-hair <-> blond-hair translation.

Adapted from the companion code for "Generative Deep Learning", 2nd ed.
(book_repos/DGAI/ch06CycleGAN.ipynb). Models and shared training helpers live
in dl_utils/genai/cyclegan.py.

Data:
    data/celeba/{black,blond}, prepared by
    tool_scripts/download_dataset_test.py.

Outputs:
    output/cyclegan/training/, reset at the start of every run. The directory
    contains titled 2x2 training grids (real black hair -> generated blond on
    top and real blond hair -> generated black on the bottom).
    - output/cyclegan/loss_curves.png: total D/G, directional adversarial, and
      10 × directional cycle-consistency losses in one four-panel figure
    - output/cyclegan/gen_black.pth: blond -> black generator
    - output/cyclegan/gen_blond.pth: black -> blond generator

Related scripts:
    - 5.1_cycle_gan_evaluation.py evaluates the saved hair-color generators.

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
from tqdm import tqdm

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.cyclegan import (
    Discriminator,
    Generator,
    LAMBDA_CYCLE,
    LoadData,
    train_epoch,
    weights_init,
)
from dl_utils.plot.figures import save_loss_curves
from dl_utils.training.timing import Timer, format_epoch_timing

# installation
# !pip install pandas albumentations


# Data lives under <project_root>/data; generated images and checkpoints
# are written under <project_root>/output/cyclegan/training
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
OUT_DIR = PROJECT_ROOT / 'output' / 'cyclegan' / 'training'
CYCLEGAN_OUT_DIR = OUT_DIR.parent


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
            ToTensorV2(),  # numpy H×W×C → torch C×H×W
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

    num_epochs = 3
    loss_steps = []
    discriminator_losses = []
    generator_losses = []
    generator_adversarial_losses = {
        "Blond → Black": [],
        "Black → Blond": [],
    }
    generator_reconstruction_losses = {
        f"Black → Blond → Black ({LAMBDA_CYCLE} × cycle)": [],
        f"Blond → Black → Blond ({LAMBDA_CYCLE} × cycle)": [],
    }
    timer = Timer()
    with tqdm(
        total=num_epochs * len(loader),
        desc=f"Epoch 1/{num_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{num_epochs}",
                refresh=False,
            )

            def update_loss_curve(
                progress,
                window_loss_D,
                window_loss_G,
                window_loss_G_A,
                window_loss_G_B,
                window_cycle_A_loss,
                window_cycle_B_loss,
            ):
                loss_steps.append(epoch - 1 + progress)
                discriminator_losses.append(window_loss_D)
                generator_losses.append(window_loss_G)
                generator_adversarial_losses["Blond → Black"].append(
                    window_loss_G_A
                )
                generator_adversarial_losses["Black → Blond"].append(
                    window_loss_G_B
                )
                generator_reconstruction_losses[
                    f"Black → Blond → Black ({LAMBDA_CYCLE} × cycle)"
                ].append(window_cycle_A_loss)
                generator_reconstruction_losses[
                    f"Blond → Black → Blond ({LAMBDA_CYCLE} × cycle)"
                ].append(window_cycle_B_loss)

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
                epoch=epoch,
                loss_callback=update_loss_curve,
                loss_updates_per_epoch=20,
                snapshot_updates_per_epoch=10,
                domain_A_label="Black hair",
                domain_B_label="Blond hair",
                progress_bar=progress_bar,
            )

    timing = format_epoch_timing(timer.stop(), num_epochs)
    print(
        f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, "
        f"{timing} on {device}"
    )

    torch.save(gen_A.state_dict(), CYCLEGAN_OUT_DIR / "gen_black.pth")
    torch.save(gen_B.state_dict(), CYCLEGAN_OUT_DIR / "gen_blond.pth")
    save_loss_curves(
        loss_steps,
        discriminator_losses,
        generator_losses,
        generator_adversarial_losses,
        generator_reconstruction_losses,
        CYCLEGAN_OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
