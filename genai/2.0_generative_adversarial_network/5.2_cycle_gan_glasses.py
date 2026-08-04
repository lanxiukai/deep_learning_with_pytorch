"""Train a CycleGAN for unpaired glasses <-> no-glasses translation.

This exercise reuses the CycleGAN models and training helpers from
dl_utils/genai/cyclegan.py for a second image-translation task.

Data:
    data/glasses/{G,NoG}, prepared by
    tool_scripts/download_dataset_test.py.

Outputs:
    output/cyclegan/glasses_training/, reset at the start of every run. The
    directory contains titled 2x2 training grids (real with-glasses -> generated
    without-glasses on top and the reverse direction on the bottom).
    - output/cyclegan/glasses_loss_curves.png: discriminator and generator loss
      curves
    - output/cyclegan/add_glasses.pth: no glasses -> glasses generator
    - output/cyclegan/remove_glasses.pth: glasses -> no glasses generator

Training data — glasses task:
With glasses (G):         2,543 images
Without glasses (NoG):    1,957 images
Available total:          4,500 unique images
Samples per epoch:        2,543 unpaired image pairs
Note: Counts include 517 repository-tracked label corrections. The smaller NoG
domain cycles, so 586 of its images are reused once per epoch.

Generator (each):      11.4 M params
Discriminator (each):   2.8 M params
Total (2 of each):     28.3 M params
"""

import albumentations
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
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
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "cyclegan" / "glasses_training"
CYCLEGAN_OUT_DIR = OUT_DIR.parent
GLASSES_DIR = DATA_DIR / "glasses"
WITH_GLASSES_DIR = GLASSES_DIR / "G"
WITHOUT_GLASSES_DIR = GLASSES_DIR / "NoG"


def main():
    """Train both translation directions for the glasses task."""
    reset_dir(str(OUT_DIR))

    missing_data_dirs = [
        path
        for path in (WITH_GLASSES_DIR, WITHOUT_GLASSES_DIR)
        if not path.is_dir()
    ]
    if missing_data_dirs:
        missing = ", ".join(str(path) for path in missing_data_dirs)
        raise FileNotFoundError(
            f"CycleGAN glasses data not found: {missing}. "
            "Run: python tool_scripts/download_dataset_test.py"
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
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )
    dataset = LoadData(
        root_A=[str(WITH_GLASSES_DIR)],
        root_B=[str(WITHOUT_GLASSES_DIR)],
        transform=transforms,
    )

    device = try_gpu()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    disc_with_glasses = Discriminator().to(device)
    disc_without_glasses = Discriminator().to(device)
    gen_add_glasses = Generator(img_channels=3, num_residuals=9).to(device)
    gen_remove_glasses = Generator(img_channels=3, num_residuals=9).to(device)
    weights_init(disc_with_glasses)
    weights_init(disc_without_glasses)
    weights_init(gen_add_glasses)
    weights_init(gen_remove_glasses)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    # Mixed-precision (AMP) gradient scalers prevent FP16 gradient underflow.
    # One scaler is used per optimizer because their update patterns differ.
    gen_scaler = torch.amp.GradScaler(device.type)
    disc_scaler = torch.amp.GradScaler(device.type)

    lr = 0.00001
    opt_disc = torch.optim.Adam(
        list(disc_with_glasses.parameters())
        + list(disc_without_glasses.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )
    opt_gen = torch.optim.Adam(
        list(gen_add_glasses.parameters())
        + list(gen_remove_glasses.parameters()),
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
            disc_with_glasses,
            disc_without_glasses,
            gen_add_glasses,
            gen_remove_glasses,
            loader,
            opt_disc,
            opt_gen,
            l1,
            mse,
            disc_scaler,
            gen_scaler,
            device,
            OUT_DIR,
            epoch=epoch,
            snapshot_updates_per_epoch=10,
            domain_A_label="With glasses",
            domain_B_label="Without glasses",
        )
        animator.add(epoch, (loss_D, loss_G))

    timing = format_epoch_timing(timer.stop(), num_epochs)
    print(
        f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, "
        f"{timing} on {device}"
    )

    torch.save(
        gen_add_glasses.state_dict(),
        CYCLEGAN_OUT_DIR / "add_glasses.pth",
    )
    torch.save(
        gen_remove_glasses.state_dict(),
        CYCLEGAN_OUT_DIR / "remove_glasses.pth",
    )
    animator.fig.savefig(
        CYCLEGAN_OUT_DIR / "glasses_loss_curves.png",
        dpi=300,
    )
    plt.close(animator.fig)


if __name__ == "__main__":
    main()
