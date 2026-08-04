"""Exercise 6.2: train CycleGAN to add or remove glasses.

Data: local data/glasses/{G,NoG}, prepared by
tool_scripts/download_dataset_test.py.
Outputs: output/cyclegan/add_glasses.pth and remove_glasses.pth.

Training data:
With glasses (G):         2,543 images
Without glasses (NoG):    1,957 images
Available total:          4,500 unique images
Samples per epoch:        2,543 unpaired image pairs
Note: Counts include 517 repository-tracked label corrections. The smaller NoG
domain cycles, so 586 of its images are reused once per epoch.
"""

import albumentations
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.cyclegan import (
    Discriminator,
    Generator,
    LoadData,
    train_epoch,
    weights_init,
)


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "cyclegan"
GLASSES_DIR = DATA_DIR / "glasses"
WITH_GLASSES_DIR = GLASSES_DIR / "G"
WITHOUT_GLASSES_DIR = GLASSES_DIR / "NoG"


def main():
    """Train both translation directions for the glasses task."""
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    for _ in range(1):
        train_epoch(
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
        )

    torch.save(gen_add_glasses.state_dict(), OUT_DIR / "add_glasses.pth")
    torch.save(
        gen_remove_glasses.state_dict(),
        OUT_DIR / "remove_glasses.pth",
    )


if __name__ == "__main__":
    main()
