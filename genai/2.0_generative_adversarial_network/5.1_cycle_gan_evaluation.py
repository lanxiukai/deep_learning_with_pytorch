"""Exercise 6.1: evaluate trained CycleGAN hair-color checkpoints.

Requires output/cyclegan/gen_black.pth and gen_blond.pth; run
5.0_cycle_gan.py first. Evaluation images are written to output/cyclegan/.
"""

import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.cyclegan import Generator, LoadData


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "cyclegan"

BLACK_DIR = DATA_DIR / "celeba" / "black"
BLOND_DIR = DATA_DIR / "celeba" / "blond"
GEN_BLACK_PATH = OUT_DIR / "gen_black.pth"
GEN_BLOND_PATH = OUT_DIR / "gen_blond.pth"


def main():
    """Translate ten black/blond image pairs and reconstruct each input."""
    missing_data_dirs = [
        path for path in (BLACK_DIR, BLOND_DIR) if not path.is_dir()
    ]
    if missing_data_dirs:
        missing = ", ".join(str(path) for path in missing_data_dirs)
        raise FileNotFoundError(
            f"CycleGAN evaluation data not found: {missing}. "
            "Run: python tool_scripts/download_dataset_test.py"
        )

    missing_checkpoints = [
        path
        for path in (GEN_BLACK_PATH, GEN_BLOND_PATH)
        if not path.is_file()
    ]
    if missing_checkpoints:
        missing = ", ".join(str(path) for path in missing_checkpoints)
        raise FileNotFoundError(
            f"CycleGAN checkpoints not found: {missing}. "
            "Run 5.0_cycle_gan.py first."
        )

    transforms = albumentations.Compose(
        [
            albumentations.Resize(width=256, height=256),
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
        root_A=[str(BLACK_DIR)],
        root_B=[str(BLOND_DIR)],
        transform=transforms,
    )

    device = try_gpu()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    gen_black = Generator(img_channels=3, num_residuals=9).to(device)
    gen_blond = Generator(img_channels=3, num_residuals=9).to(device)
    gen_black.load_state_dict(torch.load(GEN_BLACK_PATH, map_location=device))
    gen_blond.load_state_dict(torch.load(GEN_BLOND_PATH, map_location=device))
    gen_black.eval()
    gen_blond.eval()

    # Answer to exercise 6.1: translate in both directions and reconstruct.
    with torch.inference_mode():
        for index, (black, blond) in enumerate(loader, start=1):
            black = black.to(device)
            blond = blond.to(device)

            fake_blond = gen_blond(black)
            fake2black = gen_black(fake_blond)
            fake_black = gen_black(blond)
            fake2blond = gen_blond(fake_black)

            save_image(black * 0.5 + 0.5, OUT_DIR / f"black{index}.png")
            save_image(
                fake_blond * 0.5 + 0.5,
                OUT_DIR / f"fakeblond{index}.png",
            )
            save_image(
                fake2black * 0.5 + 0.5,
                OUT_DIR / f"fake2black{index}.png",
            )
            save_image(blond * 0.5 + 0.5, OUT_DIR / f"blond{index}.png")
            save_image(
                fake_black * 0.5 + 0.5,
                OUT_DIR / f"fakeblack{index}.png",
            )
            save_image(
                fake2blond * 0.5 + 0.5,
                OUT_DIR / f"fake2blond{index}.png",
            )

            if index == 10:
                break


if __name__ == "__main__":
    main()
