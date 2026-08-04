"""Evaluate pix2pix on ten paired CelebA test images.

This mirrors 5.1_cycle_gan_evaluation.py, but every generated image has an
aligned target that can be inspected directly.

Input:
    output/pix2pix/pix2pix_generator.pth, created by 6.0_pix2pix.py.

Outputs:
    output/pix2pix/evaluation/, reset at the start of every run.
    - input{n}.png: grayscale source
    - generated{n}.png: predicted color image
    - target{n}.png: paired RGB target
"""

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.pix2pix import (
    CelebAColorizationDataset,
    UNetGenerator,
    build_paired_transform,
    denormalize,
)


PROJECT_ROOT = infer_project_root()
CELEBA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "pix2pix" / "evaluation"
GENERATOR_PATH = OUT_DIR.parent / "pix2pix_generator.pth"


def main():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"pix2pix checkpoint not found: {GENERATOR_PATH}. "
            "Run 6.0_pix2pix.py first."
        )
    reset_dir(str(OUT_DIR))

    dataset = CelebAColorizationDataset(
        CELEBA_DIR,
        "test",
        build_paired_transform(training=False),
    )
    device = try_gpu()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    generator = UNetGenerator().to(device)
    generator.load_state_dict(
        torch.load(GENERATOR_PATH, map_location=device, weights_only=True)
    )
    generator.eval()

    with torch.inference_mode():
        for index, (source, target) in enumerate(loader, start=1):
            source = source.to(device)
            generated = generator(source)

            save_image(
                denormalize(source),
                OUT_DIR / f"input{index}.png",
            )
            save_image(
                denormalize(generated),
                OUT_DIR / f"generated{index}.png",
            )
            save_image(
                denormalize(target),
                OUT_DIR / f"target{index}.png",
            )
            if index == 10:
                break


if __name__ == "__main__":
    main()
