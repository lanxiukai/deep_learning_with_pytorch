"""Sample a trained compact SAGAN/BigGAN generator by CIFAR-10 class.

The two grids use different truncation thresholds to expose BigGAN's
fidelity/diversity control.  Each row is one CIFAR-10 class.

Input:
    output/sagan/conditional_generator.pth, created by 6.0.

Outputs:
    output/sagan/evaluation/truncation_1.0.png
    output/sagan/evaluation/truncation_0.5.png
"""

import torch
from torchvision.utils import save_image

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.sagan_biggan import (
    ConditionalGenerator,
    denormalize,
    truncated_normal,
)


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "sagan" / "evaluation"
GENERATOR_PATH = OUT_DIR.parent / "conditional_generator.pth"

NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 128
BASE_CHANNELS = 32


def main():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"SAGAN checkpoint not found: {GENERATOR_PATH}. "
            "Run 6.0_sn_sagan_biggan.py first."
        )
    reset_dir(str(OUT_DIR))
    device = try_gpu()

    generator = ConditionalGenerator(
        z_dim=Z_DIM,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)
    generator.load_state_dict(
        torch.load(GENERATOR_PATH, map_location=device, weights_only=True)
    )
    generator.eval()
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )

    with torch.inference_mode():
        for truncation in (1.0, 0.5):
            torch.manual_seed(42)
            noise = truncated_normal(
                (labels.shape[0], Z_DIM), truncation, device
            )
            samples = generator(noise, labels)
            save_image(
                denormalize(samples),
                OUT_DIR / f"truncation_{truncation:.1f}.png",
                nrow=SAMPLES_PER_CLASS,
            )


if __name__ == "__main__":
    main()
