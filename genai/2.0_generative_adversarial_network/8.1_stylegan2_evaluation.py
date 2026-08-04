"""Qualitatively evaluate a trained compact StyleGAN2 generator.

Input:
    output/stylegan2/stylegan2_generator.pth, created by 8.0.

Outputs:
    output/stylegan2/evaluation/, reset at the start of every run.
    - random_samples.png: 64 independently sampled z vectors
    - noise_variations.png: one w with different per-layer stochastic noise
    - style_mixing.png: row coarse styles + column fine styles

In style_mixing.png, the top row and left column are unmixed sources.  Each
inner image takes styles 0..5 from its row and styles 6..11 from its column;
stochastic noise is disabled so the style boundary is easier to inspect.
"""

import torch
from torchvision.utils import save_image

from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.stylegan2 import StyleGenerator, denormalize


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "stylegan2" / "evaluation"
GENERATOR_PATH = OUT_DIR.parent / "stylegan2_generator.pth"


def main():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"StyleGAN2 checkpoint not found: {GENERATOR_PATH}. "
            "Run 8.0_stylegan2.py first."
        )
    reset_dir(str(OUT_DIR))
    device = try_gpu()
    checkpoint = torch.load(
        GENERATOR_PATH, map_location=device, weights_only=True
    )
    generator = StyleGenerator(
        checkpoint["z_dim"],
        checkpoint["style_dim"],
        checkpoint["base_channels"],
        checkpoint.get("mapping_layers", 4),
    ).to(device)
    generator.load_state_dict(checkpoint["model"])
    generator.eval()
    z_dim = checkpoint["z_dim"]

    with torch.inference_mode():
        torch.manual_seed(42)
        samples = generator(torch.randn(64, z_dim, device=device))
        save_image(
            denormalize(samples), OUT_DIR / "random_samples.png", nrow=8
        )

        torch.manual_seed(43)
        shared_z = torch.randn(1, z_dim, device=device).repeat(8, 1)
        variations = generator(shared_z)
        save_image(
            denormalize(variations),
            OUT_DIR / "noise_variations.png",
            nrow=8,
        )

        torch.manual_seed(44)
        row_w = generator.mapping(torch.randn(4, z_dim, device=device))
        column_w = generator.mapping(torch.randn(4, z_dim, device=device))
        row_images = generator.synthesize(
            row_w[:, None, :].repeat(1, generator.num_ws, 1),
            randomize_noise=False,
        )
        column_images = generator.synthesize(
            column_w[:, None, :].repeat(1, generator.num_ws, 1),
            randomize_noise=False,
        )
        cutoff = generator.num_ws // 2
        mixed_ws = torch.stack(
            [
                torch.cat(
                    [
                        row_w[row].repeat(cutoff, 1),
                        column_w[column].repeat(
                            generator.num_ws - cutoff, 1
                        ),
                    ]
                )
                for row in range(4)
                for column in range(4)
            ]
        )
        mixed = generator.synthesize(mixed_ws, randomize_noise=False)
        grid = [torch.full_like(row_images[0], -1), *column_images]
        for row, source in enumerate(row_images):
            grid.append(source)
            grid.extend(mixed[4 * row : 4 * row + 4])
        save_image(
            denormalize(torch.stack(grid)),
            OUT_DIR / "style_mixing.png",
            nrow=5,
        )


if __name__ == "__main__":
    main()
