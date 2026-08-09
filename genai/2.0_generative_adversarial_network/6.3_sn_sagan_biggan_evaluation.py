"""Qualitatively compare SN-GAN, SAGAN, and compact BigGAN samples.

SN-GAN is unconditional, so its row contains ten independent fixed-z samples.
SAGAN and BigGAN are class conditional: their columns hold z fixed while the
CIFAR-10 label changes.  BigGAN is also shown at two truncation thresholds.

The rows therefore compare the lessons' generation interfaces and visible
sample quality; the unconditional row is not a controlled class-response
experiment.  Quantitative diagnostics remain in the individual training
scripts.

Inputs:
    output/sn_gan/generator.pth, created by 6.0_sn_gan.py
    output/sagan/generator.pth, created by 6.1_sagan.py
    output/biggan/generator.pth, created by 6.2_biggan.py

Output:
    output/sn_sagan_biggan_evaluation/sn_sagan_biggan_evaluation.png
"""

import torch

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.biggan import CompactBigGANGenerator, truncated_normal
from dl_utils.genai.sagan import (
    CIFAR10_CLASS_NAMES,
    SAGANGenerator,
    make_fixed_class_latent_grid,
)
from dl_utils.genai.sn_gan import (
    SNGenerator,
    count_spectral_norm_layers,
)
from dl_utils.plot.images import save_image_row_grid


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "sn_sagan_biggan_evaluation"

NUM_CLASSES = 10
IMAGE_SIZE = 32
SEED = 42

DISPLAY_NAMES = {
    "sn_gan": "SN-GAN (unconditional)",
    "sagan": "SAGAN",
    "biggan": "Compact BigGAN",
}

MODEL_SPECS = (
    (
        "sn_gan",
        SNGenerator,
        PROJECT_ROOT / "output" / "sn_gan" / "generator.pth",
        "6.0_sn_gan.py",
    ),
    (
        "sagan",
        SAGANGenerator,
        PROJECT_ROOT / "output" / "sagan" / "generator.pth",
        "6.1_sagan.py",
    ),
    (
        "biggan",
        CompactBigGANGenerator,
        PROJECT_ROOT / "output" / "biggan" / "generator.pth",
        "6.2_biggan.py",
    ),
)


def load_generator(
    model_name,
    model_class,
    checkpoint_path,
    script_name,
    device,
):
    """Load one metadata-rich generator checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{model_name} checkpoint not found: {checkpoint_path}. "
            f"Run {script_name} first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected a metadata-rich checkpoint: {checkpoint_path}."
        )
    if checkpoint.get("model_name") != model_name:
        raise ValueError(
            f"Expected a {model_name} checkpoint at {checkpoint_path}, "
            f"found {checkpoint.get('model_name')!r}."
        )
    if (
        model_name == "sn_gan"
        and checkpoint.get("conditioning") != "unconditional"
    ):
        raise ValueError(
            "The SN-GAN checkpoint predates the unconditional lesson. "
            "Retrain it with 6.0_sn_gan.py."
        )

    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint metadata is incomplete: {checkpoint_path}."
        )
    if model_name == "sagan" and "condition_dim" in model_config:
        raise ValueError(
            "The SAGAN checkpoint uses the former shared SN-GAN conditioning "
            "path. Retrain it with 6.1_sagan.py."
        )
    if model_config.get("image_size") != IMAGE_SIZE:
        raise ValueError(
            f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} checkpoint at "
            f"{checkpoint_path}. Retrain {script_name} with its default "
            "IMAGE_SIZE setting."
        )

    generator = model_class(**model_config).to(device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator, model_config


def main():
    set_seed(SEED)
    device = try_gpu()

    generators = {}
    configurations = {}
    for model_name, model_class, checkpoint_path, script_name in MODEL_SPECS:
        generator, model_config = load_generator(
            model_name,
            model_class,
            checkpoint_path,
            script_name,
            device,
        )
        generators[model_name] = generator
        configurations[model_name] = model_config
        print(
            f"{DISPLAY_NAMES[model_name]} spectral-normalized generator "
            f"layers: {count_spectral_norm_layers(generator)}"
        )

    conditional_dimensions = {
        configurations[name]["z_dim"] for name in ("sagan", "biggan")
    }
    if len(conditional_dimensions) != 1:
        raise ValueError(
            "SAGAN and BigGAN must use the same latent dimension for the "
            "fixed-z class comparison."
        )
    conditional_z_dim = conditional_dimensions.pop()
    class_counts = {
        configurations[name]["num_classes"]
        for name in ("sagan", "biggan")
    }
    if class_counts != {NUM_CLASSES}:
        raise ValueError(
            f"Conditional checkpoints must use {NUM_CLASSES} classes, "
            f"found {sorted(class_counts)}."
        )

    reset_dir(str(OUT_DIR))
    torch.manual_seed(SEED)
    sn_noise = torch.randn(
        NUM_CLASSES,
        configurations["sn_gan"]["z_dim"],
        device=device,
    )
    conditional_noise, labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        1,
        conditional_z_dim,
        device,
    )

    with torch.inference_mode():
        image_rows = [generators["sn_gan"](sn_noise).cpu()]
        row_labels = [DISPLAY_NAMES["sn_gan"]]

        for model_name in ("sagan", "biggan"):
            samples = generators[model_name](
                conditional_noise,
                labels,
            ).cpu()
            image_rows.append(samples)
            row_labels.append(DISPLAY_NAMES[model_name])

        biggan = generators["biggan"]
        for truncation in (1.0, 0.5):
            torch.manual_seed(SEED)
            base_noise = truncated_normal(
                (1, conditional_z_dim),
                truncation,
                device,
            )
            noise, _ = make_fixed_class_latent_grid(
                NUM_CLASSES,
                1,
                conditional_z_dim,
                device,
                base_noise=base_noise,
            )
            image_rows.append(biggan(noise, labels).cpu())
            row_labels.append(f"BigGAN truncation {truncation:.1f}")

    save_image_row_grid(
        image_rows,
        row_labels,
        OUT_DIR / "sn_sagan_biggan_evaluation.png",
        title=(
            "Unconditional SN-GAN samples and conditional fixed-z "
            "CIFAR-10 response"
        ),
        column_labels=[
            name.title() for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
        ],
    )


if __name__ == "__main__":
    main()
