"""Compare trained SN-GAN, SAGAN, and compact BigGAN generators.

All three models receive the same class labels and one standard-normal latent
vector repeated across the class columns, so each row isolates the response to
changing ``y`` at fixed ``z``. BigGAN is additionally sampled with two
truncation thresholds; each truncation row likewise repeats one latent vector
across classes.

Inputs:
    output/sn_gan/generator.pth, created by 6.0_sn_gan.py
    output/sagan/generator.pth, created by 6.1_sagan.py
    output/biggan/generator.pth, created by 6.2_biggan.py

Outputs:
    output/sn_sagan_biggan_evaluation/sn_sagan_biggan_evaluation.png:
    one titled grid containing all model and truncation rows
"""

import torch

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.biggan import CompactBigGANGenerator, truncated_normal
from dl_utils.genai.sagan import SAGANGenerator
from dl_utils.genai.sn_gan import (
    CIFAR10_CLASS_NAMES,
    SNGenerator,
    count_spectral_norm_layers,
    make_fixed_class_latent_grid,
)
from dl_utils.plot.images import save_image_row_grid


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "sn_sagan_biggan_evaluation"

NUM_CLASSES = 10
IMAGE_SIZE = 32
SEED = 42

DISPLAY_NAMES = {
    "sn_gan": "SN-GAN",
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


def load_generator(model_name, model_class, checkpoint_path, script_name, device):
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

    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint metadata is incomplete: {checkpoint_path}."
        )
    if model_config.get("image_size") != IMAGE_SIZE:
        raise ValueError(
            f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} checkpoint at "
            f"{checkpoint_path}. Retrain {script_name} with its default "
            f"IMAGE_SIZE setting."
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

    latent_dimensions = {
        model_config["z_dim"] for model_config in configurations.values()
    }
    if len(latent_dimensions) != 1:
        raise ValueError(
            "The three checkpoints must use the same latent dimension for "
            "a controlled comparison."
        )
    z_dim = latent_dimensions.pop()

    class_counts = {
        model_config["num_classes"]
        for model_config in configurations.values()
    }
    if class_counts != {NUM_CLASSES}:
        raise ValueError(
            f"All checkpoints must use {NUM_CLASSES} classes, "
            f"found {sorted(class_counts)}."
        )

    reset_dir(str(OUT_DIR))

    torch.manual_seed(SEED)
    fixed_noise, labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        1,
        z_dim,
        device,
    )
    image_rows = []
    row_labels = []

    with torch.inference_mode():
        for model_name, _, _, _ in MODEL_SPECS:
            samples = generators[model_name](fixed_noise, labels).cpu()
            image_rows.append(samples)
            row_labels.append(DISPLAY_NAMES[model_name])

        biggan = generators["biggan"]
        for truncation in (1.0, 0.5):
            torch.manual_seed(SEED)
            base_noise = truncated_normal(
                (1, z_dim),
                truncation,
                device,
            )
            noise, _ = make_fixed_class_latent_grid(
                NUM_CLASSES,
                1,
                z_dim,
                device,
                base_noise=base_noise,
            )
            image_rows.append(biggan(noise, labels).cpu())
            row_labels.append(f"BigGAN truncation {truncation:.1f}")

    save_image_row_grid(
        image_rows,
        row_labels,
        OUT_DIR / "sn_sagan_biggan_evaluation.png",
        title="CIFAR-10 conditional response at fixed per-row z",
        column_labels=[
            name.title() for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
        ],
    )


if __name__ == "__main__":
    main()
