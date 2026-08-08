"""Compare trained SN-GAN, SAGAN, and compact BigGAN generators.

All three models receive the same class labels and standard-normal latent
vectors. SN-GAN produces native 32x32 images; SAGAN and compact BigGAN produce
64x64 images. BigGAN is additionally sampled with two truncation thresholds
to show the fidelity-diversity control introduced by truncated sampling.

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
from dl_utils.genai.sn_gan import CIFAR10_CLASS_NAMES, SNGenerator
from dl_utils.plot.images import save_image_row_grid


PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "sn_sagan_biggan_evaluation"

NUM_CLASSES = 10
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

    labels = torch.arange(NUM_CLASSES, device=device)
    torch.manual_seed(SEED)
    fixed_noise = torch.randn(NUM_CLASSES, z_dim, device=device)
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
            noise = truncated_normal(
                (NUM_CLASSES, z_dim),
                truncation,
                device,
            )
            image_rows.append(biggan(noise, labels).cpu())
            row_labels.append(f"BigGAN truncation {truncation:.1f}")

    save_image_row_grid(
        image_rows,
        row_labels,
        OUT_DIR / "sn_sagan_biggan_evaluation.png",
        title="CIFAR-10 conditional generation - SN-GAN to Compact BigGAN",
        column_labels=[
            name.title() for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
        ],
    )


if __name__ == "__main__":
    main()
