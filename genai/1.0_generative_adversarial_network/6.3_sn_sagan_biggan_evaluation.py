"""Qualitatively evaluate the SN-GAN to BigGAN lesson sequence.

The figures keep unlike comparisons separate:
    - SN-GAN receives independent z samples and no class labels;
    - SAGAN and BigGAN receive the same z while the CIFAR-10 label changes;
    - BigGAN repeats the class sweep with progressively tighter truncation.

This is a compact mechanism check rather than an FID reproduction.  Models are
trained independently, so shared inputs fix sampling conditions but do not
create semantic correspondence between generators.

Inputs:
    output/sn_gan/generator.pth, created by 6.0_sn_gan.py
    output/sagan/generator.pth, created by 6.1_sagan.py
    output/biggan/generator.pth, created by 6.2_biggan.py

Outputs:
    Each evaluation directory is reset at the start of every run.
    output/sn_gan/evaluation/sn_gan_samples.png
    output/sagan/evaluation/conditional_class_sweep.png
    output/biggan/evaluation/conditional_class_sweep.png
    output/biggan/evaluation/biggan_truncation.png
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
MODEL_OUT_DIRS = {
    model_name: PROJECT_ROOT / "output" / model_name
    for model_name in ("sn_gan", "sagan", "biggan")
}
EVALUATION_DIRS = {
    model_name: output_dir / "evaluation"
    for model_name, output_dir in MODEL_OUT_DIRS.items()
}

NUM_CLASSES = 10
NUM_UNCONDITIONAL_SAMPLES = 10
TRUNCATION_THRESHOLDS = (None, 1.0, 0.5)
SEED = 42

MODEL_SPECS = (
    (
        "sn_gan",
        "SN-GAN",
        SNGenerator,
        MODEL_OUT_DIRS["sn_gan"] / "generator.pth",
        "6.0_sn_gan.py",
        5,
        "unconditional",
        None,
    ),
    (
        "sagan",
        "SAGAN",
        SAGANGenerator,
        MODEL_OUT_DIRS["sagan"] / "generator.pth",
        "6.1_sagan.py",
        7,
        "class_conditional",
        None,
    ),
    (
        "biggan",
        "Compact BigGAN EMA",
        CompactBigGANGenerator,
        MODEL_OUT_DIRS["biggan"] / "generator.pth",
        "6.2_biggan.py",
        7,
        "class_conditional",
        "ema",
    ),
)


def load_generator(
    model_name,
    model_class,
    checkpoint_path,
    script_name,
    expected_format_version,
    expected_conditioning,
    expected_weights,
    device,
):
    """Load one generator produced by the current lesson scripts."""
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

    expected_metadata = {
        "model_name": model_name,
        "format_version": expected_format_version,
        "conditioning": expected_conditioning,
    }
    if expected_weights is not None:
        expected_metadata["weights"] = expected_weights
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"{checkpoint_path} has {key}={checkpoint.get(key)!r}; "
                f"expected {expected!r}. Retrain it with {script_name}."
            )

    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint metadata is incomplete: {checkpoint_path}."
        )
    checkpoint_image_size = model_config.get("image_size", 32)
    if checkpoint_image_size != 32:
        raise ValueError(
            "Expected a 32x32 checkpoint at "
            f"{checkpoint_path}."
        )
    if expected_conditioning == "class_conditional":
        if model_config.get("num_classes") != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} classes at {checkpoint_path}."
            )

    # All lesson generators are fixed at 32x32; discard legacy metadata.
    constructor_config = {
        key: value for key, value in model_config.items() if key != "image_size"
    }
    generator = model_class(**constructor_config).to(device)
    generator.load_state_dict(state_dict, strict=True)
    return generator.eval(), model_config


def validate_spectral_normalization(generators):
    """Check the generator-side SN increment introduced by SAGAN."""
    counts = {
        name: count_spectral_norm_layers(generator)
        for name, generator in generators.items()
    }
    if counts["sn_gan"] != 0:
        raise RuntimeError("SN-GAN should keep spectral normalization out of G.")
    if counts["sagan"] == 0 or counts["biggan"] == 0:
        raise RuntimeError("SAGAN and BigGAN require spectral normalization in G.")
    print(
        "Spectral-normalized generator layers: "
        f"SN-GAN={counts['sn_gan']}, "
        f"SAGAN={counts['sagan']}, "
        f"BigGAN={counts['biggan']}"
    )


@torch.inference_mode()
def save_sn_gan_samples(generator, z_dim, device):
    """Save unconditional samples without misleading class-column labels."""
    torch.manual_seed(SEED)
    z = torch.randn(NUM_UNCONDITIONAL_SAMPLES, z_dim, device=device)
    samples = generator(z).cpu()
    save_image_row_grid(
        [samples],
        ["SN-GAN"],
        EVALUATION_DIRS["sn_gan"] / "sn_gan_samples.png",
        title="Unconditional SN-GAN samples",
        column_labels=[
            f"Sample {index + 1}"
            for index in range(NUM_UNCONDITIONAL_SAMPLES)
        ],
    )


@torch.inference_mode()
def save_conditional_class_sweep(
    generators,
    z_dim,
    device,
):
    """Hold one z fixed while changing only the class label."""
    torch.manual_seed(SEED + 1)
    base_z = torch.randn(1, z_dim, device=device)
    noise, labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        1,
        z_dim,
        device,
        base_noise=base_z,
    )
    image_rows = [
        generators[model_name](noise, labels).cpu()
        for model_name in ("sagan", "biggan")
    ]
    for model_name in ("sagan", "biggan"):
        save_image_row_grid(
            image_rows,
            ["SAGAN", "Compact BigGAN EMA"],
            EVALUATION_DIRS[model_name] / "conditional_class_sweep.png",
            title="Fixed-z CIFAR-10 class response",
            column_labels=[
                name.title() for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
            ],
        )


@torch.inference_mode()
def save_biggan_truncation(generator, z_dim, device):
    """Compare BigGAN class sweeps under three latent distributions."""
    image_rows = []
    row_labels = []
    for threshold in TRUNCATION_THRESHOLDS:
        torch.manual_seed(SEED + 2)
        if threshold is None:
            base_z = torch.randn(1, z_dim, device=device)
            row_labels.append("Standard normal")
        else:
            base_z = truncated_normal(
                (1, z_dim),
                threshold,
                device,
            )
            row_labels.append(f"Truncated |z| <= {threshold:.1f}")
        noise, labels = make_fixed_class_latent_grid(
            NUM_CLASSES,
            1,
            z_dim,
            device,
            base_noise=base_z,
        )
        image_rows.append(generator(noise, labels).cpu())

    save_image_row_grid(
        image_rows,
        row_labels,
        EVALUATION_DIRS["biggan"] / "biggan_truncation.png",
        title="Compact BigGAN EMA truncation comparison",
        column_labels=[
            name.title() for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
        ],
    )


def main():
    for evaluation_dir in EVALUATION_DIRS.values():
        reset_dir(str(evaluation_dir))

    set_seed(SEED)
    device = try_gpu()

    generators = {}
    configurations = {}
    for (
        model_name,
        _,
        model_class,
        checkpoint_path,
        script_name,
        format_version,
        conditioning,
        weights,
    ) in MODEL_SPECS:
        generator, model_config = load_generator(
            model_name,
            model_class,
            checkpoint_path,
            script_name,
            format_version,
            conditioning,
            weights,
            device,
        )
        generators[model_name] = generator
        configurations[model_name] = model_config

    validate_spectral_normalization(generators)
    conditional_z_dims = {
        int(configurations[name]["z_dim"])
        for name in ("sagan", "biggan")
    }
    if len(conditional_z_dims) != 1:
        raise ValueError(
            "SAGAN and BigGAN need the same z_dim for the fixed-z class sweep."
        )
    conditional_z_dim = conditional_z_dims.pop()

    save_sn_gan_samples(
        generators["sn_gan"],
        int(configurations["sn_gan"]["z_dim"]),
        device,
    )
    save_conditional_class_sweep(
        generators,
        conditional_z_dim,
        device,
    )
    save_biggan_truncation(
        generators["biggan"],
        conditional_z_dim,
        device,
    )


if __name__ == "__main__":
    main()
