"""Compare CIFAR-10 reference images with fresh GAN samples.

Each available model gets one focused two-row figure.  The first row contains
one CIFAR-10 test image per class; the second uses evaluation-only latent
vectors that are independent of the fixed inputs saved under ``training/``.
SAGAN and BigGAN generate the class named above each column, while SN-GAN
remains unconditional and makes no per-column class claim.

Missing generator checkpoints are reported and skipped so any completed model
in the lesson sequence can be evaluated independently.

Inputs:
    data/cifar10, prepared by tool_scripts/download_dataset.py
    output/sn_gan/generator.pth, created by 6.0_sn_gan.py
    output/sagan/generator.pth, created by 6.1_sagan.py
    output/biggan/generator.pth, created by 6.2_biggan.py

Outputs:
    Each evaluation directory is reset at the start of every run.
    output/sn_gan/evaluation/real_vs_generated.png
    output/sagan/evaluation/real_vs_generated.png
    output/biggan/evaluation/real_vs_generated.png
"""

import torch

from dl_utils.data.cifar10 import (
    make_cifar10_dataset,
    normalized_cifar10_transform,
)
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.biggan import CompactBigGANGenerator
from dl_utils.gan.sagan import (
    CIFAR10_CLASS_NAMES,
    SAGANGenerator,
)
from dl_utils.gan.sn_gan import (
    SNGenerator,
)
from dl_utils.gan.inference import generate_in_batches
from dl_utils.plot.images import save_image_row_grid
from dl_utils.training.checkpoints import load_model_weights


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
MODEL_OUT_DIRS = {
    model_name: PROJECT_ROOT / "output" / model_name
    for model_name in ("sn_gan", "sagan", "biggan")
}
EVALUATION_DIRS = {
    model_name: MODEL_OUT_DIRS[model_name] / "evaluation"
    for model_name in MODEL_OUT_DIRS
}

NUM_CLASSES = 10
EVALUATION_SEED = 10_042
MODEL_EVALUATION_SEEDS = {
    "sn_gan": EVALUATION_SEED + 1,
    "sagan": EVALUATION_SEED + 2,
    "biggan": EVALUATION_SEED + 3,
}

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
        8,
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


def load_cifar10_test_references():
    """Return one deterministic, normalized test image for every class."""
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    dataset = make_cifar10_dataset(
        DATA_DIR,
        train=False,
        transform=normalized_cifar10_transform(horizontal_flip=False),
    )
    indices_by_class = [[] for _ in range(NUM_CLASSES)]
    for sample_index, class_index in enumerate(dataset.targets):
        indices_by_class[class_index].append(sample_index)
    if any(not indices for indices in indices_by_class):
        raise RuntimeError("CIFAR-10 test data is missing a class.")

    random_generator = torch.Generator().manual_seed(EVALUATION_SEED)
    selected_indices = [
        indices[
            torch.randint(
                len(indices),
                (),
                generator=random_generator,
            ).item()
        ]
        for indices in indices_by_class
    ]

    images = torch.stack(
        [dataset[index][0] for index in selected_indices]
    )
    labels = torch.arange(NUM_CLASSES, dtype=torch.long)
    return images, labels


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

    expected_metadata = {
        "model_name": model_name,
        "format_version": expected_format_version,
        "conditioning": expected_conditioning,
    }
    if expected_weights is not None:
        expected_metadata["weights"] = expected_weights

    def prepare_constructor_config(model_config):
        checkpoint_image_size = model_config.get("image_size", 32)
        if checkpoint_image_size != 32:
            raise ValueError(
                f"Expected a 32x32 checkpoint at {checkpoint_path}."
            )
        if (
            expected_conditioning == "class_conditional"
            and model_config.get("num_classes") != NUM_CLASSES
        ):
            raise ValueError(
                f"Expected {NUM_CLASSES} classes at {checkpoint_path}."
            )
        return {
            key: value
            for key, value in model_config.items()
            if key != "image_size"
        }

    try:
        return load_model_weights(
            checkpoint_path,
            model_class,
            device=device,
            expected_metadata=expected_metadata,
            config_transform=prepare_constructor_config,
        )
    except ValueError as error:
        raise ValueError(f"{error} Retrain it with {script_name}.") from error


@torch.inference_mode()
def save_real_vs_generated(
    model_name,
    display_name,
    generator,
    model_config,
    conditioning,
    reference_images,
    reference_labels,
    device,
):
    """Save one model's CIFAR-10 test-versus-fresh-sample comparison."""
    z_dim = model_config.get("z_dim")
    if not isinstance(z_dim, int) or z_dim < 1:
        raise ValueError(f"{model_name} checkpoint has an invalid z_dim.")

    random_generator = torch.Generator().manual_seed(
        MODEL_EVALUATION_SEEDS[model_name]
    )
    noise = torch.randn(
        NUM_CLASSES,
        z_dim,
        generator=random_generator,
    ).to(device)
    if conditioning == "class_conditional":
        generated_images = generate_in_batches(
            (noise, reference_labels.to(device)),
            NUM_CLASSES,
            lambda noise_batch, label_batch: generator(
                noise_batch,
                label_batch,
            ).float(),
            module=generator,
        )
        generated_row_label = display_name
        column_labels = [
            name.title() for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
        ]
        title = f"{display_name}: CIFAR-10 test vs class-conditioned samples"
    else:
        generated_images = generate_in_batches(
            noise,
            NUM_CLASSES,
            lambda noise_batch: generator(noise_batch).float(),
            module=generator,
        )
        generated_row_label = f"{display_name} (unconditional)"
        column_labels = [
            f"Reference: {name.title()}"
            for name in CIFAR10_CLASS_NAMES[:NUM_CLASSES]
        ]
        title = f"{display_name}: CIFAR-10 test vs unconditional samples"

    expected_shape = (NUM_CLASSES, 3, 32, 32)
    if tuple(generated_images.shape) != expected_shape:
        raise ValueError(
            f"{model_name} generated {tuple(generated_images.shape)}; "
            f"expected {expected_shape}."
        )
    save_image_row_grid(
        [reference_images, generated_images],
        ["CIFAR-10 test", generated_row_label],
        EVALUATION_DIRS[model_name] / "real_vs_generated.png",
        title=title,
        column_labels=column_labels,
    )
    return EVALUATION_DIRS[model_name] / "real_vs_generated.png"


def main():
    for evaluation_dir in EVALUATION_DIRS.values():
        reset_dir(str(evaluation_dir))

    available_specs = []
    for spec in MODEL_SPECS:
        model_name, display_name, _, checkpoint_path, script_name, *_ = spec
        if checkpoint_path.is_file():
            available_specs.append(spec)
        else:
            print(
                f"Skipping {display_name}: {checkpoint_path} is missing. "
                f"Run {script_name} first."
            )
    if not available_specs:
        print("No generator checkpoints are available; nothing to evaluate.")
        return

    set_seed(EVALUATION_SEED)
    device = try_gpu()
    reference_images, reference_labels = load_cifar10_test_references()

    for (
        model_name,
        display_name,
        model_class,
        checkpoint_path,
        script_name,
        format_version,
        conditioning,
        weights,
    ) in available_specs:
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
        output_path = save_real_vs_generated(
            model_name,
            display_name,
            generator,
            model_config,
            conditioning,
            reference_images,
            reference_labels,
            device,
        )
        print(f"Saved {display_name} comparison: {output_path}")


if __name__ == "__main__":
    main()
