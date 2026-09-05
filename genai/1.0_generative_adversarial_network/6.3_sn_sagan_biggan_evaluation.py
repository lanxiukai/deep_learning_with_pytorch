"""Compare Imagenette-128 references with conditional GAN samples.

Each available generator receives the same ten Imagenette class indices and
model-specific evaluation noise. Missing checkpoints are reported and skipped
so the three lessons can still be trained and inspected independently.
"""

import argparse
import os
from pathlib import Path

import torch

from dl_utils.data.imagenette import (
    IMAGENETTE_CLASS_NAMES,
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    make_imagenette_dataset,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.biggan import BigGANGenerator
from dl_utils.gan.inference import generate_in_batches
from dl_utils.gan.sagan import SAGANGenerator
from dl_utils.gan.sn_gan import SNGenerator
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights

PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "imagenette-128"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("DL_OUTPUT_ROOT", PROJECT_ROOT / "output" / "gan")
)
IMAGE_SIZE = IMAGENETTE_IMAGE_SIZE
NUM_CLASSES = IMAGENETTE_NUM_CLASSES
DISPLAY_CLASS_INDICES = tuple(range(NUM_CLASSES))
EVALUATION_SEED = 10_042
MODEL_EVALUATION_SEEDS = {
    "sn_gan": EVALUATION_SEED + 1,
    "sagan": EVALUATION_SEED + 2,
    "biggan": EVALUATION_SEED + 3,
}
MODEL_SPECS = (
    ("sn_gan", "SN-GAN", SNGenerator, "6.0_sn_gan.py", 9, None),
    ("sagan", "SAGAN", SAGANGenerator, "6.1_sagan.py", 12, None),
    ("biggan", "BigGAN EMA", BigGANGenerator, "6.2_biggan.py", 11, "ema"),
)


def load_imagenette_references(data_dir):
    """Return one deterministic normalized image for each display class."""
    dataset = make_imagenette_dataset(data_dir, preprocessed=True)
    candidates = {class_index: [] for class_index in DISPLAY_CLASS_INDICES}
    for sample_index, class_index in enumerate(dataset.targets):
        if class_index in candidates:
            candidates[class_index].append(sample_index)
    if any(not indices for indices in candidates.values()):
        raise RuntimeError("The Imagenette tree is missing a display class.")

    random_generator = torch.Generator().manual_seed(EVALUATION_SEED)
    selected_indices = [
        indices[
            int(
                torch.randint(
                    len(indices),
                    (),
                    generator=random_generator,
                ).item()
            )
        ]
        for indices in candidates.values()
    ]
    images = torch.stack([dataset[index][0] for index in selected_indices])
    labels = torch.tensor(DISPLAY_CLASS_INDICES, dtype=torch.long)
    class_names = [
        IMAGENETTE_CLASS_NAMES[dataset.classes[index]]
        for index in DISPLAY_CLASS_INDICES
    ]
    return images, labels, class_names


def load_generator(
    model_name,
    model_class,
    checkpoint_path,
    script_name,
    expected_format_version,
    expected_weights,
    device,
):
    """Load a generator produced by the current Imagenette lesson scripts."""
    expected_metadata = {
        "model_name": model_name,
        "format_version": expected_format_version,
        "conditioning": "class_conditional",
    }
    if expected_weights is not None:
        expected_metadata["weights"] = expected_weights

    def validate_config(model_config):
        if model_config.get("image_size") != IMAGE_SIZE:
            raise ValueError(
                f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} checkpoint at {checkpoint_path}."
            )
        if model_config.get("num_classes") != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} classes at {checkpoint_path}.")
        return model_config

    try:
        return load_model_weights(
            checkpoint_path,
            model_class,
            device=device,
            expected_metadata=expected_metadata,
            config_transform=validate_config,
        )
    except ValueError as error:
        raise ValueError(f"{error} Retrain it with {script_name}.") from error


@torch.inference_mode()
def save_real_vs_generated(
    model_name,
    display_name,
    generator,
    model_config,
    reference_images,
    reference_labels,
    class_names,
    output_path,
    device,
    batch_size,
):
    """Save one model's Imagenette reference-versus-sample comparison."""
    z_dim = model_config.get("z_dim")
    if not isinstance(z_dim, int) or z_dim < 1:
        raise ValueError(f"{model_name} checkpoint has an invalid z_dim.")

    random_generator = torch.Generator().manual_seed(MODEL_EVALUATION_SEEDS[model_name])
    noise = torch.randn(
        len(DISPLAY_CLASS_INDICES),
        z_dim,
        generator=random_generator,
    ).to(device)
    generated_images = generate_in_batches(
        (noise, reference_labels.to(device)),
        batch_size,
        lambda noise_batch, label_batch: generator(
            noise_batch,
            label_batch,
        ).float(),
        module=generator,
    )
    expected_shape = (
        len(DISPLAY_CLASS_INDICES),
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )
    if tuple(generated_images.shape) != expected_shape:
        raise ValueError(
            f"{model_name} generated {tuple(generated_images.shape)}; "
            f"expected {expected_shape}."
        )
    save_image_row_grid(
        [reference_images, generated_images],
        ["Imagenette train", display_name],
        output_path,
        title=f"{display_name}: Imagenette-128 reference vs conditional sample",
        column_labels=class_names,
    )


def main(args):
    for model_name, *_ in MODEL_SPECS:
        reset_dir(str(args.output_root / model_name / "evaluation"))
    available_specs = []
    for spec in MODEL_SPECS:
        model_name, display_name, _, script_name, *_ = spec
        checkpoint_path = args.output_root / model_name / "generator.pth"
        if checkpoint_path.is_file():
            available_specs.append((*spec, checkpoint_path))
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
    references, labels, class_names = load_imagenette_references(args.data_dir)
    for (
        model_name,
        display_name,
        model_class,
        script_name,
        format_version,
        weights,
        checkpoint_path,
    ) in available_specs:
        evaluation_dir = args.output_root / model_name / "evaluation"
        generator, model_config = load_generator(
            model_name,
            model_class,
            checkpoint_path,
            script_name,
            format_version,
            weights,
            device,
        )
        output_path = evaluation_dir / "real_vs_generated.png"
        save_real_vs_generated(
            model_name,
            display_name,
            generator,
            model_config,
            references,
            labels,
            class_names,
            output_path,
            device,
            args.batch_size,
        )
        print(f"Saved {display_name} comparison: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Imagenette-128 SN-GAN, SAGAN, and BigGAN."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
