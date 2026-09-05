"""Train a 64-channel BigGAN on Imagenette at 128x128.

This lesson keeps SAGAN's conditional ResNet, attention, spectral
normalization, projection discriminator, and hinge loss, then adds BigGAN's
shared class embedding, six-way hierarchical ``z=120``, orthogonal
initialization, modified orthogonal regularization, two D updates per G
update, and a ``0.9999`` generator EMA. Adam uses the paper's ``G=5e-5``,
``D=2e-4``, and ``beta=(0, 0.999)`` settings. The 64-channel baseline and
batch 8 are deliberate single-RTX-5090 choices. The 20-epoch default is a
teaching-scale Imagenette budget; larger distributed training details are not
silently approximated.

Data:
    data/imagenette-128/train/<WNID>/*.JPEG, prepared explicitly with
    tool_scripts/download_dataset.py --dataset imagenette.

Outputs:
    Fresh runs replace the biggan output directory; --resume-from preserves it.
    output/gan/biggan/{training,checkpoints,generator.pth,
    online_generator.pth,discriminator.pth,loss_curves.png}. The final
    generator.pth contains EMA weights. Set DL_OUTPUT_ROOT to relocate all
    model directories together on a cloud volume.
"""

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import cast

from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dl_utils.data.imagenette import (
    IMAGENETTE_CLASS_NAMES,
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    make_imagenette_loader,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.biggan import (
    BigGANDiscriminator,
    BigGANGenerator,
    initialize_orthogonal_weights,
    modified_orthogonal_regularization,
)
from dl_utils.gan.conditional_training import train_conditional_hinge_epoch
from dl_utils.gan.sagan import make_fixed_class_latent_grid
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.accelerator import (
    configure_device,
    make_fused_adam,
    resolve_training_precision,
)
from dl_utils.training.optimization import UpdateRatioSchedule, update_ema
from dl_utils.training.session import TrainingSession

PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "imagenette-128"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("DL_OUTPUT_ROOT", PROJECT_ROOT / "output" / "gan")
)

NUM_EPOCHS = 20
BATCH_SIZE = 8
NUM_WORKERS = 8
NUM_CLASSES = IMAGENETTE_NUM_CLASSES
Z_DIM = 120
BASE_CHANNELS = 64
CLASS_EMBEDDING_DIM = 128
IMAGE_SIZE = IMAGENETTE_IMAGE_SIZE
ATTENTION_RESOLUTION = 64
GENERATOR_LR = 5e-5
DISCRIMINATOR_LR = 2e-4
DISCRIMINATOR_UPDATES_PER_GENERATOR = 2
ORTHOGONAL_REGULARIZATION_STRENGTH = 1e-4
EMA_DECAY = 0.9999
DISPLAY_CLASS_INDICES = tuple(range(NUM_CLASSES))
SAMPLES_PER_CLASS = 4
SAMPLE_EVERY_EPOCHS = 1
CHECKPOINT_EVERY_EPOCHS = 1
ARCHIVE_EVERY_EPOCHS = 5
SEED = 42
FORMAT_VERSION = 10

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "class_embedding_dim": CLASS_EMBEDDING_DIM,
    "image_size": IMAGE_SIZE,
    "attention_resolution": ATTENTION_RESOLUTION,
}
DISCRIMINATOR_CONFIG = {
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "image_size": IMAGE_SIZE,
    "attention_resolution": ATTENTION_RESOLUTION,
}


def main(args):
    output_dir = args.output_root / "biggan"
    training_dir = output_dir / "training"

    set_seed(SEED)
    device = try_gpu()
    configure_device(device)
    precision = resolve_training_precision(device, args.precision)
    loader = make_imagenette_loader(
        args.data_dir,
        args.batch_size,
        device,
        horizontal_flip=False,
        num_workers=args.num_workers,
        preprocessed=True,
    )
    update_schedule = UpdateRatioSchedule(
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
        len(loader),
        args.epochs,
    )

    generator = BigGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = BigGANDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    generator.apply(initialize_orthogonal_weights)
    discriminator.apply(initialize_orthogonal_weights)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=device,
        lr=GENERATOR_LR,
        betas=(0.0, 0.999),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=device,
        lr=DISCRIMINATOR_LR,
        betas=(0.0, 0.999),
    )

    run_metadata = {
        "format_version": FORMAT_VERSION,
        "dataset": "imagenette-128",
        "image_size": IMAGE_SIZE,
        "conditioning": "class_conditional",
        "discriminator_conditioning": "projection",
        "precision": precision.name,
        "batch_size": args.batch_size,
        "update_ratio": DISCRIMINATOR_UPDATES_PER_GENERATOR,
        "ema_decay": EMA_DECAY,
        "orthogonal_regularization_strength": (ORTHOGONAL_REGULARIZATION_STRENGTH),
    }
    session = TrainingSession(
        output_dir,
        total_epochs=args.epochs,
        models={
            "online_generator": generator,
            "generator": averaged_generator,
            "discriminator": discriminator,
        },
        optimizers={
            "generator": optimizer_g,
            "discriminator": optimizer_d,
        },
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata=run_metadata,
        model_metadata={
            "online_generator": {
                "model_name": "biggan_online_generator",
                "model_config": MODEL_CONFIG,
                "weights": "online",
            },
            "generator": {
                "model_name": "biggan",
                "model_config": MODEL_CONFIG,
                "weights": "ema",
            },
            "discriminator": {
                "model_name": "biggan_discriminator",
                "model_config": DISCRIMINATOR_CONFIG,
            },
        },
    )
    fixed_noise, fixed_labels = make_fixed_class_latent_grid(
        DISPLAY_CLASS_INDICES,
        SAMPLES_PER_CLASS,
        Z_DIM,
        device,
    )
    start_epoch, state = session.start(
        args.resume_from,
        initial_state={
            "loss_history": {
                "epoch": [],
                "discriminator": [],
                "generator_total": [],
                "generator_adversarial": [],
                "generator_regularization": [],
            },
            "fixed_noise": fixed_noise.cpu(),
            "fixed_labels": fixed_labels.cpu(),
            "discriminator_steps": 0,
        },
    )
    training_dir.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_noise = state["fixed_noise"].to(device)
    fixed_labels = state["fixed_labels"].to(device)
    discriminator_steps = state["discriminator_steps"]
    expected_steps = update_schedule.completed_discriminator_updates(start_epoch - 1)
    if discriminator_steps != expected_steps:
        raise ValueError("Checkpoint discriminator step count is inconsistent.")
    if loss_history["epoch"] != list(range(1, start_epoch)):
        raise ValueError("Checkpoint loss history does not match its epoch.")

    dataset = cast(ImageFolder, loader.dataset)
    class_names = [
        IMAGENETTE_CLASS_NAMES[dataset.classes[index]]
        for index in DISPLAY_CLASS_INDICES
    ]
    planned_d_steps = update_schedule.total_discriminator_updates
    planned_g_steps = update_schedule.total_generator_updates
    print(
        f"Device={device}; precision={precision.name}; "
        f"Imagenette batches/epoch={len(loader):,}"
    )
    print(
        f"Planned optimizer updates: D={planned_d_steps:,}, "
        f"G={planned_g_steps:,} ({DISCRIMINATOR_UPDATES_PER_GENERATOR}:1)"
    )

    with tqdm(
        total=planned_d_steps,
        initial=discriminator_steps,
        desc=f"Epoch {min(start_epoch, args.epochs)}/{args.epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(start_epoch, args.epochs + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{args.epochs}",
                refresh=False,
            )
            result = train_conditional_hinge_epoch(
                generator,
                discriminator,
                loader,
                optimizer_g,
                optimizer_d,
                device,
                precision,
                z_dim=Z_DIM,
                num_classes=NUM_CLASSES,
                generator_batch_size=args.batch_size,
                discriminator_steps=discriminator_steps,
                update_schedule=update_schedule,
                generator_regularizer=lambda model: modified_orthogonal_regularization(
                    model,
                    strength=ORTHOGONAL_REGULARIZATION_STRENGTH,
                ),
                after_generator_step=lambda: update_ema(
                    averaged_generator,
                    generator,
                    decay=EMA_DECAY,
                ),
                progress_bar=progress_bar,
            )
            discriminator_steps = result.discriminator_steps
            loss_history["epoch"].append(epoch)
            loss_history["discriminator"].append(result.discriminator)
            loss_history["generator_total"].append(result.generator_total)
            loss_history["generator_adversarial"].append(result.generator_adversarial)
            loss_history["generator_regularization"].append(
                result.generator_regularization
            )

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    averaged_generator,
                    fixed_noise,
                    fixed_labels,
                    training_dir / f"epoch_{epoch:03d}.png",
                    class_names=class_names,
                    class_indices=DISPLAY_CLASS_INDICES,
                    title=f"BigGAN EMA Imagenette-128 samples - epoch {epoch:03d}",
                    shared_latents_across_classes=True,
                    inference_batch_size=2,
                )

            state["discriminator_steps"] = discriminator_steps
            session.checkpoint(epoch, state)

    session.finish()
    save_loss_panels(
        loss_history["epoch"],
        {
            "Discriminator hinge loss": {
                "D hinge loss": loss_history["discriminator"],
            },
            "Generator objective": {
                "G total loss": loss_history["generator_total"],
                "G adversarial loss": loss_history["generator_adversarial"],
            },
            "Modified orthogonal regularization": {
                "G orthogonal penalty": loss_history["generator_regularization"],
            },
        },
        output_dir / "loss_curves.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Imagenette-128 class-conditional BigGAN."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="auto",
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch-size must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
