"""Train a class-conditional projection SN-GAN on Imagenette at 128x128.

This lesson keeps conditional BatchNorm, projection discrimination, hinge
loss, and discriminator-only spectral normalization. Imagenette defaults use
``z=128``, ``ch=32``, batches of 32, Adam ``(2e-4, beta1=0, beta2=0.9)``,
and one discriminator update per generator update. The narrower network and
40-epoch budget give G about 11,800 updates instead of the previous 2,364,
while reducing the cost of each update. Random horizontal flips provide
lightweight data augmentation. These are teaching defaults, not paper-scale
reproduction settings; SAGAN next adds generator SN and self-attention.

Data:
    data/imagenette-128/train/<WNID>/*.JPEG, prepared explicitly with
    tool_scripts/download_dataset.py --dataset imagenette.

Outputs:
    Fresh runs replace the sn_gan output directory; --resume-from preserves it.
    output/gan/sn_gan/{training,checkpoints,generator.pth,
    discriminator.pth,loss_curves.png}. Set DL_OUTPUT_ROOT to relocate the
    three model directories together on a cloud volume.
"""

import argparse
import os
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
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.conditional_training import train_conditional_hinge_epoch
from dl_utils.gan.sagan import make_fixed_class_latent_grid
from dl_utils.gan.sn_gan import SNDiscriminator, SNGenerator
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.accelerator import (
    configure_device,
    make_fused_adam,
    resolve_training_precision,
)
from dl_utils.training.optimization import UpdateRatioSchedule
from dl_utils.training.session import TrainingSession

PROJECT_ROOT = infer_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "imagenette-128"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("DL_OUTPUT_ROOT", PROJECT_ROOT / "output" / "gan")
)

NUM_EPOCHS = 40
DISCRIMINATOR_BATCH_SIZE = 32
GENERATOR_BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_CLASSES = IMAGENETTE_NUM_CLASSES
Z_DIM = 128
BASE_CHANNELS = 32
IMAGE_SIZE = IMAGENETTE_IMAGE_SIZE
LEARNING_RATE = 2e-4
DISCRIMINATOR_UPDATES_PER_GENERATOR = 1
DISPLAY_CLASS_INDICES = tuple(range(NUM_CLASSES))
SAMPLES_PER_CLASS = 4
SAMPLE_EVERY_EPOCHS = 1
CHECKPOINT_EVERY_EPOCHS = 1
ARCHIVE_EVERY_EPOCHS = 5
SEED = 42
FORMAT_VERSION = 9

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "image_size": IMAGE_SIZE,
}
DISCRIMINATOR_CONFIG = {
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "image_size": IMAGE_SIZE,
}


def main(args):
    output_dir = args.output_root / "sn_gan"
    training_dir = output_dir / "training"

    set_seed(SEED)
    device = try_gpu()
    configure_device(device)
    precision = resolve_training_precision(device, args.precision)
    loader = make_imagenette_loader(
        args.data_dir,
        args.batch_size,
        device,
        dequantize=True,
        horizontal_flip=True,
        num_workers=args.num_workers,
        preprocessed=True,
    )
    update_schedule = UpdateRatioSchedule(
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
        len(loader),
        args.epochs,
    )

    generator = SNGenerator(**MODEL_CONFIG).to(device)
    discriminator = SNDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=device,
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=device,
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )

    run_metadata = {
        "format_version": FORMAT_VERSION,
        "dataset": "imagenette-128",
        "image_size": IMAGE_SIZE,
        "conditioning": "class_conditional",
        "discriminator_conditioning": "projection",
        "precision": precision.name,
        "discriminator_batch_size": args.batch_size,
        "generator_batch_size": args.generator_batch_size,
        "generator_lr": LEARNING_RATE,
        "discriminator_lr": LEARNING_RATE,
        "horizontal_flip": True,
        "update_ratio": DISCRIMINATOR_UPDATES_PER_GENERATOR,
    }
    session = TrainingSession(
        output_dir,
        total_epochs=args.epochs,
        models={"generator": generator, "discriminator": discriminator},
        optimizers={
            "generator": optimizer_g,
            "discriminator": optimizer_d,
        },
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata=run_metadata,
        model_metadata={
            "generator": {
                "model_name": "sn_gan",
                "model_config": MODEL_CONFIG,
            },
            "discriminator": {
                "model_name": "sn_gan_discriminator",
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
                "generator": [],
            },
            "fixed_noise": fixed_noise.cpu(),
            "fixed_labels": fixed_labels.cpu(),
            "discriminator_steps": 0,
        },
    )
    if args.resume_from is None:
        reset_dir(str(training_dir))
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
                generator_batch_size=args.generator_batch_size,
                discriminator_steps=discriminator_steps,
                update_schedule=update_schedule,
                progress_bar=progress_bar,
            )
            discriminator_steps = result.discriminator_steps
            loss_history["epoch"].append(epoch)
            loss_history["discriminator"].append(result.discriminator)
            loss_history["generator"].append(result.generator_total)

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    generator,
                    fixed_noise,
                    fixed_labels,
                    training_dir / f"epoch_{epoch:03d}.png",
                    class_names=class_names,
                    class_indices=DISPLAY_CLASS_INDICES,
                    title=f"SN-GAN Imagenette-128 samples - epoch {epoch:03d}",
                    shared_latents_across_classes=True,
                    inference_batch_size=4,
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
            "Generator hinge objective": {
                "G hinge objective": loss_history["generator"],
            },
        },
        output_dir / "loss_curves.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Imagenette-128 class-conditional SN-GAN."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DISCRIMINATOR_BATCH_SIZE,
        help="Real/fake discriminator batch size.",
    )
    parser.add_argument(
        "--generator-batch-size",
        type=int,
        default=GENERATOR_BATCH_SIZE,
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="auto",
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.generator_batch_size) < 1:
        parser.error("epochs and batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
