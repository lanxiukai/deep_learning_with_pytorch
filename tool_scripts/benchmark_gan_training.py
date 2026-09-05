"""Run finite forward/backward checks for one Imagenette-128 GAN on RTX 5090."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from dl_utils.data.imagenette import (
    IMAGENETTE_IMAGE_SIZE,
    IMAGENETTE_NUM_CLASSES,
    make_imagenette_loader,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.gan.biggan import (
    BigGANDiscriminator,
    BigGANGenerator,
    initialize_orthogonal_weights,
    modified_orthogonal_regularization,
)
from dl_utils.gan.conditional_training import train_conditional_hinge_step
from dl_utils.gan.sagan import SAGANDiscriminator, SAGANGenerator
from dl_utils.gan.sn_gan import SNDiscriminator, SNGenerator
from dl_utils.training.accelerator import (
    configure_device,
    make_fused_adam,
    resolve_training_precision,
)
from dl_utils.training.optimization import update_ema

MODEL_DEFAULT_BATCH_SIZES = {
    "sn_gan": (32, 32),
    "sagan": (32, 32),
    "biggan": (32, 32),
}
NUM_CLASSES = IMAGENETTE_NUM_CLASSES
IMAGE_SIZE = IMAGENETTE_IMAGE_SIZE


def build_model(model_name, device, base_channels):
    """Construct the lesson's default architecture and optimizer pair."""
    common = {
        "num_classes": NUM_CLASSES,
        "base_channels": base_channels,
        "image_size": IMAGE_SIZE,
    }
    if model_name == "sn_gan":
        z_dim = 128
        generator = SNGenerator(z_dim=z_dim, **common).to(device)
        discriminator = SNDiscriminator(**common).to(device)
        generator_lr = discriminator_lr = 2e-4
        betas = (0.0, 0.9)
        update_ratio = 1
        averaged_generator = None
    elif model_name == "sagan":
        z_dim = 128
        generator = SAGANGenerator(
            z_dim=z_dim,
            attention_resolution=32,
            **common,
        ).to(device)
        discriminator = SAGANDiscriminator(
            attention_resolution=32,
            **common,
        ).to(device)
        generator_lr = discriminator_lr = 1e-4
        betas = (0.0, 0.9)
        update_ratio = 1
        averaged_generator = None
    else:
        z_dim = 120
        generator = BigGANGenerator(
            z_dim=z_dim,
            class_embedding_dim=128,
            attention_resolution=32,
            **common,
        ).to(device)
        discriminator = BigGANDiscriminator(
            attention_resolution=32,
            **common,
        ).to(device)
        generator.apply(initialize_orthogonal_weights)
        discriminator.apply(initialize_orthogonal_weights)
        averaged_generator = deepcopy(generator).eval().requires_grad_(False)
        generator_lr = discriminator_lr = 1e-4
        betas = (0.0, 0.9)
        update_ratio = 2

    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=device,
        lr=generator_lr,
        betas=betas,
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=device,
        lr=discriminator_lr,
        betas=betas,
    )
    return (
        generator,
        discriminator,
        averaged_generator,
        optimizer_g,
        optimizer_d,
        z_dim,
        update_ratio,
    )


def make_real_batch_source(args, device, batch_size):
    """Use prepared Imagenette when requested, otherwise deterministic noise."""
    if args.data_dir is None:
        random_generator = torch.Generator(device=device).manual_seed(args.seed)

        def synthetic_batch():
            images = (
                torch.rand(
                    batch_size,
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    device=device,
                    generator=random_generator,
                )
                .mul_(2)
                .sub_(1)
            )
            labels = torch.randint(
                NUM_CLASSES,
                (batch_size,),
                device=device,
                generator=random_generator,
            )
            return images, labels

        return synthetic_batch

    loader = make_imagenette_loader(
        args.data_dir,
        batch_size,
        device,
        dequantize=args.model == "sn_gan",
        horizontal_flip=True,
        num_workers=args.num_workers,
        preprocessed=True,
    )
    iterator = iter(loader)

    def imagenette_batch():
        nonlocal iterator
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        return (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

    return imagenette_batch


def benchmark(args):
    """Run warmup and timed optimization steps, then return JSON-ready data."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if not args.allow_other_gpu and "RTX 5090" not in gpu_name.upper():
        raise RuntimeError(f"Expected an RTX 5090, found {gpu_name!r}.")
    configure_device(device)
    precision = resolve_training_precision(device, args.precision)
    default_batch_size, default_generator_batch_size = MODEL_DEFAULT_BATCH_SIZES[
        args.model
    ]
    batch_size = args.batch_size or default_batch_size
    generator_batch_size = args.generator_batch_size or (
        batch_size if args.batch_size is not None else default_generator_batch_size
    )
    torch.manual_seed(args.seed)

    (
        generator,
        discriminator,
        averaged_generator,
        optimizer_g,
        optimizer_d,
        z_dim,
        update_ratio,
    ) = build_model(args.model, device, args.base_channels)
    next_real_batch = make_real_batch_source(args, device, batch_size)
    last_loss_tensors = {}
    discriminator_steps = 0
    generator_steps = 0
    generator_regularizer = None
    after_generator_step = None
    if averaged_generator is not None:
        generator_regularizer = partial(
            modified_orthogonal_regularization,
            strength=1e-4,
        )
        after_generator_step = partial(
            update_ema,
            averaged_generator,
            generator,
            decay=0.999,
        )

    def train_step():
        nonlocal discriminator_steps, generator_steps, last_loss_tensors
        real, labels = next_real_batch()
        result = train_conditional_hinge_step(
            generator,
            discriminator,
            real,
            labels,
            optimizer_g,
            optimizer_d,
            device,
            precision,
            z_dim=z_dim,
            num_classes=NUM_CLASSES,
            generator_batch_size=generator_batch_size,
            discriminator_steps=discriminator_steps,
            discriminator_updates_per_generator=update_ratio,
            generator_regularizer=generator_regularizer,
            after_generator_step=after_generator_step,
        )
        discriminator_steps = result.discriminator_steps
        last_loss_tensors["discriminator"] = result.discriminator
        if result.generator_total is not None:
            assert result.generator_adversarial is not None
            assert result.generator_regularization is not None
            generator_steps += 1
            last_loss_tensors.update(
                generator=result.generator_total,
                generator_adversarial=result.generator_adversarial,
                generator_regularization=result.generator_regularization,
            )

    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timed_start_d = discriminator_steps
    timed_start_g = generator_steps
    start = time.perf_counter()
    for _ in range(args.steps):
        train_step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if generator_steps == 0:
        raise ValueError(
            "warmup-steps plus steps must reach one generator update for "
            f"the {update_ratio}:1 schedule."
        )
    last_losses = {name: float(value) for name, value in last_loss_tensors.items()}
    if not all(math.isfinite(value) for value in last_losses.values()):
        raise FloatingPointError(f"Non-finite loss: {last_losses}")

    result = {
        "model": args.model,
        "gpu": gpu_name,
        "precision": precision.name,
        "base_channels": args.base_channels,
        "batch_size": batch_size,
        "generator_batch_size": generator_batch_size,
        "resolution": IMAGE_SIZE,
        "warmup_steps": args.warmup_steps,
        "timed_discriminator_steps": discriminator_steps - timed_start_d,
        "timed_generator_steps": generator_steps - timed_start_g,
        "total_generator_steps": generator_steps,
        "seconds": elapsed,
        "images_per_second": args.steps * batch_size / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "losses": last_losses,
        "data": "imagenette-128" if args.data_dir is not None else "synthetic",
    }
    if args.json_output is not None:
        # This run owns one JSON file; its parent may hold other benchmarks.
        if not args.json_output.parent.exists():
            reset_dir(str(args.json_output.parent))
        args.json_output.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_DEFAULT_BATCH_SIZES),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--generator-batch-size", type=int)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-other-gpu",
        action="store_true",
        help="permit development checks on a CUDA GPU other than RTX 5090",
    )
    args = parser.parse_args()
    positive_values = [args.steps, args.base_channels]
    if args.batch_size is not None:
        positive_values.append(args.batch_size)
    if args.generator_batch_size is not None:
        positive_values.append(args.generator_batch_size)
    if min(positive_values) < 1 or args.warmup_steps < 0:
        parser.error("steps, batch-size, and channels must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    return args


if __name__ == "__main__":
    print(json.dumps(benchmark(parse_args()), indent=2))
