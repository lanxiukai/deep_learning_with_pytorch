"""Run finite forward/backward checks for one Imagenette-128 GAN on RTX 5090."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch

from dl_utils.data.imagenette import make_imagenette_loader
from dl_utils.gan.biggan import (
    BigGANDiscriminator,
    BigGANGenerator,
    initialize_orthogonal_weights,
    modified_orthogonal_regularization,
)
from dl_utils.gan.sagan import SAGANDiscriminator, SAGANGenerator
from dl_utils.gan.sn_gan import (
    SNDiscriminator,
    SNGenerator,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.training.accelerator import (
    configure_device,
    make_fused_adam,
    resolve_training_precision,
)
from dl_utils.training.optimization import update_ema

MODEL_DEFAULT_BATCH_SIZES = {
    "sn_gan": 16,
    "sagan": 16,
    "biggan": 8,
}
NUM_CLASSES = 10


def build_model(model_name, device, base_channels):
    """Construct the lesson's default architecture and optimizer pair."""
    common = {
        "num_classes": NUM_CLASSES,
        "base_channels": base_channels,
        "image_size": 128,
    }
    if model_name == "sn_gan":
        z_dim = 128
        generator = SNGenerator(z_dim=z_dim, **common).to(device)
        discriminator = SNDiscriminator(**common).to(device)
        generator_lr = discriminator_lr = 2e-4
        betas = (0.0, 0.9)
        update_ratio = 5
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
        generator_lr, discriminator_lr = 1e-4, 4e-4
        betas = (0.0, 0.9)
        update_ratio = 1
        averaged_generator = None
    else:
        z_dim = 120
        generator = BigGANGenerator(
            z_dim=z_dim,
            class_embedding_dim=128,
            attention_resolution=64,
            **common,
        ).to(device)
        discriminator = BigGANDiscriminator(
            attention_resolution=64,
            **common,
        ).to(device)
        generator.apply(initialize_orthogonal_weights)
        discriminator.apply(initialize_orthogonal_weights)
        averaged_generator = deepcopy(generator).eval().requires_grad_(False)
        generator_lr, discriminator_lr = 5e-5, 2e-4
        betas = (0.0, 0.999)
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
                    128,
                    128,
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
    batch_size = args.batch_size or MODEL_DEFAULT_BATCH_SIZES[args.model]
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

    def train_step():
        nonlocal discriminator_steps, generator_steps, last_loss_tensors
        real, labels = next_real_batch()
        current_batch_size = real.shape[0]
        optimizer_d.zero_grad(set_to_none=True)
        with precision.autocast():
            noise = torch.randn(current_batch_size, z_dim, device=device)
            fake_labels = torch.randint(
                NUM_CLASSES,
                (current_batch_size,),
                device=device,
            )
            with torch.no_grad():
                fake = generator(noise, fake_labels)
            loss_d = discriminator_hinge_loss(
                discriminator(real, labels),
                discriminator(fake, fake_labels),
            )
        precision.backward_step(loss_d, optimizer_d)
        discriminator_steps += 1
        last_loss_tensors["discriminator"] = loss_d.detach()

        if discriminator_steps % update_ratio == 0:
            optimizer_g.zero_grad(set_to_none=True)
            sampled_labels = torch.randint(
                NUM_CLASSES,
                (current_batch_size,),
                device=device,
            )
            noise = torch.randn(current_batch_size, z_dim, device=device)
            discriminator.requires_grad_(False)
            try:
                with precision.autocast():
                    fake = generator(noise, sampled_labels)
                    adversarial = generator_hinge_loss(
                        discriminator(fake, sampled_labels)
                    )
                regularization = adversarial.new_zeros(())
                if args.model == "biggan":
                    regularization = modified_orthogonal_regularization(
                        generator,
                        strength=1e-4,
                    )
                loss_g = adversarial.float() + regularization
                precision.backward_step(loss_g, optimizer_g)
                if averaged_generator is not None:
                    update_ema(
                        averaged_generator,
                        generator,
                        decay=0.9999,
                    )
            finally:
                discriminator.requires_grad_(True)
            generator_steps += 1
            last_loss_tensors.update(
                generator=loss_g.detach(),
                generator_adversarial=adversarial.detach(),
                generator_regularization=regularization.detach(),
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
        "resolution": 128,
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
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--base-channels", type=int, default=64)
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
    if min(positive_values) < 1 or args.warmup_steps < 0:
        parser.error("steps, batch-size, and channels must be positive")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative")
    return args


if __name__ == "__main__":
    print(json.dumps(benchmark(parse_args()), indent=2))
