"""Train a compact, paper-oriented StyleGAN2 on CIFAR-10.

Compared with StyleGAN, this lesson presents the main StyleGAN2 increments:
    - per-sample weight modulation/demodulation instead of AdaIN;
    - the original one-convolution 4x4 block and eight overlapping W slots;
    - skip-connected RGB outputs and full-resolution training;
    - residual discriminator downsampling;
    - lazy R1 and path-length regularization in separate optimizer passes;
    - lazy-regularization compensation for Adam's learning rate and beta2;
    - style mixing, scalar-strength layer noise, truncation state, and an
      image-count-based G-EMA.

The complete mapping, synthesis, and discriminator models live in
``dl_utils/genai/stylegan2.py``.  This is a pure-PyTorch 32x32 teaching model,
so it omits fused CUDA kernels, multi-GPU training, and high-resolution
capacity while leaving the new losses and update schedule explicit.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/stylegan2/training/epoch_*.png: fixed-z/noise EMA samples
    output/stylegan2/checkpoints/latest.pth: full recoverable training state
    output/stylegan2/checkpoints/epoch_*.pth: sparse full-state archives
    output/stylegan2/stylegan2_generator.pth: final EMA generator with w_avg
    output/stylegan2/online_generator.pth: final non-averaged generator
    output/stylegan2/discriminator.pth: final discriminator and configuration
    output/stylegan2/loss_curves.png: objectives and lazy regularization losses

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.2_stylegan2.py \
        --resume-from output/stylegan2/checkpoints/latest.pth

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (1,562 full batches; drop_last=True)
Training epochs:         100
Main optimizer updates:  156,200 D / 156,200 G (1:1)

Generator:                2.73 M params (plus one EMA copy)
Discriminator:            3.48 M params
Trainable total:          6.21 M params
"""

import argparse
import math
import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.stylegan2 import (
    StyleDiscriminator,
    StyleGenerator,
    denormalize,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan2"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
D_REG_EVERY = 16
PATH_WEIGHT = 2.0
G_REG_EVERY = 8
PATH_BATCH_SHRINK = 2
# StyleGAN2 defines smoothing by the number of images seen.  This is clearer
# than a fixed per-step decay and stays correct if the teaching batch changes.
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY_EPOCHS = 5
CHECKPOINT_EVERY_EPOCHS = 5
ARCHIVE_EVERY_EPOCHS = 25
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
}

DISCRIMINATOR_CONFIG = {
    "base_channels": BASE_CHANNELS,
}

METRIC_NAMES = (
    "loss_d",
    "loss_d_main",
    "loss_g",
    "loss_g_main",
    "weighted_r1",
    "weighted_path",
)


def sample_mixing_latents(batch_size, z_dim, device):
    """Usually return two z batches so adjacent styles cannot co-adapt."""
    z = torch.randn(batch_size, z_dim, device=device)
    mixing_z = (
        torch.randn_like(z)
        if random.random() < STYLE_MIXING_PROBABILITY
        else None
    )
    return z, mixing_z


def r1_penalty(real_scores, real_images):
    """Measure squared discriminator gradients at real images."""
    gradients = torch.autograd.grad(
        real_scores.sum(),
        real_images,
        create_graph=True,
    )[0]
    return gradients.square().flatten(1).sum(dim=1).mean()


def path_length_penalty(images, ws, running_mean, decay=0.01):
    """Regularize image-space change per unit movement in W."""
    noise = torch.randn_like(images) / math.sqrt(
        images.shape[2] * images.shape[3]
    )
    gradients = torch.autograd.grad(
        (images * noise).sum(),
        ws,
        create_graph=True,
    )[0]
    lengths = torch.sqrt(
        gradients.square().sum(dim=2).mean(dim=1) + 1e-8
    )
    updated_mean = running_mean.lerp(lengths.mean().detach(), decay)
    return (lengths - updated_mean).square().mean(), updated_mean


def ema_decay(batch_size):
    """Convert an image-count half-life into this update's EMA decay."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    half_life_images = EMA_HALF_LIFE_KIMG * 1_000
    return 0.5 ** (batch_size / half_life_images)


@torch.no_grad()
def update_ema(averaged_generator, generator, batch_size):
    """Move StyleGAN2's sampling weights toward G after one image batch."""
    decay = ema_decay(batch_size)
    for averaged_parameter, parameter in zip(
        averaged_generator.parameters(),
        generator.parameters(),
        strict=True,
    ):
        averaged_parameter.lerp_(parameter, 1 - decay)
    for averaged_buffer, buffer in zip(
        averaged_generator.buffers(),
        generator.buffers(),
        strict=True,
    ):
        averaged_buffer.copy_(buffer)


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    path_mean,
    global_step,
    device,
    progress_bar=None,
):
    """Train one epoch with separate main and lazy-regularization passes."""
    metric_sums = torch.zeros(len(METRIC_NAMES), device=device)
    num_examples = 0

    for real, _ in loader:
        real = real.to(device, non_blocking=True)
        batch_size = real.shape[0]

        # Main discriminator pass: non-saturating logistic objective.
        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
        )
        with torch.no_grad():
            fake = generator(z, mixing_z=mixing_z)
        real_scores = discriminator(real)
        fake_scores = discriminator(fake)
        loss_d_main = F.softplus(-real_scores).mean()
        loss_d_main = loss_d_main + F.softplus(fake_scores).mean()
        optimizer_d.zero_grad(set_to_none=True)
        loss_d_main.backward()
        optimizer_d.step()

        # Lazy R1 is a separate optimizer pass that shares Adam's state.
        use_r1 = global_step % D_REG_EVERY == 0
        weighted_r1 = real.new_zeros(())
        if use_r1:
            real_for_r1 = real.detach().requires_grad_(True)
            penalty_r1 = r1_penalty(
                discriminator(real_for_r1),
                real_for_r1,
            )
            weighted_r1 = 0.5 * R1_GAMMA * D_REG_EVERY * penalty_r1
            optimizer_d.zero_grad(set_to_none=True)
            weighted_r1.backward()
            optimizer_d.step()
        loss_d = loss_d_main.detach() + weighted_r1.detach()

        discriminator.requires_grad_(False)
        try:
            # Main generator pass: non-saturating logistic objective.
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                device,
            )
            fake = generator(
                z,
                mixing_z=mixing_z,
                update_w_avg=True,
            )
            loss_g_main = F.softplus(-discriminator(fake)).mean()
            optimizer_g.zero_grad(set_to_none=True)
            loss_g_main.backward()
            optimizer_g.step()

            # Lazy path length is also its own optimizer pass.
            use_path = global_step % G_REG_EVERY == 0
            weighted_path = real.new_zeros(())
            if use_path:
                path_batch = max(1, batch_size // PATH_BATCH_SHRINK)
                path_z = torch.randn(
                    path_batch,
                    generator.z_dim,
                    device=device,
                )
                path_images, path_ws = generator(path_z, return_ws=True)
                penalty_path, path_mean = path_length_penalty(
                    path_images,
                    path_ws,
                    path_mean,
                )
                weighted_path = PATH_WEIGHT * G_REG_EVERY * penalty_path
                optimizer_g.zero_grad(set_to_none=True)
                weighted_path.backward()
                optimizer_g.step()
            loss_g = loss_g_main.detach() + weighted_path.detach()
            update_ema(averaged_generator, generator, batch_size)
        finally:
            discriminator.requires_grad_(True)

        metrics = torch.stack(
            [
                loss_d,
                loss_d_main.detach(),
                loss_g,
                loss_g_main.detach(),
                weighted_r1.detach(),
                weighted_path.detach(),
            ]
        )
        metric_sums += metrics * batch_size
        num_examples += batch_size
        global_step += 1

        if progress_bar is not None:
            progress_bar.update(1)

    epoch_metrics = dict(
        zip(
            METRIC_NAMES,
            (value / num_examples for value in metric_sums.tolist()),
            strict=True,
        )
    )
    return epoch_metrics, path_mean, global_step


def save_training_samples(generator, fixed_z, output_path, epoch):
    """Save fixed-z samples with the same layer noise at every checkpoint."""
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            samples = generator(fixed_z, noise_mode="fixed").cpu()
    finally:
        generator.train(was_training)
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=f"StyleGAN2 EMA fixed samples - epoch {epoch:03d}",
    )


def main(resume_from=None):
    if resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    set_seed(SEED)
    device = try_gpu()
    dataset = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    generator = StyleGenerator(**MODEL_CONFIG).to(device)
    discriminator = StyleDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)

    # Each lazy interval adds one extra optimizer step after k main steps.
    g_ratio = G_REG_EVERY / (G_REG_EVERY + 1)
    d_ratio = D_REG_EVERY / (D_REG_EVERY + 1)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE * g_ratio,
        betas=(0.0, 0.99**g_ratio),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )

    session = TrainingSession(
        OUT_DIR,
        total_epochs=NUM_EPOCHS,
        models={
            "online_generator": generator,
            "stylegan2_generator": averaged_generator,
            "discriminator": discriminator,
        },
        optimizers={"generator": optimizer_g, "discriminator": optimizer_d},
        checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
        archive_every_epochs=ARCHIVE_EVERY_EPOCHS,
        metadata={
            "format_version": 7,
            "conditioning": "unconditional",
            "objective": "non_saturating_logistic_r1_path",
            "progressive_training": False,
            "style_mixing_probability": STYLE_MIXING_PROBABILITY,
            "r1_interval": D_REG_EVERY,
            "path_interval": G_REG_EVERY,
            "ema_half_life_kimg": EMA_HALF_LIFE_KIMG,
        },
        model_metadata={
            "online_generator": {
                "model_name": "stylegan2_online_generator",
                "model_config": MODEL_CONFIG,
                "weights": "online",
            },
            "stylegan2_generator": {
                "model_name": "stylegan2",
                "model_config": MODEL_CONFIG,
                "weights": "ema",
            },
            "discriminator": {
                "model_name": "stylegan2_discriminator",
                "model_config": DISCRIMINATOR_CONFIG,
            },
        },
    )
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    start_epoch, state = session.start(
        resume_from,
        reset_output_dir=False,
        initial_state={
            "loss_history": {
                "epoch": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
            "path_mean": torch.zeros(()),
            "global_step": 0,
        },
    )
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)
    path_mean = state["path_mean"].to(device)
    global_step = state["global_step"]

    expected_steps = (start_epoch - 1) * len(loader)
    if global_step != expected_steps:
        raise ValueError("Checkpoint global step is inconsistent with its epoch.")
    if loss_history["epoch"] != list(range(1, start_epoch)):
        raise ValueError("Checkpoint loss history does not match its epoch.")
    if tuple(fixed_z.shape) != (SAMPLES_TO_DISPLAY, Z_DIM):
        raise ValueError("Checkpoint fixed z has an unexpected shape.")
    if path_mean.ndim != 0:
        raise ValueError("Checkpoint path mean must be a scalar.")
    if resume_from is not None:
        print(f"Resumed training from epoch {start_epoch - 1}: {resume_from}")

    planned_steps = NUM_EPOCHS * len(loader)
    with tqdm(
        total=planned_steps,
        initial=global_step,
        desc=f"StyleGAN2 | Epoch {min(start_epoch, NUM_EPOCHS)}/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            progress_bar.set_description(
                f"StyleGAN2 | Epoch {epoch}/{NUM_EPOCHS}",
                refresh=False,
            )
            metrics, path_mean, global_step = train_epoch(
                generator,
                discriminator,
                loader,
                optimizer_g,
                optimizer_d,
                averaged_generator,
                path_mean,
                global_step,
                device,
                progress_bar=progress_bar,
            )
            loss_history["epoch"].append(epoch)
            for name, value in metrics.items():
                loss_history[name].append(value)

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    averaged_generator,
                    fixed_z,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    epoch,
                )

            state["path_mean"] = path_mean.detach().cpu()
            state["global_step"] = global_step
            session.checkpoint(epoch, state)

    if global_step != planned_steps:
        raise RuntimeError("Training ended with an unexpected step count.")
    session.finish()
    save_loss_panels(
        loss_history["epoch"],
        {
            "Discriminator objective": {
                "D total loss": loss_history["loss_d"],
                "D logistic loss": loss_history["loss_d_main"],
            },
            "Generator objective": {
                "G total loss": loss_history["loss_g"],
                "G non-saturating loss": loss_history["loss_g_main"],
            },
            "R1 regularization": {
                "Weighted lazy R1 contribution": loss_history[
                    "weighted_r1"
                ],
            },
            "Path length regularization": {
                "Weighted lazy path contribution": loss_history[
                    "weighted_path"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the full-resolution CIFAR-10 StyleGAN2."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
