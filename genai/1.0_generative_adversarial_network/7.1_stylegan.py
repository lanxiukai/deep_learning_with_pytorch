"""Train a compact, paper-oriented StyleGAN on 128x128 aligned CelebA.

This lesson keeps ProGAN's 4x4-to-128x128 progressive schedule and adds the
original StyleGAN generator ideas:
    - a normalized Z-to-W mapping network and learned constant input;
    - per-layer AdaIN styles and independent stochastic noise;
    - style-mixing regularization and a moving W center for truncation;
    - non-saturating logistic loss with R1 regularization;
    - an image-count-based exponential moving average of G for sampling.

The original model reaches 1024x1024 with wider features and a much longer
schedule.  This single-GPU teaching profile keeps its eight-layer mapping
depth and every algorithmic mechanism, but stops at 128x128 and measures each
phase in thousands of real images (kimg).  Lazy regularization remains
intentionally reserved for StyleGAN2.

Data:
    data/celeba aligned images and official train split, prepared by
    tool_scripts/download_dataset.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/stylegan/training/kimg_*.png: fixed-z/noise EMA samples
    output/stylegan/checkpoints/latest.pth: full recoverable training state
    output/stylegan/checkpoints/epoch_*.pth: phase-boundary state archives
    output/stylegan/stylegan_generator.pth: final EMA generator with w_avg
    output/stylegan/online_generator.pth: final non-averaged generator
    output/stylegan/discriminator.pth: final discriminator and configuration
    output/stylegan/loss_curves.png: logistic objectives and R1 loss

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.1_stylegan.py \
        --resume-from output/stylegan/checkpoints/latest.pth

Training data -- aligned CelebA train split:
Training images:         162,770
Progressive phases:      11 x 400 kimg (88 recoverable training ticks)
Real images shown to D:  4.4 million
Optimizer updates:       96,875 D / 96,875 G (1:1)

Generator:                4.18 M params (plus one EMA copy)
Discriminator:            4.59 M params
Trainable total:          8.77 M params
"""

import argparse
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dl_utils.data.celeba import (
    CELEBA_ALIGNED_CROP_SIZE,
    CelebAAlignedDataset,
)
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.stylegan import (
    StyleGANDiscriminator,
    StyleGANGenerator,
    denormalize,
)
from dl_utils.gan.stylegan_common import RESOLUTIONS
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.session import TrainingSession


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32, 64: 32, 128: 32}
PHASE_KIMG = 400
TICK_KIMG = 50
TICKS_PER_PHASE = PHASE_KIMG // TICK_KIMG
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
# Express smoothing in images rather than optimizer steps: progressive stages
# use different batches, but every 10 kimg always halves old EMA information.
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 4
SAMPLE_EVERY_TICKS = 4
CHECKPOINT_EVERY_TICKS = 4
ARCHIVE_EVERY_TICKS = TICKS_PER_PHASE
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
    "weighted_r1",
)


@dataclass(frozen=True)
class TrainingTick:
    """One recoverable slice of a fixed-resolution progressive phase."""

    resolution: int
    name: str
    tick_index: int
    num_ticks: int
    batch_size: int
    num_batches: int
    phase_batch_start: int
    phase_num_batches: int

    @property
    def num_images(self):
        return self.num_batches * self.batch_size


def build_training_schedule(num_examples):
    """Split every image-count phase into recoverable training ticks."""
    if PHASE_KIMG % TICK_KIMG:
        raise ValueError("PHASE_KIMG must be divisible by TICK_KIMG.")
    ticks = []
    for resolution in RESOLUTIONS:
        names = (
            ("stabilization",)
            if resolution == 4
            else ("fade-in", "stabilization")
        )
        batch_size = BATCH_SIZES[resolution]
        dataset_batches = num_examples // batch_size
        if dataset_batches == 0:
            raise ValueError(
                f"batch size {batch_size} exceeds dataset size {num_examples}"
            )
        target_images = PHASE_KIMG * 1_000
        if target_images % batch_size:
            raise ValueError(
                "phase image budget must be divisible by every batch size"
            )
        phase_num_batches = target_images // batch_size
        base_batches, extra_batches = divmod(
            phase_num_batches,
            TICKS_PER_PHASE,
        )
        for name in names:
            phase_batch_start = 0
            for tick_index in range(TICKS_PER_PHASE):
                num_batches = base_batches + (
                    tick_index < extra_batches
                )
                if num_batches > dataset_batches:
                    raise ValueError(
                        "one training tick exceeds a complete CelebA pass"
                    )
                ticks.append(
                    TrainingTick(
                        resolution=resolution,
                        name=name,
                        tick_index=tick_index,
                        num_ticks=TICKS_PER_PHASE,
                        batch_size=batch_size,
                        num_batches=num_batches,
                        phase_batch_start=phase_batch_start,
                        phase_num_batches=phase_num_batches,
                    )
                )
                phase_batch_start += num_batches
    return tuple(ticks)


def phase_alpha(tick, batch_index):
    """Increase alpha from zero to one over a complete fade-in phase."""
    if tick.name != "fade-in":
        return 1.0
    total_steps = tick.phase_num_batches
    if total_steps == 1:
        return 1.0
    current_step = tick.phase_batch_start + batch_index
    return current_step / (total_steps - 1)


def make_loader(resolution, batch_size, device):
    """Create one aligned CelebA loader for the active resolution."""
    transform = transforms.Compose(
        [
            transforms.CenterCrop(CELEBA_ALIGNED_CROP_SIZE),
            transforms.Resize(
                (resolution, resolution),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = CelebAAlignedDataset(
        DATA_DIR,
        split="train",
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )


def sample_mixing_latents(batch_size, z_dim, device):
    """Usually return two z batches to decorrelate adjacent styles."""
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


def ema_decay(batch_size):
    """Convert an image-count half-life into this update's EMA decay."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    half_life_images = EMA_HALF_LIFE_KIMG * 1_000
    return 0.5 ** (batch_size / half_life_images)


@torch.no_grad()
def update_ema(averaged_generator, generator, batch_size):
    """Move StyleGAN's sampling weights toward G after one image batch."""
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


def train_tick(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    tick,
    device,
    progress_bar=None,
):
    """Train one image-count tick with logistic and R1 objectives."""
    if len(loader) < tick.num_batches:
        raise ValueError("loader is shorter than the scheduled training tick.")

    metric_sums = torch.zeros(len(METRIC_NAMES), device=device)
    num_examples = 0

    batches = islice(loader, tick.num_batches)
    for batch_index, (real_images, _) in enumerate(batches):
        alpha = phase_alpha(tick, batch_index)
        real_images = real_images.to(device, non_blocking=True)
        real_images.requires_grad_(True)
        batch_size = real_images.shape[0]

        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
        )
        with torch.no_grad():
            fake_images = generator(
                z,
                resolution=tick.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
            )
        real_scores = discriminator(
            real_images,
            resolution=tick.resolution,
            alpha=alpha,
        )
        fake_scores = discriminator(
            fake_images,
            resolution=tick.resolution,
            alpha=alpha,
        )
        loss_d_main = F.softplus(-real_scores).mean()
        loss_d_main = loss_d_main + F.softplus(fake_scores).mean()
        weighted_r1 = 0.5 * R1_GAMMA * r1_penalty(
            real_scores,
            real_images,
        )
        loss_d = loss_d_main + weighted_r1
        optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        optimizer_d.step()
        real_images.requires_grad_(False)

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                device,
            )
            fake_images = generator(
                z,
                resolution=tick.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
                update_w_avg=True,
            )
            loss_g = F.softplus(
                -discriminator(
                    fake_images,
                    resolution=tick.resolution,
                    alpha=alpha,
                )
            ).mean()
            loss_g.backward()
            optimizer_g.step()
            update_ema(averaged_generator, generator, batch_size)
        finally:
            discriminator.requires_grad_(True)

        metrics = torch.stack(
            [loss_d, loss_d_main, loss_g, weighted_r1]
        ).detach()
        metric_sums += metrics * batch_size
        num_examples += batch_size

        if progress_bar is not None:
            progress_bar.update(1)

    return dict(
        zip(
            METRIC_NAMES,
            (value / num_examples for value in metric_sums.tolist()),
            strict=True,
        )
    )


def save_training_samples(
    generator,
    fixed_z,
    output_path,
    tick,
    seen_kimg,
    alpha,
):
    """Save fixed-z and fixed-noise samples at the active stage."""
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            samples = torch.cat(
                [
                    generator(
                        z_batch,
                        resolution=tick.resolution,
                        alpha=alpha,
                        noise_mode="fixed",
                    ).cpu()
                    for z_batch in fixed_z.split(SAMPLE_BATCH_SIZE)
                ]
            )
    finally:
        generator.train(was_training)
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=(
            f"StyleGAN EMA fixed samples - {tick.resolution}x"
            f"{tick.resolution} {tick.name} - {seen_kimg:.0f} kimg - "
            f"alpha={alpha:.2f}"
        ),
    )


def main(resume_from=None):
    if resume_from is None:
        reset_dir(str(TRAINING_DIR))
        reset_dir(str(CHECKPOINT_DIR))

    if not (DATA_DIR / "list_eval_partition.csv").is_file():
        raise FileNotFoundError(
            f"CelebA data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    set_seed(SEED)
    device = try_gpu()
    dataset_info = CelebAAlignedDataset(
        DATA_DIR,
        split="train",
    )
    training_schedule = build_training_schedule(len(dataset_info))
    total_ticks = len(training_schedule)
    total_steps = sum(tick.num_batches for tick in training_schedule)

    generator = StyleGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = StyleGANDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )

    schedule_metadata = [asdict(phase) for phase in training_schedule]
    session = TrainingSession(
        OUT_DIR,
        total_epochs=total_ticks,
        models={
            "online_generator": generator,
            "stylegan_generator": averaged_generator,
            "discriminator": discriminator,
        },
        optimizers={"generator": optimizer_g, "discriminator": optimizer_d},
        checkpoint_every_epochs=CHECKPOINT_EVERY_TICKS,
        archive_every_epochs=ARCHIVE_EVERY_TICKS,
        metadata={
            "format_version": 8,
            "conditioning": "unconditional",
            "dataset": "celeba_aligned_train",
            "final_resolution": RESOLUTIONS[-1],
            "objective": "non_saturating_logistic_r1",
            "progressive_training": True,
            "training_schedule": schedule_metadata,
            "phase_kimg": PHASE_KIMG,
            "tick_kimg": TICK_KIMG,
            "style_mixing_probability": STYLE_MIXING_PROBABILITY,
            "ema_half_life_kimg": EMA_HALF_LIFE_KIMG,
        },
        model_metadata={
            "online_generator": {
                "model_name": "stylegan_online_generator",
                "model_config": MODEL_CONFIG,
                "weights": "online",
            },
            "stylegan_generator": {
                "model_name": "stylegan",
                "model_config": MODEL_CONFIG,
                "weights": "ema",
            },
            "discriminator": {
                "model_name": "stylegan_discriminator",
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
                "kimg": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
        },
    )
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)

    expected_kimg = []
    expected_images = 0
    for tick in training_schedule[: start_epoch - 1]:
        expected_images += tick.num_images
        expected_kimg.append(expected_images / 1_000)
    if loss_history["kimg"] != expected_kimg:
        raise ValueError("Checkpoint loss history does not match its tick.")
    if tuple(fixed_z.shape) != (SAMPLES_TO_DISPLAY, Z_DIM):
        raise ValueError("Checkpoint fixed z has an unexpected shape.")
    if resume_from is not None:
        print(f"Resumed training from tick {start_epoch - 1}: {resume_from}")

    completed_ticks = training_schedule[: start_epoch - 1]
    completed_steps = sum(tick.num_batches for tick in completed_ticks)
    seen_images = sum(tick.num_images for tick in completed_ticks)
    active_phase_key = None
    loader = None
    with tqdm(
        total=total_steps,
        initial=completed_steps,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for global_tick, tick in enumerate(
            training_schedule,
            start=1,
        ):
            if global_tick < start_epoch:
                continue
            phase_key = (tick.resolution, tick.name)
            if phase_key != active_phase_key:
                loader = make_loader(
                    tick.resolution,
                    tick.batch_size,
                    device,
                )
                active_phase_key = phase_key
            assert loader is not None

            progress_bar.set_description(
                f"StyleGAN | {tick.resolution}x{tick.resolution} "
                f"{tick.name} | Tick {global_tick}/{total_ticks}",
                refresh=False,
            )
            metrics = train_tick(
                generator,
                discriminator,
                loader,
                optimizer_g,
                optimizer_d,
                averaged_generator,
                tick,
                device,
                progress_bar=progress_bar,
            )
            seen_images += tick.num_images
            seen_kimg = seen_images / 1_000
            loss_history["kimg"].append(seen_kimg)
            for name, value in metrics.items():
                loss_history[name].append(value)

            alpha = phase_alpha(tick, tick.num_batches - 1)
            should_sample = (
                tick.tick_index == 0
                or global_tick % SAMPLE_EVERY_TICKS == 0
                or global_tick == total_ticks
            )
            if should_sample:
                save_training_samples(
                    averaged_generator,
                    fixed_z,
                    TRAINING_DIR / f"kimg_{round(seen_kimg):05d}.png",
                    tick,
                    seen_kimg,
                    alpha,
                )

            session.checkpoint(global_tick, state)

    session.finish()
    save_loss_panels(
        loss_history["kimg"],
        {
            "Logistic discriminator objective": {
                "D total": loss_history["loss_d"],
                "D logistic": loss_history["loss_d_main"],
            },
            "Non-saturating generator objective": {
                "G non-saturating": loss_history["loss_g"],
            },
            "R1 regularization": {
                "Weighted R1 contribution": loss_history["weighted_r1"],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive 128x128 CelebA StyleGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
