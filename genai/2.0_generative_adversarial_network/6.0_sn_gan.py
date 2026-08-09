"""Train an unconditional, paper-oriented SN-GAN on CIFAR-10.

This lesson follows the CIFAR ResNet experiment in *Spectral Normalization for
Generative Adversarial Networks*:
    - CIFAR-10 is modeled without using its class labels;
    - z has 128 dimensions and initializes a 4x4x256 feature map;
    - G contains three constant-width upsampling residual blocks;
    - D contains two downsampling and two same-resolution residual blocks;
    - spectral normalization is applied to every learned map in D, but not G;
    - the hinge objective and an exact five D updates per G update use the
      paper's ResNet training setting;
    - D uses batches of 64 real/fake images while G uses batches of 128;
    - real uint8 pixels use uniform dequantization before entering D.

Projection discrimination and conditional BatchNorm are intentionally absent.
They first appear in ``6.1_sagan.py``, whose original paper explicitly uses
both mechanisms.  Fixed-noise monitoring, lightweight evaluation, and full
checkpoints are teaching conveniences and do not alter the adversarial updates.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/sn_gan/training/epoch_*.png: fixed-z unconditional samples
    output/sn_gan/checkpoints/latest.pth: full recoverable training state
    output/sn_gan/checkpoints/epoch_*.pth: sparse full-state archives
    output/sn_gan/generator.pth: final generator and configuration
    output/sn_gan/discriminator.pth: final discriminator and configuration
    output/sn_gan/training_metrics.csv: per-epoch optimization diagnostics
    output/sn_gan/evaluation_metrics.csv: real-referenced collapse checks
    output/sn_gan/loss_curves.png: separate D and G loss panels
    output/sn_gan/mechanism_diagnostics.png: hinge, score, and feature health
    output/sn_gan/diversity_diagnostics.png: diversity and coverage checks
    output/sn_gan/training_cost.png: throughput and optional CUDA memory

The evaluation statistics use pixels pooled to 8x8.  They are inexpensive
sanity checks, not the paper's Inception-feature FID or Inception Score.

Training data -- CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)
Training epochs:         650
Optimizer updates:       507,650 D / 101,530 G (exactly 5:1)

Generator:                4.28 M params
Discriminator:            1.05 M params
Total:                    5.33 M params
"""

import math
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.sn_gan import (
    SNDiscriminator,
    SNGenerator,
    count_spectral_norm_layers,
    discriminator_diagnostic_sums,
    discriminator_hinge_loss,
    evaluate_unconditional_images,
    generator_hinge_loss,
    uniform_dequantize_uint8,
)
from dl_utils.plot.figures import Animator, save_loss_panels
from dl_utils.plot.images import save_image_row_grid
from dl_utils.training.metrics import save_metrics_csv
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sn_gan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

NUM_EPOCHS = 650
DISCRIMINATOR_BATCH_SIZE = 64
GENERATOR_BATCH_SIZE = 128
NUM_WORKERS = 8
Z_DIM = 128
GENERATOR_BASE_CHANNELS = 256
DISCRIMINATOR_BASE_CHANNELS = 128
IMAGE_SIZE = 32
LEARNING_RATE = 2e-4
DISCRIMINATOR_UPDATES_PER_GENERATOR = 5
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY_EPOCHS = 10
EVALUATE_EVERY_EPOCHS = 10
EVALUATION_SAMPLES = 512
EVALUATION_BATCH_SIZE = 128
CHECKPOINT_EVERY_EPOCHS = 10
ARCHIVE_EVERY_EPOCHS = 50
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "base_channels": GENERATOR_BASE_CHANNELS,
    "image_size": IMAGE_SIZE,
}

DISCRIMINATOR_CONFIG = {
    "base_channels": DISCRIMINATOR_BASE_CHANNELS,
}

TRAINING_CONFIG = {
    "num_epochs": NUM_EPOCHS,
    "discriminator_batch_size": DISCRIMINATOR_BATCH_SIZE,
    "generator_batch_size": GENERATOR_BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "drop_last": True,
    "random_horizontal_flip": False,
    "uniform_dequantization": True,
    "input_scaling": "(uint8 + U[0, 1)) / 128 - 1",
    "discriminator_updates_per_generator": (
        DISCRIMINATOR_UPDATES_PER_GENERATOR
    ),
    "learning_rate": LEARNING_RATE,
    "adam_betas": (0.0, 0.9),
    "sample_every_epochs": SAMPLE_EVERY_EPOCHS,
    "evaluate_every_epochs": EVALUATE_EVERY_EPOCHS,
    "evaluation_samples": EVALUATION_SAMPLES,
    "evaluation_batch_size": EVALUATION_BATCH_SIZE,
    "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
    "archive_every_epochs": ARCHIVE_EVERY_EPOCHS,
    "seed": SEED,
}


def append_metric_row(history, row):
    """Append one complete scalar row to a metric-history mapping."""
    if history and set(history) != set(row):
        missing = sorted(set(history) - set(row))
        extra = sorted(set(row) - set(history))
        raise ValueError(
            f"Metric columns changed; missing={missing}, extra={extra}."
        )
    for name, value in row.items():
        history.setdefault(name, []).append(value)


def collect_real_evaluation_images(dataset, num_samples, seed):
    """Select deterministic CIFAR references without training-time noise."""
    if not 2 <= num_samples <= len(dataset):
        raise ValueError(
            f"num_samples must be in [2, {len(dataset)}], got {num_samples}."
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return torch.stack(
        [transform(dataset.data[index]) for index in indices.tolist()]
    )


@torch.inference_mode()
def generate_in_batches(generator, noise, batch_size):
    """Generate a CPU image tensor without changing the caller's mode."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    was_training = generator.training
    generator.eval()
    try:
        return torch.cat(
            [
                generator(noise[start : start + batch_size]).float().cpu()
                for start in range(0, len(noise), batch_size)
            ]
        )
    finally:
        generator.train(was_training)


def evaluate_generator(generator, evaluation_noise, real_images):
    """Generate fixed samples and compare them with fixed real references."""
    generated_images = generate_in_batches(
        generator,
        evaluation_noise,
        EVALUATION_BATCH_SIZE,
    )
    return evaluate_unconditional_images(generated_images, real_images)


def save_fixed_samples(generator, fixed_noise, path, epoch):
    """Save an 8x8 fixed-noise grid while preserving training mode."""
    samples = generate_in_batches(
        generator,
        fixed_noise,
        EVALUATION_BATCH_SIZE,
    )
    rows = samples.reshape(
        -1,
        SAMPLE_GRID_COLUMNS,
        *samples.shape[1:],
    )
    row_labels = [
        (
            f"z {row_index * SAMPLE_GRID_COLUMNS + 1:02d}-"
            f"{(row_index + 1) * SAMPLE_GRID_COLUMNS:02d}"
        )
        for row_index in range(len(rows))
    ]
    save_image_row_grid(
        rows,
        row_labels,
        path,
        title=f"Unconditional SN-GAN fixed-z samples - epoch {epoch:03d}",
        column_labels=[
            f"Sample {index + 1}" for index in range(SAMPLE_GRID_COLUMNS)
        ],
        dpi=200,
    )


def save_atomic(payload, path):
    """Write a torch checkpoint through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def make_training_checkpoint(
    epoch,
    generator,
    discriminator,
    opt_g,
    opt_d,
    parameter_counts,
    sn_layer_counts,
    training_history,
    evaluation_history,
    training_progress,
):
    """Build a complete epoch-boundary state for recovery and inspection."""
    rng_state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return {
        "format_version": 4,
        "model_name": "sn_gan",
        "conditioning": "unconditional",
        "epoch": epoch,
        "model_config": MODEL_CONFIG,
        "discriminator_config": DISCRIMINATOR_CONFIG,
        "training_config": TRAINING_CONFIG,
        "parameter_counts": parameter_counts,
        "sn_layer_counts": sn_layer_counts,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "training_metrics": training_history,
        "evaluation_metrics": evaluation_history,
        "training_progress": dict(training_progress),
        "rng_state": rng_state,
    }


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    device,
    discriminator_updates_since_generator,
    progress_bar=None,
):
    """Train one epoch while carrying the exact five-to-one update phase."""
    if not (
        0
        <= discriminator_updates_since_generator
        < DISCRIMINATOR_UPDATES_PER_GENERATOR
    ):
        raise ValueError(
            "discriminator_updates_since_generator must be in "
            f"[0, {DISCRIMINATOR_UPDATES_PER_GENERATOR})."
        )
    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    discriminator_loss_sum = torch.zeros((), device=device)
    generator_loss_sum = torch.zeros((), device=device)
    diagnostic_sums = torch.zeros(8, device=device)
    discriminator_examples = 0
    generator_examples = 0
    generator_updates = 0
    last_generator_loss = None

    for real, _ in loop:
        real = real.to(device, non_blocking=True)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, Z_DIM, device=device)
        with torch.no_grad():
            fake = generator(noise)
        real_scores, real_features = discriminator.forward_with_features(real)
        fake_scores = discriminator(fake)
        loss_d = discriminator_hinge_loss(real_scores, fake_scores)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        discriminator_loss_sum += loss_d.detach() * batch_size
        diagnostic_sums += discriminator_diagnostic_sums(
            real_scores,
            fake_scores,
            real_features,
        )
        discriminator_examples += batch_size
        discriminator_updates_since_generator += 1

        should_update_generator = (
            discriminator_updates_since_generator
            == DISCRIMINATOR_UPDATES_PER_GENERATOR
        )
        if should_update_generator:
            noise = torch.randn(
                GENERATOR_BATCH_SIZE,
                Z_DIM,
                device=device,
            )
            discriminator.requires_grad_(False)
            try:
                opt_g.zero_grad(set_to_none=True)
                fake = generator(noise)
                loss_g = generator_hinge_loss(discriminator(fake))
                loss_g.backward()
                opt_g.step()
            finally:
                discriminator.requires_grad_(True)

            generator_loss_sum += loss_g.detach() * GENERATOR_BATCH_SIZE
            generator_examples += GENERATOR_BATCH_SIZE
            generator_updates += 1
            discriminator_updates_since_generator = 0
            last_generator_loss = loss_g.detach().item()

        postfix = {"loss_D": loss_d.detach().item()}
        if last_generator_loss is not None:
            postfix["loss_G"] = last_generator_loss
        if progress_bar is None:
            loop.set_postfix(**postfix, refresh=False)
        else:
            progress_bar.set_postfix(**postfix, refresh=False)
            progress_bar.update(1)

    if generator_examples == 0:
        raise RuntimeError("No generator update occurred in this epoch.")

    averages = (diagnostic_sums / discriminator_examples).tolist()
    real_score_variance = max(averages[4] - averages[2] ** 2, 0.0)
    fake_score_variance = max(averages[5] - averages[3] ** 2, 0.0)
    return (
        (discriminator_loss_sum / discriminator_examples).item(),
        (generator_loss_sum / generator_examples).item(),
        averages[0],
        averages[1],
        averages[2],
        averages[3],
        math.sqrt(real_score_variance),
        math.sqrt(fake_score_variance),
        averages[6],
        averages[7],
        generator_updates,
        discriminator_updates_since_generator,
    )


def main():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    reset_dir(str(OUT_DIR))
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = try_gpu()

    dataset = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(uniform_dequantize_uint8),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=DISCRIMINATOR_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )
    real_evaluation_images = collect_real_evaluation_images(
        dataset,
        EVALUATION_SAMPLES,
        SEED,
    )

    generator = SNGenerator(**MODEL_CONFIG).to(device)
    discriminator = SNDiscriminator(**DISCRIMINATOR_CONFIG).to(device)
    parameter_counts = {
        "generator": sum(
            parameter.numel() for parameter in generator.parameters()
        ),
        "discriminator": sum(
            parameter.numel() for parameter in discriminator.parameters()
        ),
    }
    sn_layer_counts = {
        "generator": count_spectral_norm_layers(generator),
        "discriminator": count_spectral_norm_layers(discriminator),
    }
    if sn_layer_counts["generator"] != 0:
        raise RuntimeError("SN-GAN must not apply spectral norm to G.")
    if sn_layer_counts["discriminator"] == 0:
        raise RuntimeError("SN-GAN must apply spectral norm to D.")
    print(
        "Spectral-normalized layers: "
        f"G={sn_layer_counts['generator']}, "
        f"D={sn_layer_counts['discriminator']}"
    )
    print(
        "Trainable parameters: "
        f"G={parameter_counts['generator'] / 1e6:.2f} M, "
        f"D={parameter_counts['discriminator'] / 1e6:.2f} M, "
        f"total={sum(parameter_counts.values()) / 1e6:.2f} M"
    )

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )

    fixed_noise = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    evaluation_noise = torch.randn(
        EVALUATION_SAMPLES,
        Z_DIM,
        device=device,
    )
    training_history = {}
    evaluation_history = {}
    samples_per_epoch = len(loader) * DISCRIMINATOR_BATCH_SIZE
    planned_discriminator_updates = NUM_EPOCHS * len(loader)
    planned_generator_updates, incomplete_update_cycle = divmod(
        planned_discriminator_updates,
        DISCRIMINATOR_UPDATES_PER_GENERATOR,
    )
    if incomplete_update_cycle:
        raise RuntimeError(
            "The configured epoch count does not end on an exact 5D:1G "
            "update boundary."
        )
    print(
        "Planned optimizer updates: "
        f"D={planned_discriminator_updates:,}, "
        f"G={planned_generator_updates:,} "
        f"(exactly {DISCRIMINATOR_UPDATES_PER_GENERATOR}:1)"
    )
    animator = Animator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, NUM_EPOCHS],
        legend=["D hinge loss", "G adversarial loss"],
        figsize=(6, 4),
    )
    neutral_margin_epochs = 0
    training_progress = {
        "discriminator_updates": 0,
        "generator_updates": 0,
        "discriminator_updates_since_generator": 0,
    }

    timer = Timer()
    with tqdm(
        total=NUM_EPOCHS * len(loader),
        desc=f"Epoch 1/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, NUM_EPOCHS + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{NUM_EPOCHS}",
                refresh=False,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()

            (
                loss_d,
                loss_g,
                real_hinge_active,
                fake_hinge_active,
                real_score_mean,
                fake_score_mean,
                real_score_std,
                fake_score_std,
                real_feature_active_fraction,
                real_feature_rms,
                generator_updates,
                discriminator_updates_since_generator,
            ) = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                device,
                training_progress[
                    "discriminator_updates_since_generator"
                ],
                progress_bar=progress_bar,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start
            training_progress["discriminator_updates"] += len(loader)
            training_progress["generator_updates"] += generator_updates
            training_progress[
                "discriminator_updates_since_generator"
            ] = discriminator_updates_since_generator
            if training_progress["discriminator_updates"] != (
                DISCRIMINATOR_UPDATES_PER_GENERATOR
                * training_progress["generator_updates"]
                + discriminator_updates_since_generator
            ):
                raise RuntimeError("The 5D:1G update accounting is inconsistent.")

            training_row = {
                "epoch": epoch,
                "discriminator_hinge_loss": loss_d,
                "generator_adversarial_loss": loss_g,
                "real_hinge_active_fraction": real_hinge_active,
                "fake_hinge_active_fraction": fake_hinge_active,
                "real_score_mean": real_score_mean,
                "fake_score_mean": fake_score_mean,
                "real_fake_score_gap": real_score_mean - fake_score_mean,
                "real_score_std": real_score_std,
                "fake_score_std": fake_score_std,
                "real_feature_active_fraction": (
                    real_feature_active_fraction
                ),
                "real_feature_rms": real_feature_rms,
                "generator_updates": generator_updates,
                "cumulative_discriminator_updates": training_progress[
                    "discriminator_updates"
                ],
                "cumulative_generator_updates": training_progress[
                    "generator_updates"
                ],
                "discriminator_updates_since_generator": (
                    discriminator_updates_since_generator
                ),
                "training_samples_per_second": (
                    samples_per_epoch / epoch_seconds
                ),
            }
            if device.type == "cuda":
                training_row["peak_cuda_memory_mib"] = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                )
            append_metric_row(training_history, training_row)
            save_metrics_csv(
                training_history,
                OUT_DIR / "training_metrics.csv",
            )

            neutral_margin = (
                abs(loss_d - 2.0) < 0.02
                and real_hinge_active > 0.98
                and fake_hinge_active > 0.98
                and abs(real_score_mean - fake_score_mean) < 0.05
            )
            neutral_margin_epochs = (
                neutral_margin_epochs + 1 if neutral_margin else 0
            )
            if neutral_margin_epochs == 3:
                tqdm.write(
                    "Warning: D has remained near the neutral hinge solution "
                    "for three epochs; inspect training_metrics.csv and "
                    "diversity_diagnostics.png."
                )

            latest_evaluation = None
            if epoch == 1 or epoch % EVALUATE_EVERY_EPOCHS == 0:
                latest_evaluation = evaluate_generator(
                    generator,
                    evaluation_noise,
                    real_evaluation_images,
                )
                append_metric_row(
                    evaluation_history,
                    {"epoch": epoch, **latest_evaluation},
                )
                save_metrics_csv(
                    evaluation_history,
                    OUT_DIR / "evaluation_metrics.csv",
                )

            animator.add(epoch, (loss_d, loss_g))
            progress_values = {
                "loss_D": loss_d,
                "loss_G": loss_g,
                "hinge_real": real_hinge_active,
                "hinge_fake": fake_hinge_active,
            }
            if latest_evaluation is not None:
                progress_values["diversity_ratio"] = latest_evaluation[
                    "pairwise_rmse_real_ratio"
                ]
                progress_values["rank_ratio"] = latest_evaluation[
                    "effective_rank_real_ratio"
                ]
            progress_bar.set_postfix(**progress_values, refresh=False)

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_fixed_samples(
                    generator,
                    fixed_noise,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    epoch,
                )

            should_checkpoint = (
                epoch == 1
                or epoch % CHECKPOINT_EVERY_EPOCHS == 0
                or epoch == NUM_EPOCHS
            )
            if should_checkpoint:
                checkpoint = make_training_checkpoint(
                    epoch,
                    generator,
                    discriminator,
                    opt_g,
                    opt_d,
                    parameter_counts,
                    sn_layer_counts,
                    training_history,
                    evaluation_history,
                    training_progress,
                )
                save_atomic(checkpoint, CHECKPOINT_DIR / "latest.pth")
                if (
                    epoch % ARCHIVE_EVERY_EPOCHS == 0
                    or epoch == NUM_EPOCHS
                ):
                    save_atomic(
                        checkpoint,
                        CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth",
                    )

    if training_progress != {
        "discriminator_updates": planned_discriminator_updates,
        "generator_updates": planned_generator_updates,
        "discriminator_updates_since_generator": 0,
    }:
        raise RuntimeError("Training ended with unexpected optimizer counts.")
    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    final_metadata = {
        "format_version": 4,
        "model_name": "sn_gan",
        "conditioning": "unconditional",
        "epoch": NUM_EPOCHS,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "parameter_counts": parameter_counts,
        "sn_layer_counts": sn_layer_counts,
        "training_progress": dict(training_progress),
        "final_evaluation": {
            name: values[-1]
            for name, values in evaluation_history.items()
        },
    }
    save_atomic(
        {
            **final_metadata,
            "state_dict": generator.state_dict(),
        },
        OUT_DIR / "generator.pth",
    )
    save_atomic(
        {
            "format_version": 4,
            "model_name": "sn_gan_discriminator",
            "conditioning": "unconditional",
            "epoch": NUM_EPOCHS,
            "model_config": DISCRIMINATOR_CONFIG,
            "training_config": TRAINING_CONFIG,
            "parameter_count": parameter_counts["discriminator"],
            "sn_layer_count": sn_layer_counts["discriminator"],
            "training_progress": dict(training_progress),
            "state_dict": discriminator.state_dict(),
        },
        OUT_DIR / "discriminator.pth",
    )

    loss_steps = training_history["epoch"]
    save_loss_panels(
        loss_steps,
        {
            "Discriminator hinge loss": {
                "D hinge loss": training_history[
                    "discriminator_hinge_loss"
                ],
            },
            "Generator adversarial loss": {
                "G adversarial loss": training_history[
                    "generator_adversarial_loss"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
    )
    save_loss_panels(
        loss_steps,
        {
            "Active hinge-margin fraction": {
                "Real: D(x) < 1": training_history[
                    "real_hinge_active_fraction"
                ],
                "Fake: D(G(z)) > -1": training_history[
                    "fake_hinge_active_fraction"
                ],
            },
            "Discriminator score means": {
                "Real score": training_history["real_score_mean"],
                "Fake score": training_history["fake_score_mean"],
                "Real - fake": training_history["real_fake_score_gap"],
            },
            "Discriminator score standard deviation": {
                "Real score std": training_history["real_score_std"],
                "Fake score std": training_history["fake_score_std"],
            },
            "Real feature health": {
                "Positive fraction": training_history[
                    "real_feature_active_fraction"
                ],
                "RMS magnitude": training_history["real_feature_rms"],
            },
        },
        OUT_DIR / "mechanism_diagnostics.png",
        ylabel="value",
    )
    save_loss_panels(
        evaluation_history["epoch"],
        {
            "Generated / real diversity": {
                "Pairwise RMSE ratio": evaluation_history[
                    "pairwise_rmse_real_ratio"
                ],
                "8x8 effective-rank ratio": evaluation_history[
                    "effective_rank_real_ratio"
                ],
            },
            "Pooled Frechet diagnostic (lower is better)": {
                "8x8 pooled pixels": evaluation_history[
                    "pooled_frechet_distance_8x8"
                ],
            },
            "Pooled nearest-neighbor RMSE (lower is better)": {
                "Generated to real (fidelity)": evaluation_history[
                    "generated_to_real_nearest_rmse_8x8"
                ],
                "Real to generated (coverage)": evaluation_history[
                    "real_to_generated_nearest_rmse_8x8"
                ],
            },
        },
        OUT_DIR / "diversity_diagnostics.png",
        ylabel="value",
    )
    cost_panels = {
        "Training throughput (real samples/s)": {
            "Training throughput": training_history[
                "training_samples_per_second"
            ],
        },
        "Generator updates per epoch": {
            "G updates": training_history["generator_updates"],
        },
    }
    if "peak_cuda_memory_mib" in training_history:
        cost_panels["Peak CUDA memory (MiB)"] = {
            "Peak allocated memory": training_history[
                "peak_cuda_memory_mib"
            ],
        }
    save_loss_panels(
        loss_steps,
        cost_panels,
        OUT_DIR / "training_cost.png",
        ylabel="value",
    )


if __name__ == "__main__":
    main()
