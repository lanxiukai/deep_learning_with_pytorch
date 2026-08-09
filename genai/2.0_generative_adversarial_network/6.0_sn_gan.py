"""Train a paper-oriented class-conditional SN-GAN on CIFAR-10.

This first lesson establishes the shared baseline for the next two scripts:
    - a CIFAR ResNet discriminator with two downsampling blocks followed by
      two same-resolution blocks;
    - spectral normalization on convolutional and linear maps, plus the
      discriminator projection embedding; generator class lookup tables are
      the deliberate exception;
    - projection conditioning in the discriminator;
    - block-local class embeddings for conditional BatchNorm in the generator;
    - a conservative whole-z reinjection into each conditional BatchNorm;
    - hinge adversarial losses.

The model intentionally has no self-attention and keeps the SN-GAN hinge and
projection-discriminator algorithm intact. The z reinjection is deliberately
smaller in scope than BigGAN: every block sees the same complete z, while the
later lesson introduces shared class embeddings and hierarchical latent
chunks. CIFAR-10 remains at its native 32x32 resolution.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/sn_gan/training/epoch_*.png: EMA fixed class samples
    output/sn_gan/checkpoints/latest.pth: recoverable full training state
    output/sn_gan/checkpoints/epoch_*.pth: sparse full-state archives
    output/sn_gan/generator.pth: final EMA generator
    output/sn_gan/generator_last.pth: final non-EMA generator
    output/sn_gan/discriminator.pth: final discriminator
    output/sn_gan/training_metrics.csv: raw per-epoch training diagnostics
    output/sn_gan/evaluation_metrics.csv: aggregate diversity diagnostics
    output/sn_gan/evaluation_by_class.csv: per-class diversity diagnostics
    output/sn_gan/loss_curves.png: separate D and G loss panels
    output/sn_gan/mechanism_diagnostics.png: score, hinge, and feature health
    output/sn_gan/diversity_diagnostics.png: collapse-sensitive evaluation
    output/sn_gan/training_cost.png: throughput and optional CUDA memory

The diversity diagnostics are pixel/pooled-feature sanity checks, not FID or
KID. They are saved so collapse is detectable without optional pretrained
feature extractors or network downloads.

Training data — CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)

Generator:                3.99 M params
Discriminator:            1.04 M params
Total:                    5.03 M params
"""

import copy
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
    CIFAR10_CLASS_NAMES,
    ProjectionSNDiscriminator,
    SNGenerator,
    class_conditional_diversity_metrics,
    class_conditional_reference_metrics,
    conditional_effect_metrics,
    count_spectral_norm_layers,
    cyclically_mismatched_labels,
    discriminator_diagnostic_sums,
    discriminator_hinge_loss,
    generator_hinge_loss,
    make_fixed_class_latent_grid,
    update_ema,
)
from dl_utils.plot.figures import Animator, save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.training.metrics import save_metrics_csv
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sn_gan"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 120
GENERATOR_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 32
IMAGE_SIZE = 32
# Widen the conditioning path without changing its block-local structure.
CONDITION_DIM = 64
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 2e-4
EMA_DECAY = 0.999
SAMPLE_EVERY_EPOCHS = 5
EVALUATE_EVERY_EPOCHS = 5
EVALUATION_SAMPLES_PER_CLASS = 128
EVALUATION_BATCH_SIZE = 64
CHECKPOINT_EVERY_EPOCHS = 5
ARCHIVE_EVERY_EPOCHS = 20
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": GENERATOR_BASE_CHANNELS,
    "condition_dim": CONDITION_DIM,
    "image_size": IMAGE_SIZE,
    "latent_reinjection": True,
}

DISCRIMINATOR_CONFIG = {
    "num_classes": NUM_CLASSES,
    "base_channels": DISCRIMINATOR_BASE_CHANNELS,
}

TRAINING_CONFIG = {
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "drop_last": True,
    "random_horizontal_flip": True,
    "discriminator_updates_per_generator": 1,
    "generator_lr": GENERATOR_LR,
    "discriminator_lr": DISCRIMINATOR_LR,
    "adam_betas": (0.0, 0.9),
    "ema_decay": EMA_DECAY,
    "sample_every_epochs": SAMPLE_EVERY_EPOCHS,
    "evaluate_every_epochs": EVALUATE_EVERY_EPOCHS,
    "evaluation_samples_per_class": EVALUATION_SAMPLES_PER_CLASS,
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


def collect_real_class_samples(dataset, samples_per_class):
    """Collect deterministic, unaugmented CIFAR references by class."""
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2.")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    class_images = [[] for _ in range(NUM_CLASSES)]
    for raw_image, label in zip(dataset.data, dataset.targets, strict=True):
        if len(class_images[label]) < samples_per_class:
            class_images[label].append(transform(raw_image))
        if all(len(images) == samples_per_class for images in class_images):
            break
    if not all(len(images) == samples_per_class for images in class_images):
        counts = [len(images) for images in class_images]
        raise ValueError(f"Insufficient evaluation samples by class: {counts}.")
    return torch.stack([torch.stack(images) for images in class_images])


@torch.inference_mode()
def generate_evaluation_images(
    generator,
    noise,
    labels,
    samples_per_class,
    batch_size,
):
    """Generate a class-major tensor while preserving module train/eval mode."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    was_training = generator.training
    generator.eval()
    try:
        batches = [
            generator(
                noise[start : start + batch_size],
                labels[start : start + batch_size],
            )
            .float()
            .cpu()
            for start in range(0, len(labels), batch_size)
        ]
    finally:
        generator.train(was_training)
    images = torch.cat(batches)
    return images.reshape(NUM_CLASSES, samples_per_class, *images.shape[1:])


def evaluate_generator(
    generator,
    evaluation_noise,
    evaluation_labels,
    real_images,
    real_diversity,
):
    """Return collapse-sensitive aggregate and per-class generator metrics."""
    generated_images = generate_evaluation_images(
        generator,
        evaluation_noise,
        evaluation_labels,
        EVALUATION_SAMPLES_PER_CLASS,
        EVALUATION_BATCH_SIZE,
    )
    generated_diversity = class_conditional_diversity_metrics(
        generated_images
    )
    controlled_effects = conditional_effect_metrics(generated_images)
    reference_metrics = class_conditional_reference_metrics(
        generated_images, real_images
    )

    aggregate = {
        "generated_pairwise_rmse": generated_diversity["pairwise_rmse"],
        "real_pairwise_rmse": real_diversity["pairwise_rmse"],
        "pairwise_rmse_real_ratio": (
            generated_diversity["pairwise_rmse"]
            / max(real_diversity["pairwise_rmse"], 1e-12)
        ),
        "generated_effective_rank_8x8": generated_diversity[
            "pooled_effective_rank"
        ],
        "real_effective_rank_8x8": real_diversity[
            "pooled_effective_rank"
        ],
        "effective_rank_real_ratio": (
            generated_diversity["pooled_effective_rank"]
            / max(real_diversity["pooled_effective_rank"], 1e-12)
        ),
        "generated_class_centroid_within_ratio": generated_diversity[
            "class_centroid_to_within_ratio"
        ],
        "real_class_centroid_within_ratio": real_diversity[
            "class_centroid_to_within_ratio"
        ],
        **controlled_effects,
        **{
            name: value
            for name, value in reference_metrics.items()
            if not name.endswith("_per_class")
        },
    }
    per_class = [
        {
            "class_index": class_index,
            "generated_pairwise_rmse": generated_diversity[
                "pairwise_rmse_per_class"
            ][class_index],
            "real_pairwise_rmse": real_diversity[
                "pairwise_rmse_per_class"
            ][class_index],
            "generated_effective_rank_8x8": generated_diversity[
                "pooled_effective_rank_per_class"
            ][class_index],
            "real_effective_rank_8x8": real_diversity[
                "pooled_effective_rank_per_class"
            ][class_index],
            "pooled_frechet_distance_8x8": reference_metrics[
                "pooled_frechet_distance_per_class"
            ][class_index],
            "generated_to_real_nearest_rmse_8x8": reference_metrics[
                "generated_to_real_nearest_rmse_per_class"
            ][class_index],
            "real_to_generated_nearest_rmse_8x8": reference_metrics[
                "real_to_generated_nearest_rmse_per_class"
            ][class_index],
        }
        for class_index in range(NUM_CLASSES)
    ]
    return aggregate, per_class


def save_atomic(payload, path):
    """Write a torch checkpoint through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def make_training_checkpoint(
    epoch,
    generator,
    generator_ema,
    discriminator,
    opt_g,
    opt_d,
    parameter_counts,
    sn_layer_counts,
    training_history,
    evaluation_history,
    evaluation_by_class_history,
):
    """Build a complete epoch-boundary state for recovery and inspection."""
    rng_state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return {
        "format_version": 2,
        "model_name": "sn_gan",
        "epoch": epoch,
        "model_config": MODEL_CONFIG,
        "discriminator_config": DISCRIMINATOR_CONFIG,
        "training_config": TRAINING_CONFIG,
        "parameter_counts": parameter_counts,
        "sn_layer_counts": sn_layer_counts,
        "generator_state_dict": generator.state_dict(),
        "generator_ema_state_dict": generator_ema.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "training_metrics": training_history,
        "evaluation_metrics": evaluation_history,
        "evaluation_by_class": evaluation_by_class_history,
        "rng_state": rng_state,
    }


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    generator_ema,
    device,
    progress_bar=None,
):
    """Train one epoch and report batch progress."""
    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    loss_sums = torch.zeros(2, device=device)
    diagnostic_sums = torch.zeros(9, device=device)
    num_examples = 0

    for real, labels in loop:
        # real shape: (B, 3, H, W), labels shape: (B,)
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = real.shape[0]
        mismatched_labels = cyclically_mismatched_labels(
            labels, NUM_CLASSES
        )

        noise = torch.randn(batch_size, Z_DIM, device=device)
        with torch.no_grad():
            fake = generator(noise, labels)
        (
            real_scores,
            correct_projection,
            mismatched_projection,
            real_features,
        ) = discriminator.forward_with_projection_diagnostics(
            real,
            labels,
            mismatched_labels,
            return_features=True,
        )
        fake_scores = discriminator(fake, labels)
        loss_d = discriminator_hinge_loss(real_scores, fake_scores)
        diagnostic_sums += discriminator_diagnostic_sums(
            real_scores,
            fake_scores,
            correct_projection,
            mismatched_projection,
            real_features,
        )
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        sampled_labels = torch.randint(
            NUM_CLASSES, (batch_size,), device=device
        )
        noise = torch.randn(batch_size, Z_DIM, device=device)

        discriminator.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            fake = generator(noise, sampled_labels)
            loss_g = generator_hinge_loss(
                discriminator(fake, sampled_labels)
            )
            loss_g.backward()
            opt_g.step()
            update_ema(generator_ema, generator, EMA_DECAY)
        finally:
            discriminator.requires_grad_(True)

        batch_losses = torch.stack([loss_d, loss_g]).detach()
        loss_sums += batch_losses * batch_size
        num_examples += batch_size
        batch_loss_values = batch_losses.tolist()
        loss_d_value, loss_g_value = batch_loss_values

        if progress_bar is None:
            loop.set_postfix(
                loss_D=loss_d_value,
                loss_G=loss_g_value,
                refresh=False,
            )
        else:
            progress_bar.set_postfix(
                loss_D=loss_d_value,
                loss_G=loss_g_value,
                refresh=False,
            )
            progress_bar.update(1)

    epoch_sums = torch.cat([loss_sums, diagnostic_sums]).tolist()
    averages = [value / num_examples for value in epoch_sums]
    real_score_variance = max(averages[7] - averages[5] ** 2, 0.0)
    fake_score_variance = max(averages[8] - averages[6] ** 2, 0.0)
    return (
        *averages[:7],
        math.sqrt(real_score_variance),
        math.sqrt(fake_score_variance),
        *averages[9:],
    )


def main():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    reset_dir(str(TRAINING_DIR))
    reset_dir(str(CHECKPOINT_DIR))
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
    real_evaluation_images = collect_real_class_samples(
        dataset, EVALUATION_SAMPLES_PER_CLASS
    )
    real_diversity = class_conditional_diversity_metrics(
        real_evaluation_images
    )

    generator = SNGenerator(**MODEL_CONFIG).to(device)
    discriminator = ProjectionSNDiscriminator(**DISCRIMINATOR_CONFIG).to(
        device
    )
    generator_ema = copy.deepcopy(generator).eval().requires_grad_(False)
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
    print(
        "Real-data diversity reference: "
        f"pairwise RMSE={real_diversity['pairwise_rmse']:.4f}, "
        f"8x8 rank={real_diversity['pooled_effective_rank']:.2f}"
    )
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=GENERATOR_LR,
        betas=(0.0, 0.9),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=DISCRIMINATOR_LR,
        betas=(0.0, 0.9),
    )

    fixed_noise, fixed_labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        SAMPLES_PER_CLASS,
        Z_DIM,
        device,
    )
    evaluation_noise, evaluation_labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        EVALUATION_SAMPLES_PER_CLASS,
        Z_DIM,
        device,
    )
    training_history = {}
    evaluation_history = {}
    evaluation_by_class_history = {}
    samples_per_epoch = len(loader) * BATCH_SIZE
    animator = Animator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, NUM_EPOCHS],
        legend=["D hinge loss", "G adversarial loss"],
        figsize=(6, 4),
    )
    neutral_margin_epochs = 0

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
                projection_score_gap,
                real_score_mean,
                fake_score_mean,
                real_score_std,
                fake_score_std,
                real_feature_active_fraction,
                real_feature_rms,
            ) = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                generator_ema,
                device,
                progress_bar=progress_bar,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start

            training_row = {
                "epoch": epoch,
                "discriminator_hinge_loss": loss_d,
                "generator_adversarial_loss": loss_g,
                "real_hinge_active_fraction": real_hinge_active,
                "fake_hinge_active_fraction": fake_hinge_active,
                "projection_score_gap": projection_score_gap,
                "real_score_mean": real_score_mean,
                "fake_score_mean": fake_score_mean,
                "real_fake_score_gap": real_score_mean - fake_score_mean,
                "real_score_std": real_score_std,
                "fake_score_std": fake_score_std,
                "real_feature_active_fraction": (
                    real_feature_active_fraction
                ),
                "real_feature_rms": real_feature_rms,
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
                training_history, OUT_DIR / "training_metrics.csv"
            )

            neutral_margin = (
                abs(loss_d - 2.0) < 0.02
                and real_hinge_active > 0.98
                and fake_hinge_active > 0.98
                and abs(real_score_mean - fake_score_mean) < 0.05
                and abs(projection_score_gap) < 0.05
            )
            neutral_margin_epochs = (
                neutral_margin_epochs + 1 if neutral_margin else 0
            )
            if neutral_margin_epochs == 3:
                tqdm.write(
                    "Warning: D has remained near the neutral hinge solution "
                    "for three epochs; inspect training_metrics.csv and "
                    "diversity diagnostics."
                )

            latest_evaluation = None
            if epoch == 1 or epoch % EVALUATE_EVERY_EPOCHS == 0:
                latest_evaluation, per_class_evaluation = evaluate_generator(
                    generator_ema,
                    evaluation_noise,
                    evaluation_labels,
                    real_evaluation_images,
                    real_diversity,
                )
                append_metric_row(
                    evaluation_history,
                    {"epoch": epoch, **latest_evaluation},
                )
                for class_metrics in per_class_evaluation:
                    append_metric_row(
                        evaluation_by_class_history,
                        {"epoch": epoch, **class_metrics},
                    )
                save_metrics_csv(
                    evaluation_history,
                    OUT_DIR / "evaluation_metrics.csv",
                )
                save_metrics_csv(
                    evaluation_by_class_history,
                    OUT_DIR / "evaluation_by_class.csv",
                )

            animator.add(epoch, (loss_d, loss_g))
            progress_values = dict(
                loss_D=loss_d,
                loss_G=loss_g,
                hinge_real=real_hinge_active,
                hinge_fake=fake_hinge_active,
                projection_gap=projection_score_gap,
            )
            if latest_evaluation is not None:
                progress_values["diversity_ratio"] = latest_evaluation[
                    "pairwise_rmse_real_ratio"
                ]
                progress_values["rank_ratio"] = latest_evaluation[
                    "effective_rank_real_ratio"
                ]
            progress_bar.set_postfix(**progress_values, refresh=False)

            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    generator_ema,
                    fixed_noise,
                    fixed_labels,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    class_names=CIFAR10_CLASS_NAMES[:NUM_CLASSES],
                    title=(
                        "SN-GAN EMA fixed class samples - "
                        f"epoch {epoch:03d}"
                    ),
                    shared_latents_across_classes=True,
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
                    generator_ema,
                    discriminator,
                    opt_g,
                    opt_d,
                    parameter_counts,
                    sn_layer_counts,
                    training_history,
                    evaluation_history,
                    evaluation_by_class_history,
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
    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    final_metadata = {
        "format_version": 2,
        "model_name": "sn_gan",
        "epoch": NUM_EPOCHS,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "parameter_counts": parameter_counts,
        "sn_layer_counts": sn_layer_counts,
        "final_evaluation": {
            name: values[-1]
            for name, values in evaluation_history.items()
        },
    }
    save_atomic(
        {
            **final_metadata,
            "weights": "ema",
            "state_dict": generator_ema.state_dict(),
        },
        OUT_DIR / "generator.pth",
    )
    save_atomic(
        {
            **final_metadata,
            "weights": "last",
            "state_dict": generator.state_dict(),
        },
        OUT_DIR / "generator_last.pth",
    )
    save_atomic(
        {
            "format_version": 2,
            "model_name": "sn_gan_discriminator",
            "epoch": NUM_EPOCHS,
            "model_config": DISCRIMINATOR_CONFIG,
            "training_config": TRAINING_CONFIG,
            "parameter_count": parameter_counts["discriminator"],
            "sn_layer_count": sn_layer_counts["discriminator"],
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
                "Real: D(x, y) < 1": training_history[
                    "real_hinge_active_fraction"
                ],
                "Fake: D(G(z), y) > -1": training_history[
                    "fake_hinge_active_fraction"
                ],
            },
            "Projection score gap": {
                "Correct - mismatched label": training_history[
                    "projection_score_gap"
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
            "Controlled conditional effects": {
                "Change label at fixed z": evaluation_history[
                    "label_effect_rmse"
                ],
                "Change z at fixed label": evaluation_history[
                    "latent_effect_rmse"
                ],
            },
            "Label / latent effect ratio": {
                "Label-to-latent ratio": evaluation_history[
                    "label_to_latent_effect_ratio"
                ],
            },
            "Class-centroid / within-class ratio": {
                "Generated": evaluation_history[
                    "generated_class_centroid_within_ratio"
                ],
                "Real": evaluation_history[
                    "real_class_centroid_within_ratio"
                ],
            },
            "Pooled Fréchet diagnostics (lower is better)": {
                "All classes": evaluation_history[
                    "pooled_frechet_distance"
                ],
                "Mean within class": evaluation_history[
                    "class_conditional_pooled_frechet_distance"
                ],
            },
            "Pooled nearest-neighbor RMSE (lower is better)": {
                "Generated to real (fidelity)": evaluation_history[
                    "generated_to_real_nearest_rmse"
                ],
                "Real to generated (coverage)": evaluation_history[
                    "real_to_generated_nearest_rmse"
                ],
            },
        },
        OUT_DIR / "diversity_diagnostics.png",
        ylabel="value",
    )
    cost_panels = {
        "Training throughput (samples/s)": {
            "Training throughput": training_history[
                "training_samples_per_second"
            ],
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
