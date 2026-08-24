"""Train a compact, paper-oriented StyleGAN2 on 128x128 aligned CelebA.

Compared with StyleGAN, this lesson presents the main StyleGAN2 increments:
    - per-sample weight modulation/demodulation instead of AdaIN;
    - the original one-convolution 4x4 block and 12 overlapping W slots;
    - skip-connected RGB outputs and full 128x128 training;
    - residual discriminator downsampling;
    - lazy R1 and path-length regularization in separate optimizer passes;
    - lazy-regularization compensation for Adam's learning rate and beta2;
    - style mixing, scalar-strength layer noise, truncation state, and an
      image-count-based G-EMA.

The complete mapping, synthesis, and discriminator models live in
``dl_utils/gan/stylegan2.py``.  This pure-PyTorch single-GPU profile stops at
128x128 and uses compact widths, so it omits fused CUDA kernels, multi-GPU
training, and paper-scale capacity while leaving every new loss explicit.

Data:
    data/celeba aligned images and official train split, prepared by
    tool_scripts/download_dataset.py.

Outputs:
    Fresh runs reset training/ and checkpoints/ before setup; --resume-from
    preserves both directories.
    output/stylegan2/training/kimg_*.png: fixed-z/noise EMA samples by epoch
    output/stylegan2/checkpoints/latest.pth: latest epoch-boundary training state
    output/stylegan2/stylegan2_generator.pth: final EMA generator with w_avg
    output/stylegan2/loss_curves.png: objectives and lazy regularization losses

Resume an interrupted run:
    python genai/1.0_generative_adversarial_network/7.2_stylegan2.py \
        --resume-from output/stylegan2/checkpoints/latest.pth
    Checkpoints with a different epoch metric schema are unsupported.

Training data -- aligned CelebA train split:
Training images:         162,770
Training budget:         5,000 kimg (31 data epochs, last one partial)
Main optimizer updates:  156,250 D / 156,250 G (1:1)

Generator:                4.05 M params (plus one EMA copy)
Discriminator:            4.76 M params
Trainable total:          8.81 M params
"""

import argparse
from copy import deepcopy
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.data.celeba import make_aligned_celeba_loader
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.stylegan2 import (
    StyleDiscriminator,
    StyleGenerator,
)
from dl_utils.gan.inference import generate_in_batches
from dl_utils.gan.stylegan_common import (
    denormalize,
    path_length_penalty,
    r1_penalty,
    sample_mixing_latents,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.optimization import update_ema_by_images
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    save_model_weights,
)


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan2"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZE = 32
TOTAL_KIMG = 5_000
TOTAL_BATCHES = TOTAL_KIMG * 1_000 // BATCH_SIZE
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
SAMPLE_BATCH_SIZE = 4
SAMPLE_EVERY_EPOCHS = 2
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
    "loss_d_main",
    "loss_g_main",
    "r1_penalty",
    "path_penalty",
)


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    path_mean,
    global_step,
    num_batches,
    device,
):
    """Train one data epoch with separate regularization passes."""
    if len(loader) < num_batches:
        raise ValueError("num_batches exceeds one data epoch.")
    metrics = MetricAccumulator(METRIC_NAMES, device=device)

    # Lazily yield only the first num_batches batches without buffering loader.
    # This supports an exact training budget in a possibly partial final epoch.
    batches = islice(loader, num_batches)
    for real, _ in tqdm(
        batches,
        total=num_batches,
        desc="Training",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ):
        real = real.to(device, non_blocking=True)
        batch_size = real.shape[0]

        # Main discriminator pass: non-saturating logistic objective.
        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
            STYLE_MIXING_PROBABILITY,
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
        discriminator.requires_grad_(False)
        try:
            # Main generator pass: non-saturating logistic objective.
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                device,
                STYLE_MIXING_PROBABILITY,
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
            update_ema_by_images(
                averaged_generator,
                generator,
                batch_size,
                EMA_HALF_LIFE_KIMG,
            )
        finally:
            discriminator.requires_grad_(True)

        metrics.update(
            (
                loss_d_main.detach(),
                loss_g_main.detach(),
                weighted_r1.detach(),
                weighted_path.detach(),
            ),
            num_examples=batch_size,
        )
        global_step += 1

    return metrics.compute(), path_mean, global_step


def save_training_samples(generator, fixed_z, output_path, seen_kimg):
    """Save fixed-z samples with the same layer noise at every checkpoint."""
    samples = generate_in_batches(
        fixed_z,
        SAMPLE_BATCH_SIZE,
        lambda z_batch: generator(z_batch, noise_mode="fixed"),
        module=generator,
    )
    save_grid(
        denormalize(samples),
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=f"StyleGAN2 EMA fixed samples - {seen_kimg:.0f} kimg",
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
    loader = make_aligned_celeba_loader(
        DATA_DIR,
        resolution=128,
        batch_size=BATCH_SIZE,
        device=device,
        num_workers=NUM_WORKERS,
    )
    if TOTAL_KIMG * 1_000 % BATCH_SIZE:
        raise ValueError("TOTAL_KIMG must contain complete training batches.")
    total_epochs = (TOTAL_BATCHES + len(loader) - 1) // len(loader)

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

    models = {
        "online_generator": generator,
        "stylegan2_generator": averaged_generator,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    checkpoint = TrainingCheckpoint(
        CHECKPOINT_DIR / "latest.pth",
        unit="epoch-main-metrics",
        models=models,
        optimizers=optimizers,
    )
    fixed_z = torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=device)
    completed_epochs, state = checkpoint.resume(
        resume_from,
        initial_state={
            "loss_history": {
                "kimg": [],
                **{name: [] for name in METRIC_NAMES},
            },
            "fixed_z": fixed_z.cpu(),
            "path_mean": torch.zeros(()),
            "global_step": 0,
        },
    )
    if resume_from is not None:
        print(f"Resumed after epoch {completed_epochs}: {resume_from}")
    if completed_epochs > total_epochs:
        raise ValueError("Checkpoint epoch exceeds the training budget.")

    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(device)
    path_mean = state["path_mean"].to(device)
    global_step = state["global_step"]

    for epoch in range(completed_epochs + 1, total_epochs + 1):
        print(f"Epoch {epoch}/{total_epochs}")
        num_batches = min(len(loader), TOTAL_BATCHES - global_step)
        metrics, path_mean, global_step = train_epoch(
            generator,
            discriminator,
            loader,
            optimizer_g,
            optimizer_d,
            averaged_generator,
            path_mean,
            global_step,
            num_batches,
            device,
        )
        seen_kimg = global_step * BATCH_SIZE / 1_000
        loss_history["kimg"].append(seen_kimg)
        for name, value in metrics.items():
            loss_history[name].append(value)

        if (
            epoch == 1
            or epoch % SAMPLE_EVERY_EPOCHS == 0
            or epoch == total_epochs
        ):
            save_training_samples(
                averaged_generator,
                fixed_z,
                TRAINING_DIR / f"kimg_{round(seen_kimg):05d}.png",
                seen_kimg,
            )

        state["path_mean"] = path_mean.detach().cpu()
        state["global_step"] = global_step
        checkpoint.save(epoch, state)

    if global_step != TOTAL_BATCHES:
        raise RuntimeError("Training ended with an unexpected step count.")
    save_model_weights(
        averaged_generator,
        OUT_DIR / "stylegan2_generator.pth",
        metadata={
            "model_name": "stylegan2",
            "model_config": MODEL_CONFIG,
        },
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D logistic loss": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G non-saturating loss": loss_history["loss_g_main"],
            },
            "R1 regularization": {
                "Weighted lazy R1 contribution": loss_history[
                    "r1_penalty"
                ],
            },
            "Path length regularization": {
                "Weighted lazy path contribution": loss_history[
                    "path_penalty"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the full-resolution 128x128 CelebA StyleGAN2."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume_from=args.resume_from)
