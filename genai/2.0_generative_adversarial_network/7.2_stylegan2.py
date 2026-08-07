"""Train a compact full-resolution StyleGAN2 on CIFAR-10.

Compared with the earlier GAN lessons, the important additions are:
    - z -> W mapping followed by a learned 4x4 constant;
    - per-layer weight modulation/demodulation and stochastic noise;
    - skip-connected RGB outputs instead of progressive growing;
    - style mixing plus lazy R1 and path-length regularization;
    - non-saturating logistic losses and lazy-reg optimizer compensation;
    - exponential moving-average weights for monitoring and evaluation.

The complete mapping, synthesis, and discriminator models live in
dl_utils/genai/stylegan2.py. This file keeps the new losses and update
schedule visible. It is a small teaching model with pure PyTorch filtered
resampling, not the paper's high-resolution fused CUDA implementation.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/stylegan2/training/epoch_*.png: fixed z and fixed layer noise
    output/stylegan2/stylegan2_generator.pth: EMA generator checkpoint
    output/stylegan2/loss_curves.png: objectives and regularizers

Training images: 50,000 (49,984 used per epoch with drop_last=True)
Generator: 3.28 M params; discriminator: 3.48 M; total: 6.77 M
"""

import math
import random
from copy import deepcopy

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
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan2"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 4
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 4
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
D_REG_EVERY = 16
PATH_WEIGHT = 2.0
G_REG_EVERY = 4
PATH_BATCH_SHRINK = 2
EMA_DECAY = 0.995
LOSS_UPDATES_PER_EPOCH = 4

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
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
    """Usually return two z batches so different layers cannot co-adapt."""
    z = torch.randn(batch_size, z_dim, device=device)
    mixing_z = (
        torch.randn_like(z)
        if random.random() < STYLE_MIXING_PROBABILITY
        else None
    )
    return z, mixing_z


def r1_penalty(real_scores, real_images):
    gradients = torch.autograd.grad(
        real_scores.sum(), real_images, create_graph=True
    )[0]
    return gradients.square().flatten(1).sum(dim=1).mean()


def path_length_penalty(images, ws, running_mean, decay=0.01):
    noise = torch.randn_like(images) / math.sqrt(
        images.shape[2] * images.shape[3]
    )
    gradients = torch.autograd.grad(
        (images * noise).sum(), ws, create_graph=True
    )[0]
    lengths = torch.sqrt(
        gradients.square().sum(dim=2).mean(dim=1) + 1e-8
    )
    updated_mean = running_mean.lerp(lengths.mean().detach(), decay)
    return (lengths - updated_mean).square().mean(), updated_mean


@torch.no_grad()
def update_ema(ema_generator, generator, decay=EMA_DECAY):
    """Move evaluation weights toward the current training generator."""
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1).")
    for ema_parameter, parameter in zip(
        ema_generator.parameters(),
        generator.parameters(),
        strict=True,
    ):
        ema_parameter.lerp_(parameter, 1 - decay)
    for ema_buffer, buffer in zip(
        ema_generator.buffers(),
        generator.buffers(),
        strict=True,
    ):
        ema_buffer.copy_(buffer)


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_G,
    opt_D,
    generator_ema,
    path_mean,
    global_step,
    device,
    loss_callback=None,
    loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
    progress_bar=None,
):
    """Train one epoch with StyleGAN2's main and lazy-regularized updates."""
    if loss_updates_per_epoch <= 0:
        raise ValueError("loss_updates_per_epoch must be positive.")

    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    loss_sums = torch.zeros(len(METRIC_NAMES), device=device)
    window_loss_sums = [0.0] * len(METRIC_NAMES)
    regularizer_sums = [0.0, 0.0]
    regularizer_examples = [0, 0]
    num_examples = 0
    window_examples = 0
    num_batches = len(loader)
    update_interval = max(
        1,
        (num_batches + loss_updates_per_epoch - 1)
        // loss_updates_per_epoch,
    )

    for batch_index, (real, _) in enumerate(loop, start=1):
        real = real.to(device, non_blocking=True)
        batch_size = real.shape[0]

        # D logistic loss; R1 is evaluated only every D_REG_EVERY steps.
        use_r1 = global_step % D_REG_EVERY == 0
        real.requires_grad_(use_r1)
        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
        )
        with torch.no_grad():
            fake = generator(z, mixing_z=mixing_z)
        real_scores = discriminator(real)
        fake_scores = discriminator(fake)
        loss_D_main = F.softplus(-real_scores).mean()
        loss_D_main += F.softplus(fake_scores).mean()
        penalty_r1 = (
            r1_penalty(real_scores, real)
            if use_r1
            else real.new_zeros(())
        )
        weighted_r1 = 0.5 * R1_GAMMA * D_REG_EVERY * penalty_r1
        loss_D = loss_D_main + weighted_r1
        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()
        real.requires_grad_(False)

        # G non-saturating loss; path regularization uses a smaller batch.
        discriminator.requires_grad_(False)
        try:
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
            loss_G_main = F.softplus(-discriminator(fake)).mean()
            use_path = global_step % G_REG_EVERY == 0
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
            else:
                penalty_path = real.new_zeros(())
            weighted_path = PATH_WEIGHT * G_REG_EVERY * penalty_path
            loss_G = loss_G_main + weighted_path
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()
            update_ema(generator_ema, generator)
        finally:
            discriminator.requires_grad_(True)

        batch_losses = torch.stack(
            [
                loss_D,
                loss_D_main,
                loss_G,
                loss_G_main,
                weighted_r1,
                weighted_path,
            ]
        ).detach()
        loss_sums += batch_losses * batch_size
        num_examples += batch_size
        loss_values = batch_losses.tolist()

        if use_r1:
            regularizer_sums[0] += penalty_r1.item() * batch_size
            regularizer_examples[0] += batch_size
        if use_path:
            regularizer_sums[1] += penalty_path.item() * path_batch
            regularizer_examples[1] += path_batch

        if loss_callback is not None:
            for index, loss_value in enumerate(loss_values):
                window_loss_sums[index] += loss_value * batch_size
            window_examples += batch_size
            if (
                batch_index % update_interval == 0
                or batch_index == num_batches
            ):
                window_metrics = dict(
                    zip(
                        METRIC_NAMES,
                        (
                            loss_sum / window_examples
                            for loss_sum in window_loss_sums
                        ),
                        strict=True,
                    )
                )
                loss_callback(batch_index / num_batches, window_metrics)
                window_loss_sums = [0.0] * len(METRIC_NAMES)
                window_examples = 0

        global_step += 1
        postfix = {
            "D": loss_values[0],
            "G": loss_values[2],
            "R1": loss_values[4],
            "Path": loss_values[5],
        }
        if progress_bar is None:
            loop.set_postfix(postfix, refresh=False)
        else:
            progress_bar.set_postfix(postfix, refresh=False)
            progress_bar.update(1)

    epoch_metrics = dict(
        zip(
            METRIC_NAMES,
            (value / num_examples for value in loss_sums.tolist()),
            strict=True,
        )
    )
    regularizer_metrics = {
        "r1_penalty": regularizer_sums[0]
        / max(1, regularizer_examples[0]),
        "path_penalty": regularizer_sums[1]
        / max(1, regularizer_examples[1]),
    }
    return epoch_metrics, regularizer_metrics, path_mean, global_step


def save_training_samples(generator, fixed_z, output_path, epoch):
    """Save fixed-z samples with the same layer noise at every epoch."""
    was_training = generator.training
    generator.eval()
    with torch.inference_mode():
        samples = generator(fixed_z, noise_mode="fixed").cpu()
    if was_training:
        generator.train()
    save_grid(
        denormalize(samples),
        output_path,
        nrow=8,
        title=f"StyleGAN2 fixed latent samples - epoch {epoch:03d}",
    )


def count_parameters(module):
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )
    reset_dir(str(TRAINING_DIR))
    set_seed(42)
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
    generator_ema = deepcopy(generator).eval().requires_grad_(False)
    discriminator = StyleDiscriminator(BASE_CHANNELS).to(device)

    # Lazy penalties occur less often, so compensate Adam's lr and beta2.
    g_ratio = G_REG_EVERY / (G_REG_EVERY + 1)
    d_ratio = D_REG_EVERY / (D_REG_EVERY + 1)
    opt_G = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE * g_ratio,
        betas=(0.0, 0.99**g_ratio),
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )

    print(
        f"Generator parameters: {count_parameters(generator):,}; "
        f"discriminator parameters: {count_parameters(discriminator):,}."
    )

    fixed_z = torch.randn(64, Z_DIM, device=device)
    path_mean = torch.zeros((), device=device)
    global_step = 0
    loss_steps = []
    metric_history = {name: [] for name in METRIC_NAMES}

    timer = Timer()
    with tqdm(
        total=NUM_EPOCHS * len(loader),
        desc=f"StyleGAN2 | 32x32 full network | Epoch 1/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, NUM_EPOCHS + 1):
            progress_bar.set_description(
                f"StyleGAN2 | 32x32 full network | "
                f"Epoch {epoch}/{NUM_EPOCHS}",
                refresh=False,
            )

            def update_loss_curve(progress, window_metrics):
                loss_steps.append(epoch - 1 + progress)
                for name, value in window_metrics.items():
                    metric_history[name].append(value)

            epoch_metrics, regularizers, path_mean, global_step = train_epoch(
                generator,
                discriminator,
                loader,
                opt_G,
                opt_D,
                generator_ema,
                path_mean,
                global_step,
                device,
                loss_callback=update_loss_curve,
                loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
                progress_bar=progress_bar,
            )
            save_training_samples(
                generator_ema,
                fixed_z,
                TRAINING_DIR / f"epoch_{epoch:03d}.png",
                epoch,
            )
            progress_bar.write(
                f"epoch {epoch:03d}: "
                f"D={epoch_metrics['loss_d']:.3f}, "
                f"G={epoch_metrics['loss_g']:.3f}, "
                f"R1={regularizers['r1_penalty']:.3f}, "
                f"path={regularizers['path_penalty']:.3f}, "
                f"path_mean={path_mean.item():.3f}"
            )

    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(
        {
            "model_name": "stylegan2",
            "model_config": MODEL_CONFIG,
            "state_dict": generator_ema.state_dict(),
        },
        OUT_DIR / "stylegan2_generator.pth",
    )
    save_loss_panels(
        loss_steps,
        {
            "Discriminator objective": {
                "D total loss": metric_history["loss_d"],
                "D logistic loss": metric_history["loss_d_main"],
            },
            "Generator objective": {
                "G total loss": metric_history["loss_g"],
                "G non-saturating loss": metric_history["loss_g_main"],
            },
            "R1 regularization": {
                "Weighted lazy R1 contribution": metric_history[
                    "weighted_r1"
                ],
            },
            "Path length regularization": {
                "Weighted lazy path contribution": metric_history[
                    "weighted_path"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
