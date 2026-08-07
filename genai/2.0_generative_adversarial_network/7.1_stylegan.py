"""Train a compact progressive StyleGAN on CIFAR-10.

This lesson keeps ProGAN's 4x4-to-32x32 schedule and adds a Z-to-W mapping
network, a learned constant, AdaIN, stochastic per-layer noise, style mixing,
and the truncation moving average. It uses logistic discriminator and
non-saturating generator objectives with lazy R1 regularization.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/stylegan/training/*.png: fixed-z/fixed-noise stage monitoring
    output/stylegan/stylegan_generator.pth: final generator with w_avg
    output/stylegan/loss_curves.png: logistic objectives and R1 contribution

Training data — CIFAR-10:
Training images:         50,000
Samples per epoch:       49,920 at 4x4/8x8; 49,984 at 16x16/32x32
Note: Each phase lasts 12 epochs. The 4x4 stage has one stabilization phase;
later stages have one fade-in and one stabilization phase each.

Generator:                2.76 M params
Discriminator:            3.38 M params
Total:                    6.14 M params
"""

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.stylegan import (
    StyleGANDiscriminator,
    StyleGANGenerator,
    denormalize,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "stylegan"
TRAINING_DIR = OUT_DIR / "training"

RESOLUTIONS = (4, 8, 16, 32)
BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32}
EPOCHS_PER_PHASE = 12
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
    "weighted_r1",
)


@dataclass(frozen=True)
class TrainingPhase:
    """One fixed-resolution segment of the progressive schedule."""

    resolution: int
    name: str
    num_epochs: int
    batch_size: int
    num_batches: int


def build_training_schedule(num_examples):
    """Describe the progressive phases before allocating data loaders."""
    phases = []
    for resolution in RESOLUTIONS:
        names = ("stabilization",) if resolution == 4 else (
            "fade-in",
            "stabilization",
        )
        batch_size = BATCH_SIZES[resolution]
        num_batches = num_examples // batch_size
        if num_batches == 0:
            raise ValueError(
                f"batch size {batch_size} exceeds dataset size {num_examples}"
            )
        for name in names:
            phases.append(
                TrainingPhase(
                    resolution=resolution,
                    name=name,
                    num_epochs=EPOCHS_PER_PHASE,
                    batch_size=batch_size,
                    num_batches=num_batches,
                )
            )
    return tuple(phases)


def phase_alpha(phase, epoch_index, batch_index):
    """Increase alpha from zero to one over the complete fade-in stage."""
    if phase.name != "fade-in":
        return 1.0
    total_steps = phase.num_epochs * phase.num_batches
    if total_steps == 1:
        return 1.0
    current_step = epoch_index * phase.num_batches + batch_index
    return current_step / (total_steps - 1)


def make_loader(resolution, batch_size, device):
    """Create one CIFAR-10 loader for the requested training resolution."""
    transform_steps = []
    if resolution != 32:
        transform_steps.append(
            transforms.Resize(
                (resolution, resolution),
                antialias=True,
            )
        )
    transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
        transform=transforms.Compose(transform_steps),
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
    """Usually return two latent batches to decorrelate adjacent styles."""
    z = torch.randn(batch_size, z_dim, device=device)
    mixing_z = (
        torch.randn_like(z)
        if random.random() < STYLE_MIXING_PROBABILITY
        else None
    )
    return z, mixing_z


def r1_penalty(real_scores, real_images):
    """Measure squared discriminator gradients at real training images."""
    gradients = torch.autograd.grad(
        real_scores.sum(),
        real_images,
        create_graph=True,
    )[0]
    return gradients.square().flatten(1).sum(dim=1).mean()


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    phase,
    epoch_index,
    global_step,
    device,
    *,
    loss_callback=None,
    loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
    progress_bar=None,
):
    """Train one StyleGAN epoch at the active resolution and fade alpha."""
    if loss_updates_per_epoch <= 0:
        raise ValueError("loss_updates_per_epoch must be positive.")
    if len(loader) != phase.num_batches:
        raise ValueError("loader length does not match the training schedule.")

    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    metric_sums = torch.zeros(len(METRIC_NAMES), device=device)
    window_sums = [0.0] * len(METRIC_NAMES)
    num_examples = 0
    window_examples = 0
    update_interval = max(
        1,
        (phase.num_batches + loss_updates_per_epoch - 1)
        // loss_updates_per_epoch,
    )

    for batch_index, (real_images, _) in enumerate(loop):
        alpha = phase_alpha(phase, epoch_index, batch_index)
        real_images = real_images.to(device, non_blocking=True)
        batch_size = real_images.shape[0]

        use_r1 = global_step % D_REG_EVERY == 0
        real_images.requires_grad_(use_r1)
        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            device,
        )
        with torch.no_grad():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
            )
        real_scores = discriminator(
            real_images,
            resolution=phase.resolution,
            alpha=alpha,
        )
        fake_scores = discriminator(
            fake_images,
            resolution=phase.resolution,
            alpha=alpha,
        )
        loss_d_main = F.softplus(-real_scores).mean()
        loss_d_main = loss_d_main + F.softplus(fake_scores).mean()
        penalty_r1 = (
            r1_penalty(real_scores, real_images)
            if use_r1
            else real_images.new_zeros(())
        )
        weighted_r1 = 0.5 * R1_GAMMA * D_REG_EVERY * penalty_r1
        loss_d = loss_d_main + weighted_r1
        optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        optimizer_d.step()
        real_images.requires_grad_(False)

        discriminator.requires_grad_(False)
        try:
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                device,
            )
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
                update_w_avg=True,
            )
            loss_g = F.softplus(
                -discriminator(
                    fake_images,
                    resolution=phase.resolution,
                    alpha=alpha,
                )
            ).mean()
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()
        finally:
            discriminator.requires_grad_(True)

        metrics = torch.stack(
            [loss_d, loss_d_main, loss_g, weighted_r1]
        ).detach()
        metric_sums += metrics * batch_size
        num_examples += batch_size
        values = metrics.tolist()

        if loss_callback is not None:
            for index, value in enumerate(values):
                window_sums[index] += value * batch_size
            window_examples += batch_size
            if (
                (batch_index + 1) % update_interval == 0
                or batch_index + 1 == phase.num_batches
            ):
                loss_callback(
                    (batch_index + 1) / phase.num_batches,
                    dict(
                        zip(
                            METRIC_NAMES,
                            (
                                value / window_examples
                                for value in window_sums
                            ),
                            strict=True,
                        )
                    ),
                )
                window_sums = [0.0] * len(METRIC_NAMES)
                window_examples = 0

        global_step += 1
        description = (
            f"StyleGAN | {phase.resolution}x{phase.resolution} "
            f"{phase.name} | Epoch {epoch_index + 1}/{phase.num_epochs} | "
            f"alpha={alpha:.2f}"
        )
        postfix = {
            "D": values[0],
            "G": values[2],
            "R1": values[3],
        }
        if progress_bar is None:
            loop.set_description(description, refresh=False)
            loop.set_postfix(postfix, refresh=False)
        else:
            progress_bar.set_description(description, refresh=False)
            progress_bar.set_postfix(postfix, refresh=False)
            progress_bar.update(1)

    epoch_metrics = dict(
        zip(
            METRIC_NAMES,
            (value / num_examples for value in metric_sums.tolist()),
            strict=True,
        )
    )
    return epoch_metrics, global_step


def save_training_samples(
    generator,
    fixed_z,
    output_path,
    phase,
    epoch,
    alpha,
):
    """Save fixed-z and fixed-noise samples labelled by progressive stage."""
    was_training = generator.training
    generator.eval()
    with torch.inference_mode():
        samples = generator(
            fixed_z,
            resolution=phase.resolution,
            alpha=alpha,
            noise_mode="fixed",
        ).cpu()
    if was_training:
        generator.train()
    save_grid(
        denormalize(samples),
        output_path,
        nrow=8,
        title=(
            f"StyleGAN fixed latent samples - {phase.resolution}x"
            f"{phase.resolution} {phase.name} - epoch {epoch:03d} - "
            f"alpha={alpha:.2f}"
        ),
    )


def main():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )
    reset_dir(str(TRAINING_DIR))
    set_seed(42)
    device = try_gpu()

    dataset_info = datasets.CIFAR10(
        DATA_DIR,
        train=True,
        download=False,
    )
    training_schedule = build_training_schedule(len(dataset_info))
    total_steps = sum(
        phase.num_epochs * phase.num_batches
        for phase in training_schedule
    )
    total_epochs = sum(phase.num_epochs for phase in training_schedule)

    generator = StyleGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = StyleGANDiscriminator(BASE_CHANNELS).to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    d_ratio = D_REG_EVERY / (D_REG_EVERY + 1)
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )
    fixed_z = torch.randn(64, Z_DIM, device=device)
    global_step = 0
    loss_steps = []
    metric_history = {name: [] for name in METRIC_NAMES}
    completed_epochs = 0
    timer = Timer()

    with tqdm(
        total=total_steps,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for phase in training_schedule:
            loader = make_loader(
                phase.resolution,
                phase.batch_size,
                device,
            )
            for epoch_index in range(phase.num_epochs):

                def update_loss_curve(progress, window_metrics):
                    loss_steps.append(
                        completed_epochs + epoch_index + progress
                    )
                    for name, value in window_metrics.items():
                        metric_history[name].append(value)

                metrics, global_step = train_epoch(
                    generator,
                    discriminator,
                    loader,
                    optimizer_g,
                    optimizer_d,
                    phase,
                    epoch_index,
                    global_step,
                    device,
                    loss_callback=update_loss_curve,
                    progress_bar=progress_bar,
                )
                alpha = phase_alpha(
                    phase,
                    epoch_index,
                    phase.num_batches - 1,
                )
                epoch_number = epoch_index + 1
                stage_slug = phase.name.replace("-", "_")
                save_training_samples(
                    generator,
                    fixed_z,
                    TRAINING_DIR
                    / (
                        f"{phase.resolution:02d}x{phase.resolution:02d}_"
                        f"{stage_slug}_epoch_{epoch_number:03d}.png"
                    ),
                    phase,
                    epoch_number,
                    alpha,
                )
                progress_bar.write(
                    f"{phase.resolution}x{phase.resolution} {phase.name} "
                    f"epoch {epoch_number:03d}: "
                    f"D={metrics['loss_d']:.3f}, "
                    f"G={metrics['loss_g']:.3f}, "
                    f"R1={metrics['weighted_r1']:.3f}, "
                    f"alpha={alpha:.2f}"
                )
            completed_epochs += phase.num_epochs

    print(f"{format_epoch_timing(timer.stop(), total_epochs)} on {device}")
    torch.save(
        {
            "model_name": "stylegan",
            "model_config": MODEL_CONFIG,
            "state_dict": generator.state_dict(),
            "training_schedule": [
                asdict(phase) for phase in training_schedule
            ],
        },
        OUT_DIR / "stylegan_generator.pth",
    )
    save_loss_panels(
        loss_steps,
        {
            "Logistic discriminator objective": {
                "D total": metric_history["loss_d"],
                "D logistic": metric_history["loss_d_main"],
            },
            "Non-saturating generator objective": {
                "G non-saturating": metric_history["loss_g"],
            },
            "R1 regularization": {
                "Weighted lazy R1 contribution": metric_history[
                    "weighted_r1"
                ],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="global epoch across progressive stages",
    )


if __name__ == "__main__":
    main()
