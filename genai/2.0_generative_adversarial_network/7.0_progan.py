"""Train a compact progressively growing GAN on CIFAR-10.

This first style-based generation lesson makes the progressive schedule and
the original ProGAN training ingredients explicit: equalized learning rates,
PixelNorm, minibatch standard deviation, Wasserstein objectives, gradient
penalty, and the critic drift penalty. The 4x4 network is stabilized first;
every later resolution has a fade-in phase followed by stabilization.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/progan/training/*.png: fixed-z samples labelled by stage and alpha
    output/progan/progan_generator.pth: final 32x32 generator checkpoint
    output/progan/loss_curves.png: independent objective and penalty panels
"""

from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.progan import (
    ProGANDiscriminator,
    ProGANGenerator,
    denormalize,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "progan"
TRAINING_DIR = OUT_DIR / "training"

RESOLUTIONS = (4, 8, 16, 32)
BATCH_SIZES = {4: 128, 8: 128, 16: 64, 32: 32}
EPOCHS_PER_PHASE = 12
NUM_WORKERS = 4
Z_DIM = 128
BASE_CHANNELS = 32
LEARNING_RATE = 1e-3
GRADIENT_PENALTY_WEIGHT = 10.0
DRIFT_PENALTY_WEIGHT = 1e-3
LOSS_UPDATES_PER_EPOCH = 4

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "base_channels": BASE_CHANNELS,
}
METRIC_NAMES = (
    "loss_d",
    "wasserstein_d",
    "loss_g",
    "weighted_gp",
    "weighted_drift",
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
    """Describe every stage before training so total progress is exact."""
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
    """Increase alpha linearly across one complete fade-in phase."""
    if phase.name != "fade-in":
        return 1.0
    total_steps = phase.num_epochs * phase.num_batches
    if total_steps == 1:
        return 1.0
    current_step = epoch_index * phase.num_batches + batch_index
    return current_step / (total_steps - 1)


def make_loader(resolution, batch_size, device):
    """Create a resolution-specific CIFAR-10 loader."""
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


def gradient_penalty(
    discriminator,
    real_images,
    fake_images,
    resolution,
    alpha,
):
    """Penalize critic gradients away from unit norm on interpolated images."""
    batch_size = real_images.shape[0]
    epsilon = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=real_images.device,
    )
    interpolated = torch.lerp(fake_images, real_images, epsilon)
    interpolated.requires_grad_(True)
    scores = discriminator(
        interpolated,
        resolution=resolution,
        alpha=alpha,
    )
    gradients = torch.autograd.grad(
        scores.sum(),
        interpolated,
        create_graph=True,
    )[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return (norms - 1.0).square().mean()


def train_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    phase,
    epoch_index,
    device,
    *,
    loss_callback=None,
    loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
    progress_bar=None,
):
    """Train one progressive epoch and report windowed objective components."""
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

        z = torch.randn(batch_size, generator.z_dim, device=device)
        with torch.no_grad():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
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
        wasserstein_d = fake_scores.mean() - real_scores.mean()
        penalty_gp = gradient_penalty(
            discriminator,
            real_images,
            fake_images,
            phase.resolution,
            alpha,
        )
        penalty_drift = real_scores.square().mean()
        weighted_gp = GRADIENT_PENALTY_WEIGHT * penalty_gp
        weighted_drift = DRIFT_PENALTY_WEIGHT * penalty_drift
        loss_d = wasserstein_d + weighted_gp + weighted_drift
        optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        optimizer_d.step()

        discriminator.requires_grad_(False)
        try:
            z = torch.randn(batch_size, generator.z_dim, device=device)
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
            )
            loss_g = -discriminator(
                fake_images,
                resolution=phase.resolution,
                alpha=alpha,
            ).mean()
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()
        finally:
            discriminator.requires_grad_(True)

        metrics = torch.stack(
            [
                loss_d,
                wasserstein_d,
                loss_g,
                weighted_gp,
                weighted_drift,
            ]
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

        description = (
            f"ProGAN | {phase.resolution}x{phase.resolution} {phase.name} | "
            f"Epoch {epoch_index + 1}/{phase.num_epochs} | alpha={alpha:.2f}"
        )
        postfix = {
            "D": values[0],
            "G": values[2],
            "GP": values[3],
        }
        if progress_bar is None:
            loop.set_description(description, refresh=False)
            loop.set_postfix(postfix, refresh=False)
        else:
            progress_bar.set_description(description, refresh=False)
            progress_bar.set_postfix(postfix, refresh=False)
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
    phase,
    epoch,
    alpha,
):
    """Save fixed latent samples with the active stage in the image title."""
    was_training = generator.training
    generator.eval()
    with torch.inference_mode():
        samples = generator(
            fixed_z,
            resolution=phase.resolution,
            alpha=alpha,
        ).cpu()
    if was_training:
        generator.train()
    save_grid(
        denormalize(samples),
        output_path,
        nrow=8,
        title=(
            f"ProGAN fixed latent samples - {phase.resolution}x"
            f"{phase.resolution} {phase.name} - epoch {epoch:03d} - "
            f"alpha={alpha:.2f}"
        ),
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

    generator = ProGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = ProGANDiscriminator(BASE_CHANNELS).to(device)
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
    print(
        f"Generator parameters: {count_parameters(generator):,}; "
        f"discriminator parameters: {count_parameters(discriminator):,}."
    )

    fixed_z = torch.randn(64, Z_DIM, device=device)
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

                metrics = train_epoch(
                    generator,
                    discriminator,
                    loader,
                    optimizer_g,
                    optimizer_d,
                    phase,
                    epoch_index,
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
                    f"GP={metrics['weighted_gp']:.3f}, "
                    f"drift={metrics['weighted_drift']:.5f}, "
                    f"alpha={alpha:.2f}"
                )
            completed_epochs += phase.num_epochs

    print(f"{format_epoch_timing(timer.stop(), total_epochs)} on {device}")
    torch.save(
        {
            "model_name": "progan",
            "model_config": MODEL_CONFIG,
            "state_dict": generator.state_dict(),
            "training_schedule": [
                asdict(phase) for phase in training_schedule
            ],
        },
        OUT_DIR / "progan_generator.pth",
    )
    save_loss_panels(
        loss_steps,
        {
            "Wasserstein discriminator objective": {
                "D total": metric_history["loss_d"],
                "D Wasserstein": metric_history["wasserstein_d"],
            },
            "Wasserstein generator objective": {
                "G Wasserstein": metric_history["loss_g"],
            },
            "Gradient penalty": {
                "Weighted gradient penalty": metric_history["weighted_gp"],
            },
            "Drift penalty": {
                "Weighted drift penalty": metric_history["weighted_drift"],
            },
        },
        OUT_DIR / "loss_curves.png",
        xlabel="global epoch across progressive stages",
    )


if __name__ == "__main__":
    main()
