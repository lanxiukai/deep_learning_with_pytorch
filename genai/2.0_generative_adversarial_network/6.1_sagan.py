"""Train a compact class-conditional SAGAN on CIFAR-10.

This attention-focused lesson uses CIFAR-10's native 32x32 resolution. It
replaces the middle identity mappings with self-attention on middle feature
maps. The learnable residual scales start at zero, so training decides how
strongly to use non-local cues. The SN-GAN in ``6.0_sn_gan.py`` remains the
simpler reference baseline, including its block-local conditional BatchNorm.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/sagan/training/epoch_*.png: titled fixed class samples
    output/sagan/generator.pth
    output/sagan/loss_curves.png: separate D and G loss panels
    output/sagan/mechanism_diagnostics.png: hinge activity and projection gap
    output/sagan/attention_diagnostics.png: gates, entropy, and cost signals

Training data — CIFAR-10:
Training images:         50,000
Samples per epoch:       49,984 (781 full batches; drop_last=True)

Generator:                1.17 M params
Discriminator:            1.22 M params
Total:                    2.39 M params

Precision:
    USE_AMP=True uses FP32 plus AMP_DTYPE for training and sample inference.
    Set USE_AMP=False for pure FP32, or choose torch.float16 instead of the
    default torch.bfloat16. FP16 automatically uses gradient scaling.
"""

import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.sagan import SAGANDiscriminator, SAGANGenerator
from dl_utils.genai.sn_gan import (
    CIFAR10_CLASS_NAMES,
    count_spectral_norm_layers,
    cyclically_mismatched_labels,
    discriminator_diagnostic_sums,
    discriminator_hinge_loss,
    generator_hinge_loss,
    make_fixed_class_latent_grid,
)
from dl_utils.plot.figures import Animator, save_loss_panels
from dl_utils.plot.images import save_training_samples
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUT_DIR = PROJECT_ROOT / "output" / "sagan"
TRAINING_DIR = OUT_DIR / "training"

NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_CLASSES = 10
SAMPLES_PER_CLASS = 8
Z_DIM = 120
BASE_CHANNELS = 32
CONDITION_DIM = 32
IMAGE_SIZE = 32
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 2e-4
SAMPLE_EVERY_EPOCHS = 5
# BF16 has FP32-like range and native Tensor Core support on Ampere/Ada GPUs.
USE_AMP = True
AMP_DTYPE = torch.bfloat16  # torch.bfloat16 (default) or torch.float16

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "num_classes": NUM_CLASSES,
    "base_channels": BASE_CHANNELS,
    "condition_dim": CONDITION_DIM,
    "image_size": IMAGE_SIZE,
}


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_g,
    opt_d,
    device,
    scaler_g,
    scaler_d,
    amp_enabled,
    amp_dtype,
    progress_bar=None,
):
    """Train one epoch and report batch progress."""
    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    loss_sums = torch.zeros(2, device=device)
    diagnostic_sums = torch.zeros(3, device=device)
    num_examples = 0

    for batch_index, (real, labels) in enumerate(loop):
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = real.shape[0]
        is_last_batch = batch_index + 1 == len(loader)
        mismatched_labels = cyclically_mismatched_labels(
            labels, NUM_CLASSES
        )

        noise = torch.randn(batch_size, Z_DIM, device=device)
        if is_last_batch:
            discriminator.attention.capture_next_attention_entropy()
        with torch.amp.autocast(
            device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            with torch.no_grad():
                fake = generator(noise, labels)
            (
                real_scores,
                correct_projection,
                mismatched_projection,
            ) = discriminator.forward_with_projection_diagnostics(
                real,
                labels,
                mismatched_labels,
            )
            fake_scores = discriminator(fake, labels)
            loss_d = discriminator_hinge_loss(real_scores, fake_scores)
        diagnostic_sums += discriminator_diagnostic_sums(
            real_scores,
            fake_scores,
            correct_projection,
            mismatched_projection,
        )
        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        sampled_labels = torch.randint(
            NUM_CLASSES, (batch_size,), device=device
        )
        noise = torch.randn(batch_size, Z_DIM, device=device)
        discriminator.requires_grad_(False)
        try:
            opt_g.zero_grad(set_to_none=True)
            if is_last_batch:
                generator.attention.capture_next_attention_entropy()
            with torch.amp.autocast(
                device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                fake = generator(noise, sampled_labels)
                loss_g = generator_hinge_loss(
                    discriminator(fake, sampled_labels)
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
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
    return tuple(value / num_examples for value in epoch_sums)


def main():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )

    reset_dir(str(TRAINING_DIR))
    set_seed(42)
    device = try_gpu()
    amp_enabled = USE_AMP and device.type == "cuda"
    if amp_enabled and AMP_DTYPE not in (torch.float16, torch.bfloat16):
        raise ValueError("AMP_DTYPE must be torch.float16 or torch.bfloat16.")
    if (
        amp_enabled
        and AMP_DTYPE == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "This CUDA device does not support BF16. Set AMP_DTYPE to "
            "torch.float16 or disable USE_AMP."
        )

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

    generator = SAGANGenerator(**MODEL_CONFIG).to(device)
    discriminator = SAGANDiscriminator(
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)
    sn_layer_counts = {
        "generator": count_spectral_norm_layers(generator),
        "discriminator": count_spectral_norm_layers(discriminator),
    }
    print(
        "Spectral-normalized layers: "
        f"G={sn_layer_counts['generator']}, "
        f"D={sn_layer_counts['discriminator']}"
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
    scaler_enabled = amp_enabled and AMP_DTYPE == torch.float16
    scaler_g = torch.amp.GradScaler(
        device.type,
        enabled=scaler_enabled,
    )
    scaler_d = torch.amp.GradScaler(
        device.type,
        enabled=scaler_enabled,
    )

    fixed_noise, fixed_labels = make_fixed_class_latent_grid(
        NUM_CLASSES,
        SAMPLES_PER_CLASS,
        Z_DIM,
        device,
    )
    loss_steps = []
    discriminator_losses = []
    generator_losses = []
    real_hinge_active_fractions = []
    fake_hinge_active_fractions = []
    projection_score_gaps = []
    generator_gammas = []
    discriminator_gammas = []
    generator_attention_entropies = []
    discriminator_attention_entropies = []
    training_throughputs = []
    peak_memory_megabytes = []
    samples_per_epoch = len(loader) * BATCH_SIZE
    animator = Animator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, NUM_EPOCHS],
        legend=["D hinge loss", "G adversarial loss"],
        figsize=(6, 4),
    )

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
            ) = train_epoch(
                generator,
                discriminator,
                loader,
                opt_g,
                opt_d,
                device,
                scaler_g,
                scaler_d,
                amp_enabled,
                AMP_DTYPE,
                progress_bar=progress_bar,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_start

            generator_gamma = generator.attention.gamma.detach().item()
            discriminator_gamma = discriminator.attention.gamma.detach().item()
            generator_entropy = generator.attention.last_attention_entropy
            discriminator_entropy = (
                discriminator.attention.last_attention_entropy
            )
            if generator_entropy is None or discriminator_entropy is None:
                raise RuntimeError("Attention entropy probe did not run.")

            loss_steps.append(epoch)
            discriminator_losses.append(loss_d)
            generator_losses.append(loss_g)
            real_hinge_active_fractions.append(real_hinge_active)
            fake_hinge_active_fractions.append(fake_hinge_active)
            projection_score_gaps.append(projection_score_gap)
            generator_gammas.append(generator_gamma)
            discriminator_gammas.append(discriminator_gamma)
            generator_attention_entropies.append(generator_entropy)
            discriminator_attention_entropies.append(discriminator_entropy)
            training_throughputs.append(samples_per_epoch / epoch_seconds)
            if device.type == "cuda":
                peak_memory_megabytes.append(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                )
            animator.add(epoch, (loss_d, loss_g))
            progress_bar.set_postfix(
                loss_D=loss_d,
                loss_G=loss_g,
                gamma_G=generator_gamma,
                gamma_D=discriminator_gamma,
                entropy_G=generator_entropy,
                entropy_D=discriminator_entropy,
                hinge_real=real_hinge_active,
                hinge_fake=fake_hinge_active,
                projection_gap=projection_score_gap,
                refresh=False,
            )
            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_training_samples(
                    generator,
                    fixed_noise,
                    fixed_labels,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                    class_names=CIFAR10_CLASS_NAMES[:NUM_CLASSES],
                    title=f"SAGAN fixed class samples - epoch {epoch:03d}",
                    amp_enabled=amp_enabled,
                    amp_dtype=AMP_DTYPE,
                    shared_latents_across_classes=True,
                )
    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(
        {
            "model_name": "sagan",
            "model_config": MODEL_CONFIG,
            "sn_layer_counts": sn_layer_counts,
            "training_precision": (
                str(AMP_DTYPE).removeprefix("torch.")
                if amp_enabled
                else "float32"
            ),
            "state_dict": generator.state_dict(),
        },
        OUT_DIR / "generator.pth",
    )
    save_loss_panels(
        loss_steps,
        {
            "Discriminator hinge loss": {
                "D hinge loss": discriminator_losses,
            },
            "Generator adversarial loss": {
                "G adversarial loss": generator_losses,
            },
        },
        OUT_DIR / "loss_curves.png",
    )
    save_loss_panels(
        loss_steps,
        {
            "Active hinge-margin fraction": {
                "Real: D(x, y) < 1": real_hinge_active_fractions,
                "Fake: D(G(z), y) > -1": fake_hinge_active_fractions,
            },
            "Projection score gap": {
                "Correct - mismatched label": projection_score_gaps,
            },
        },
        OUT_DIR / "mechanism_diagnostics.png",
        ylabel="value",
    )
    diagnostic_panels = {
        "Attention residual gates": {
            "Generator gamma": generator_gammas,
            "Discriminator gamma": discriminator_gammas,
        },
        "Normalized attention entropy": {
            "Generator entropy": generator_attention_entropies,
            "Discriminator entropy": discriminator_attention_entropies,
        },
        "Training throughput (samples/s)": {
            "Training throughput": training_throughputs,
        },
    }
    if peak_memory_megabytes:
        diagnostic_panels["Peak CUDA memory (MiB)"] = {
            "Peak allocated memory": peak_memory_megabytes,
        }
    save_loss_panels(
        loss_steps,
        diagnostic_panels,
        OUT_DIR / "attention_diagnostics.png",
        ylabel="value",
    )


if __name__ == "__main__":
    main()
