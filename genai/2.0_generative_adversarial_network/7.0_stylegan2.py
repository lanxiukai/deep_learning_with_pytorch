"""Train a compact 32x32 StyleGAN2 on CIFAR-10.

Compared with the earlier GAN lessons, the important additions are:
    - z -> W mapping followed by a learned 4x4 constant;
    - per-layer weight modulation/demodulation and stochastic noise;
    - skip-connected RGB outputs instead of progressive growing;
    - style mixing plus lazy R1 and path-length regularization;
    - non-saturating logistic losses and lazy-reg optimizer compensation.

The complete mapping, synthesis, and discriminator models live in
dl_utils/genai/stylegan2.py.  This file keeps the new losses and update
schedule visible.  It is a small teaching model, not the paper's
high-resolution implementation with fused kernels and filtered resampling.

Data:
    data/cifar10, prepared by tool_scripts/download_dataset_test.py.

Outputs:
    output/stylegan2/training/epoch_*.png: fixed z without stochastic noise
    output/stylegan2/stylegan2_generator.pth
    output/stylegan2/loss_curves.png

Training images: 50,000 (49,984 used per epoch with drop_last=True)
Generator: 3.28 M params; discriminator: 3.48 M; total: 6.77 M
"""

import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
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
from dl_utils.plot.figures import Animator
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
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
D_REG_EVERY = 16
PATH_WEIGHT = 2.0
G_REG_EVERY = 4
PATH_BATCH_SHRINK = 2


def sample_mixing_latents(batch_size, device):
    """Usually return two z batches so different layers cannot co-adapt."""
    z = torch.randn(batch_size, Z_DIM, device=device)
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


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_G,
    opt_D,
    path_mean,
    global_step,
    device,
):
    """Train one epoch with StyleGAN2's main and lazy-regularized updates."""
    loop = tqdm(loader, leave=True)
    loss_sums = torch.zeros(4, device=device)
    num_examples = 0

    for real, _ in loop:
        real = real.to(device, non_blocking=True)
        batch_size = real.shape[0]

        # D logistic loss; R1 is evaluated only every D_REG_EVERY steps.
        use_r1 = global_step % D_REG_EVERY == 0
        real.requires_grad_(use_r1)
        z, mixing_z = sample_mixing_latents(batch_size, device)
        with torch.no_grad():
            fake = generator(z, mixing_z=mixing_z)
        real_scores = discriminator(real)
        fake_scores = discriminator(fake)
        loss_D_main = F.softplus(-real_scores).mean()
        loss_D_main += F.softplus(fake_scores).mean()
        penalty_r1 = r1_penalty(real_scores, real) if use_r1 else real.new_zeros(())
        loss_D = loss_D_main + 0.5 * R1_GAMMA * D_REG_EVERY * penalty_r1
        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()
        real.requires_grad_(False)

        # G non-saturating loss; path regularization uses a smaller batch.
        discriminator.requires_grad_(False)
        z, mixing_z = sample_mixing_latents(batch_size, device)
        fake = generator(z, mixing_z=mixing_z)
        loss_G_main = F.softplus(-discriminator(fake)).mean()
        use_path = global_step % G_REG_EVERY == 0
        if use_path:
            path_batch = max(1, batch_size // PATH_BATCH_SHRINK)
            path_z = torch.randn(path_batch, Z_DIM, device=device)
            path_images, path_ws = generator(path_z, return_ws=True)
            penalty_path, path_mean = path_length_penalty(
                path_images, path_ws, path_mean
            )
        else:
            penalty_path = real.new_zeros(())
        loss_G = loss_G_main + PATH_WEIGHT * G_REG_EVERY * penalty_path
        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()
        discriminator.requires_grad_(True)

        loss_sums += torch.stack(
            [loss_G_main, loss_D_main, penalty_r1, penalty_path]
        ).detach() * batch_size
        num_examples += batch_size
        global_step += 1
        loop.set_postfix(loss_D=loss_D_main.item(), loss_G=loss_G_main.item())

    means = tuple(value / num_examples for value in loss_sums.tolist())
    return (*means, path_mean, global_step)


def save_training_samples(generator, fixed_z, output_path):
    """Disable stochastic noise so only learned model changes are compared."""
    generator.eval()
    with torch.inference_mode():
        samples = generator(fixed_z, randomize_noise=False)
    generator.train()
    save_image(denormalize(samples), output_path, nrow=8)


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

    generator = StyleGenerator(
        Z_DIM, STYLE_DIM, BASE_CHANNELS, MAPPING_LAYERS
    ).to(device)
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

    fixed_z = torch.randn(64, Z_DIM, device=device)
    path_mean = torch.zeros((), device=device)
    global_step = 0
    animator = Animator(
        xlabel="epoch",
        ylabel="logistic loss",
        xlim=[1, NUM_EPOCHS],
        legend=["discriminator", "generator"],
        figsize=(5, 3.5),
    )

    timer = Timer()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss_G, loss_D, penalty_r1, penalty_path, path_mean, global_step = (
            train_epoch(
                generator,
                discriminator,
                loader,
                opt_G,
                opt_D,
                path_mean,
                global_step,
                device,
            )
        )
        animator.add(epoch, (loss_D, loss_G))
        save_training_samples(
            generator, fixed_z, TRAINING_DIR / f"epoch_{epoch:03d}.png"
        )
        print(
            f"epoch {epoch:03d}: D={loss_D:.3f}, G={loss_G:.3f}, "
            f"R1={penalty_r1:.3f}, path={penalty_path:.3f}, "
            f"path_mean={path_mean.item():.3f}"
        )

    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(
        {
            "model": generator.state_dict(),
            "z_dim": Z_DIM,
            "style_dim": STYLE_DIM,
            "base_channels": BASE_CHANNELS,
            "mapping_layers": MAPPING_LAYERS,
            "path_mean": path_mean,
        },
        OUT_DIR / "stylegan2_generator.pth",
    )
    animator.fig.savefig(OUT_DIR / "loss_curves.png", dpi=300)
    plt.close(animator.fig)


if __name__ == "__main__":
    main()
