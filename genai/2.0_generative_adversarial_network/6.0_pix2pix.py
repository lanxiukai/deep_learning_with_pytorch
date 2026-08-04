"""Train pix2pix for paired CelebA grayscale -> color translation.

Compared with 5.0_cycle_gan.py:
    - Each grayscale/RGB pair comes from the same photograph.
    - One U-Net generator replaces two ResNet generators.
    - D sees (source, target), and G uses GAN + 100 * paired L1 loss.
    - No reverse generator or cycle-consistency loss is needed.

The reusable dataset and complete model definitions live in
dl_utils/genai/pix2pix.py.  The configuration follows common pix2pix practice:
256x256 U-Net, 70x70 PatchGAN, batch size 1, Adam(lr=2e-4, beta1=0.5), and
200 epochs with linear learning-rate decay after epoch 100.

Data:
    data/celeba/{black,blond}, also used by 5.0_cycle_gan.py.  Grayscale inputs
    are derived in memory, so no second dataset copy is required.

Outputs:
    output/pix2pix/training/epoch_*.png: input/generated/target rows
    output/pix2pix/pix2pix_generator.pth
    output/pix2pix/loss_curves.png
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.devices.randomness import set_seed
from dl_utils.devices.selection import try_gpu
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.pix2pix import (
    CelebAColorizationDataset,
    ConditionalPatchDiscriminator,
    UNetGenerator,
    build_paired_transform,
    denormalize,
    initialize_weights,
)
from dl_utils.plot.figures import Animator
from dl_utils.training.timing import Timer, format_epoch_timing


PROJECT_ROOT = infer_project_root()
CELEBA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "pix2pix" / "training"
PIX2PIX_OUT_DIR = OUT_DIR.parent

NUM_EPOCHS = 200
DECAY_START_EPOCH = 100
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
LAMBDA_L1 = 100


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_G,
    opt_D,
    bce,
    l1,
    scaler_G,
    scaler_D,
    device,
):
    """Train one epoch; these two updates are the core pix2pix algorithm."""
    loop = tqdm(loader, leave=True)
    loss_sums = torch.zeros(4, device=device)
    num_examples = 0

    for source, target in loop:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = source.shape[0]

        # D(source, real target) -> 1; D(source, generated target) -> 0.
        opt_G.zero_grad(set_to_none=True)
        opt_D.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            generated = generator(source)
            real_logits = discriminator(source, target)
            fake_logits = discriminator(source, generated.detach())
            loss_D = 0.5 * (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
            )
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # G(source) must fool D and reconstruct its aligned target.
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            fake_logits = discriminator(source, generated)
            loss_G_GAN = bce(fake_logits, torch.ones_like(fake_logits))
            loss_G_L1 = l1(generated, target)
            loss_G = loss_G_GAN + LAMBDA_L1 * loss_G_L1
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        loss_sums += torch.stack(
            [loss_D, loss_G, loss_G_GAN, loss_G_L1]
        ).detach() * batch_size
        num_examples += batch_size
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

    return tuple(value / num_examples for value in loss_sums.tolist())


def save_training_samples(generator, source, target, device, output_path):
    """Save fixed validation pairs as input/generated/target rows."""
    generator.eval()
    with torch.inference_mode():
        generated = generator(source.to(device)).cpu()
    generator.train()
    images = torch.cat(
        [denormalize(source), denormalize(generated), denormalize(target)]
    )
    save_image(images, output_path, nrow=source.shape[0])


def main():
    set_seed(42)
    reset_dir(str(OUT_DIR))
    device = try_gpu()

    train_dataset = CelebAColorizationDataset(
        CELEBA_DIR,
        "train",
        build_paired_transform(training=True),
    )
    validation_dataset = CelebAColorizationDataset(
        CELEBA_DIR,
        "validation",
        build_paired_transform(training=False),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    sample_source, sample_target = next(
        iter(DataLoader(validation_dataset, batch_size=4, shuffle=False))
    )

    generator = UNetGenerator().to(device)
    discriminator = ConditionalPatchDiscriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    opt_G = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    scaler_G = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scaler_D = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    def lr_factor(epoch):
        if epoch < DECAY_START_EPOCH:
            return 1.0
        decay_epochs = NUM_EPOCHS - DECAY_START_EPOCH
        return max(0.0, 1 - (epoch - DECAY_START_EPOCH + 1) / (decay_epochs + 1))

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_factor)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_factor)
    animator = Animator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, NUM_EPOCHS],
        legend=["discriminator", "generator"],
        figsize=(5, 3.5),
    )

    timer = Timer()
    for epoch in range(1, NUM_EPOCHS + 1):
        loss_D, loss_G, loss_G_GAN, loss_G_L1 = train_epoch(
            generator,
            discriminator,
            loader,
            opt_G,
            opt_D,
            bce,
            l1,
            scaler_G,
            scaler_D,
            device,
        )
        animator.add(epoch, (loss_D, loss_G))
        save_training_samples(
            generator,
            sample_source,
            sample_target,
            device,
            OUT_DIR / f"epoch_{epoch:03d}.png",
        )
        print(
            f"epoch {epoch:03d}: D={loss_D:.3f}, G={loss_G:.3f}, "
            f"GAN={loss_G_GAN:.3f}, L1={loss_G_L1:.3f}"
        )
        scheduler_G.step()
        scheduler_D.step()

    print(f"{format_epoch_timing(timer.stop(), NUM_EPOCHS)} on {device}")
    torch.save(
        generator.state_dict(),
        PIX2PIX_OUT_DIR / "pix2pix_generator.pth",
    )
    animator.fig.savefig(PIX2PIX_OUT_DIR / "loss_curves.png", dpi=300)
    plt.close(animator.fig)


if __name__ == "__main__":
    main()
