"""
Conditional GAN (cGAN) with a WGAN-GP critic for generating faces with or
without glasses.

Adapted from "Generative Deep Learning", 2nd ed.
Data: data/glasses-256, prepared by tool_scripts/download_dataset.py.
Outputs: output/gan/cgan/.
Checkpoint evaluation: 4.1_conditional_gan_evaluation.py.

Training data:
With glasses (G):       2,543 images
Without glasses (NoG):  1,957 images
Available total:         4,500 images
Samples per epoch:       4,480
Note: Counts include 517 repository-tracked label corrections. drop_last=True
omits 20 shuffled images per epoch.

Generator:  12.9 M params
Critic:      2.8 M params
Total:      15.7 M params
"""

# Data lives under <project_root>/data; generated images and checkpoints
# are written under <project_root>/output/gan/cgan
import os

import torch
from torch import nn
from torch.nn import functional
from tqdm import tqdm

from dl_utils.data.vision import image_folder_dataset
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.gan import Critic, Generator, gradient_penalty
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import save_loss_panels
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CACHED_DATA_DIR = DATA_DIR / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "cgan"
GENERATOR_PATH = OUT_DIR / "cgan.pth"

NUM_EPOCHS = 120
BATCH_SIZE = 32
MAX_NUM_WORKERS = 8
Z_DIM = 100
IMAGE_CHANNELS = 3
FEATURES = 16
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 5e-5
N_CRITIC = 5
LAMBDA_GP = 10
SAMPLE_GRID_ROWS = 4
SAMPLE_GRID_COLUMNS = 8
SAMPLES_TO_DISPLAY = SAMPLE_GRID_ROWS * SAMPLE_GRID_COLUMNS
SAMPLE_EVERY_EPOCHS = 6


def weights_init(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def plot_epoch(gen, z_dim, epoch, device, epoch_dir):
    for label_idx, prefix in ((0, "G"), (1, "NoG")):
        noise = torch.randn(SAMPLES_TO_DISPLAY, z_dim, 1, 1)
        labels = torch.zeros(SAMPLES_TO_DISPLAY, 2, 1, 1)
        # use one-hot label so G knows what to generate:
        # [1,0] for glasses ('G'), [0,1] for no glasses ('NoG')
        labels[:, label_idx, :, :] = 1
        noise_and_labels = torch.cat([noise, labels], dim=1).to(device)
        generated_images = gen(noise_and_labels).cpu().detach()
        with plt.ioff():
            fig = plt.figure(figsize=(20, 10), dpi=100)
            for sample_index in range(SAMPLES_TO_DISPLAY):
                plt.subplot(
                    SAMPLE_GRID_ROWS,
                    SAMPLE_GRID_COLUMNS,
                    sample_index + 1,
                )
                image = (generated_images[sample_index] / 2 + 0.5).permute(1, 2, 0)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])
            plt.subplots_adjust(hspace=-0.6)
            plt.savefig(epoch_dir / f"{prefix}{epoch}.png")
            plt.close(fig)


def update_critic(
    real_img_labels,
    noise,
    onehots,
    net_critic,
    net_gen,
    opt_critic,
    device,
    lambda_gp=10,
):
    """One critic update: push E[D(real)] - E[D(fake)] as high as possible,
    subject to the gradient penalty.

    real_img_labels is img_and_labels (image + 2 condition channels); noise is
    raw noise and onehots the (B, 2) condition concatenated onto it for net_gen."""
    batch_size = noise.shape[0]
    opt_critic.zero_grad(set_to_none=True)
    real_critic = net_critic(real_img_labels).mean()  # E[D(real)]
    noise_and_labels = torch.cat([noise, onehots.reshape(batch_size, 2, 1, 1)], dim=1)
    # The generator is not updated during a critic step, so avoid building
    # and immediately discarding its autograd graph.
    with torch.no_grad():
        fake_img = net_gen(noise_and_labels)
    fake_img_labels = torch.cat([fake_img, real_img_labels[:, 3:, :, :]], dim=1)
    fake_critic = net_critic(fake_img_labels).mean()  # E[D(fake)]
    wasserstein_gap = real_critic - fake_critic
    penalty = gradient_penalty(
        net_critic, real_img_labels, fake_img_labels, device, lambda_gp
    )
    # Minimize the negative Wasserstein gap plus the weighted penalty.
    loss_critic = -wasserstein_gap + penalty
    loss_critic.backward()
    opt_critic.step()  # Only update the net_critic
    return loss_critic, wasserstein_gap.detach(), penalty.detach()


def update_gen(noise, onehots, img_and_labels, net_critic, net_gen, opt_gen):
    """One generator update: maximize E[D(G(z))], i.e. fool the critic."""
    batch_size = noise.shape[0]
    opt_gen.zero_grad(set_to_none=True)
    net_critic.requires_grad_(False)
    try:
        noise_and_labels = torch.cat(
            [noise, onehots.reshape(batch_size, 2, 1, 1)], dim=1
        )
        fake_img = net_gen(noise_and_labels)
        # Gradients still flow through the critic to fake_img, but frozen
        # critic parameters do not accumulate gradients during this step.
        fake_img_labels = torch.cat([fake_img, img_and_labels[:, 3:, :, :]], dim=1)
        loss_gen = -net_critic(fake_img_labels).mean()  # -E[D(fake)]
        loss_gen.backward()
        opt_gen.step()  # Only update net_gen
    finally:
        net_critic.requires_grad_(True)
    return loss_gen


def train(
    net_critic,
    net_gen,
    data_loader,
    num_epochs,
    z_dim,
    out_dir,
    n_critic=5,
    lambda_gp=10,
    lr_gen=1e-4,
    lr_critic=1e-4,
    device=None,
):
    device = try_gpu() if device is None else device
    net_gen.apply(weights_init)
    net_critic.apply(weights_init)
    net_critic, net_gen = net_critic.to(device), net_gen.to(device)
    opt_gen = torch.optim.Adam(net_gen.parameters(), lr=lr_gen, betas=(0.0, 0.9))
    opt_critic = torch.optim.Adam(
        net_critic.parameters(), lr=lr_critic, betas=(0.0, 0.9)
    )

    epoch_dir = out_dir / "epochs"
    reset_dir(str(epoch_dir))
    epochs = []
    wasserstein_gaps = []
    penalties = []
    generator_losses = []

    d_steps = 0  # critic updates so far; generator is updated once per n_critic
    with tqdm(
        total=num_epochs * len(data_loader),
        desc=f"Epoch 1/{num_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs}", refresh=False)
            # Train one epoch
            # Keep diagnostics on the GPU and synchronize only once per epoch.
            # Critic and generator update counts are tracked separately.
            value_sums = torch.zeros(3, device=device)
            critic_examples = gen_examples = 0
            for images, labels in data_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch_size, _, height, width = images.shape
                one_hot_labels = functional.one_hot(labels, num_classes=2).to(
                    images.dtype
                )
                label_channels = one_hot_labels[:, :, None, None].expand(
                    -1, -1, height, width
                )
                img_and_labels = torch.cat([images, label_channels], dim=1)
                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                _, wasserstein_gap, penalty = update_critic(
                    img_and_labels,
                    noise,
                    one_hot_labels,
                    net_critic,
                    net_gen,
                    opt_critic,
                    device,
                    lambda_gp=lambda_gp,
                )
                value_sums[0] += wasserstein_gap * batch_size
                value_sums[1] += penalty * batch_size
                critic_examples += batch_size
                d_steps += 1
                if d_steps % n_critic == 0:
                    noise = torch.normal(
                        0, 1, size=(batch_size, z_dim, 1, 1), device=device
                    )
                    loss_gen = update_gen(
                        noise,
                        one_hot_labels,
                        img_and_labels,
                        net_critic,
                        net_gen,
                        opt_gen,
                    )
                    value_sums[2] += loss_gen.detach() * batch_size
                    gen_examples += batch_size
                progress_bar.update(1)
            # Save generated examples (G / NoG grids) and update the live monitor
            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                plot_epoch(net_gen, z_dim, epoch, device, epoch_dir)
            gap_sum, penalty_sum, loss_gen_sum = value_sums.tolist()
            wasserstein_gap = gap_sum / critic_examples
            penalty = penalty_sum / critic_examples
            loss_gen = loss_gen_sum / max(gen_examples, 1)
            epochs.append(epoch)
            wasserstein_gaps.append(wasserstein_gap)
            penalties.append(penalty)
            generator_losses.append(loss_gen)
            progress_bar.set_postfix(
                gap=f"{wasserstein_gap:.3f}",
                penalty=f"{penalty:.3f}",
                loss_G=f"{loss_gen:.3f}",
            )
    print(
        f"wasserstein_gap {wasserstein_gap:.3f}, "
        f"gradient_penalty {penalty:.3f}, loss_gen {loss_gen:.3f}, "
        f"on {device}"
    )
    torch.save(net_gen.state_dict(), GENERATOR_PATH)
    save_loss_panels(
        epochs,
        {
            "Wasserstein gap": {"Wasserstein gap": wasserstein_gaps},
            "Gradient penalty": {"Gradient penalty": penalties},
            "Generator loss": {"G loss": generator_losses},
        },
        out_dir / "loss_curves.png",
    )


def main():
    reset_dir(str(OUT_DIR))
    net_gen = Generator(Z_DIM + 2, IMAGE_CHANNELS, FEATURES)
    net_critic = Critic(IMAGE_CHANNELS + 2, FEATURES)

    device = try_gpu()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not CACHED_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Resized dataset cache not found: {CACHED_DATA_DIR}\n"
            "Create it with:\n"
            "python tool_scripts/download_dataset.py"
        )
    data_set = image_folder_dataset(
        CACHED_DATA_DIR, normalize=(0.5, 0.5)
    )  # (pixel - mean) / std

    # Decode and transform images lazily in worker processes instead of
    # materializing duplicate float32 tensors for the entire dataset.
    num_workers = min(MAX_NUM_WORKERS, os.cpu_count() or 1)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )

    train(
        net_critic,
        net_gen,
        data_loader,
        NUM_EPOCHS,
        Z_DIM,
        OUT_DIR,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
        lr_gen=LEARNING_RATE_G,
        lr_critic=LEARNING_RATE_D,
        device=device,
    )


if __name__ == "__main__":
    main()
