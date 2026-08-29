"""
Wasserstein GAN (WGAN) and WGAN with Gradient Penalty (WGAN-GP) on anime faces.

Uses the DCGAN generator with a BatchNorm-free critic and Wasserstein loss.
Select `lipschitz='clip'` for WGAN weight clipping with RMSProp, or `'gp'`
(default) for WGAN-GP with gradient penalty and Adam. The critic is updated
`n_critic` times per generator update.

Training data:
Anime faces:      63,565 images
Samples per epoch: 63,488
Note: drop_last=True discards the final incomplete batch of 77 images each
epoch; shuffling means the omitted images may differ between epochs.

Generator:   3.6 M params
Critic:      2.8 M params
Total:       6.4 M params
"""

import os
import warnings

import torch
import torchvision
from torch import nn
from tqdm import tqdm

from dl_utils.data.images import load_rgb_image
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.gan import gradient_penalty
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import save_loss_panels
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "wgan_gp"

NUM_EPOCHS = 120
BATCH_SIZE = 128
MAX_NUM_WORKERS = 8
Z_DIM = 100
GENERATOR_BASE_CHANNELS = 64
CRITIC_BASE_CHANNELS = 64
LEARNING_RATE = 1e-4
N_CRITIC = 5
LAMBDA_GP = 10
LIPSCHITZ = "gp"
SAMPLES_TO_DISPLAY = 18
SAMPLE_GRID_COLUMNS = 6
SAMPLE_EVERY_EPOCHS = 6


class G_block(nn.Module):
    def __init__(
        self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        return self.activation(self.batch_norm(self.conv2d_trans(inputs)))


class D_block(nn.Module):
    # No BatchNorm in the critic (see module docstring, point B).
    def __init__(
        self,
        out_channels,
        in_channels=3,
        kernel_size=4,
        strides=2,
        padding=1,
        alpha=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False
        )
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, inputs):
        return self.activation(self.conv2d(inputs))


def update_critic(
    real_images,
    noise,
    critic,
    generator,
    critic_optimizer,
    device,
    lipschitz="gp",
    lambda_gp=10,
    clip_value=0.01,
):
    """One critic update: push E[D(real)] - E[D(fake)] as high as possible,
    subject to the chosen Lipschitz constraint."""
    critic_optimizer.zero_grad(set_to_none=True)
    real_score = critic(real_images).mean()  # E[D(real)]
    # The generator is not updated during a critic step, so avoid building
    # and immediately discarding its autograd graph.
    with torch.no_grad():
        generated_images = generator(noise)
    fake_score = critic(generated_images).mean()  # E[D(fake)]
    wasserstein_gap = real_score - fake_score
    penalty = torch.zeros((), device=device)
    loss_critic = -wasserstein_gap
    if lipschitz == "gp":
        # Gradient penalty: soft constraint, part of the loss
        penalty = gradient_penalty(
            critic,
            real_images,
            generated_images,
            device,
            lambda_gp,
        )
        loss_critic = loss_critic + penalty
    loss_critic.backward()
    critic_optimizer.step()  # Only update the critic
    if lipschitz == "clip":
        # Weight clipping: hard constraint, applied after the update
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.clamp_(-clip_value, clip_value)
    return loss_critic, wasserstein_gap.detach(), penalty.detach()


def update_gen(noise, critic, generator, generator_optimizer):
    """One generator update: maximize E[D(G(z))], i.e. fool the critic."""
    generator_optimizer.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    try:
        # Gradients still flow through the critic to the generated images,
        # but frozen critic parameters do not accumulate gradients.
        fake_scores = critic(generator(noise))
        loss_gen = -fake_scores.mean()  # -E[D(fake)]
        loss_gen.backward()
        generator_optimizer.step()  # Only update the generator
    finally:
        critic.requires_grad_(True)
    return loss_gen


def train(
    critic,
    generator,
    data_loader,
    num_epochs,
    z_dim,
    output_dir,
    lipschitz="gp",
    learning_rate=None,
    n_critic=5,
    lambda_gp=10,
    clip_value=0.01,
    device=None,
):
    device = try_gpu() if device is None else device
    for parameter in critic.parameters():
        nn.init.normal_(parameter, 0, 0.02)
    for parameter in generator.parameters():
        nn.init.normal_(parameter, 0, 0.02)
    critic, generator = critic.to(device), generator.to(device)
    # The choice of optimizer follows the Lipschitz enforcement method:
    # RMSProp (no momentum) for clipping, Adam for gradient penalty.
    if lipschitz == "clip":
        learning_rate = learning_rate or 5e-5
        critic_optimizer = torch.optim.RMSprop(
            critic.parameters(),
            lr=learning_rate,
        )
        generator_optimizer = torch.optim.RMSprop(
            generator.parameters(),
            lr=learning_rate,
        )
    else:
        learning_rate = learning_rate or 1e-4
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9),
        )
        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9),
        )
    epoch_dir = output_dir / "epochs"
    reset_dir(str(epoch_dir))
    epochs = []
    wasserstein_gaps = []
    penalties = []
    generator_losses = []

    d_steps = 0  # critic updates so far; generator is updated once per n_critic
    fixed_noise = torch.randn(SAMPLES_TO_DISPLAY, z_dim, 1, 1, device=device)
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
            for real_images, _ in data_loader:
                batch_size = real_images.shape[0]
                real_images = real_images.to(device, non_blocking=True)
                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                _, wasserstein_gap, penalty = update_critic(
                    real_images,
                    noise,
                    critic,
                    generator,
                    critic_optimizer,
                    device,
                    lipschitz,
                    lambda_gp,
                    clip_value,
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
                        critic,
                        generator,
                        generator_optimizer,
                    )
                    value_sums[2] += loss_gen.detach() * batch_size
                    gen_examples += batch_size
                progress_bar.update(1)
            # Show generated examples
            was_training = generator.training
            generator.eval()
            try:
                with torch.inference_mode():
                    # Normalize generated images from [-1, 1] to [0, 1] and
                    # transfer the complete batch to the CPU only once.
                    generated_images = (
                        generator(fixed_noise).permute(0, 2, 3, 1) / 2 + 0.5
                    ).cpu()
            finally:
                generator.train(was_training)
            image_grid = torch.cat(  # → (3×H, 7×W, C)
                [
                    torch.cat(
                        [  # → (H, 7×W, C)
                            generated_images[
                                row_index * SAMPLE_GRID_COLUMNS + column_index
                            ]
                            for column_index in range(SAMPLE_GRID_COLUMNS)
                        ],
                        dim=1,
                    )
                    for row_index in range(len(generated_images) // SAMPLE_GRID_COLUMNS)
                ],
                dim=0,
            )
            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                plt.imsave(
                    epoch_dir / f"epoch_{epoch:03d}.png",
                    image_grid.numpy(),
                )
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
        f"lipschitz={lipschitz}: wasserstein_gap {wasserstein_gap:.3f}, "
        f"gradient_penalty {penalty:.3f}, loss_gen {loss_gen:.3f}, "
        f"on {device}"
    )
    checkpoint_name = "wgan_gp.pth" if lipschitz == "gp" else "wgan.pth"
    torch.save(generator.state_dict(), output_dir / checkpoint_name)
    save_loss_panels(
        epochs,
        {
            "Wasserstein gap": {"Wasserstein gap": wasserstein_gaps},
            "Gradient penalty": {"Gradient penalty": penalties},
            "Generator loss": {"G loss": generator_losses},
        },
        output_dir / "loss_curves.png",
    )


def main():
    reset_dir(str(OUT_DIR))
    device = try_gpu()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),  # int [0, 255] -> Pytorch Tensor FP32 [0.0, 1.0]
            torchvision.transforms.Normalize(
                0.5, 0.5
            ),  # (pixel - 0.5) / 0.5, [0, 1] -> [-1, 1]
        ]
    )
    anime_face = torchvision.datasets.ImageFolder(
        DATA_DIR / "anime_face", transform=transformer, loader=load_rgb_image
    )
    num_workers = min(MAX_NUM_WORKERS, os.cpu_count() or 1)
    data_loader = torch.utils.data.DataLoader(
        anime_face,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )

    # Close the matplotlib / torch warning output
    warnings.filterwarnings("ignore")

    # Generator: identical to the anime-face DCGAN (BatchNorm is fine here —
    # the Lipschitz constraint only applies to the critic).
    generator_channels = GENERATOR_BASE_CHANNELS
    generator = nn.Sequential(
        G_block(
            in_channels=Z_DIM,
            out_channels=generator_channels * 8,
            strides=1,
            padding=0,
        ),  # Output: (64 * 8, 4, 4)
        G_block(
            in_channels=generator_channels * 8,
            out_channels=generator_channels * 4,
        ),  # Output: (64 * 4, 8, 8)
        G_block(
            in_channels=generator_channels * 4,
            out_channels=generator_channels * 2,
        ),  # Output: (64 * 2, 16, 16)
        G_block(
            in_channels=generator_channels * 2,
            out_channels=generator_channels,
        ),  # Output: (64, 32, 32)
        nn.ConvTranspose2d(
            in_channels=generator_channels,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.Tanh(),
    )  # Output: (3, 64, 64)

    # Critic: no sigmoid (raw score), no BatchNorm (see module docstring).
    critic_channels = CRITIC_BASE_CHANNELS
    critic = nn.Sequential(
        D_block(critic_channels),  # Output: (64, 32, 32)
        D_block(
            in_channels=critic_channels,
            out_channels=critic_channels * 2,
        ),  # Output: (64 * 2, 16, 16)
        D_block(
            in_channels=critic_channels * 2,
            out_channels=critic_channels * 4,
        ),  # Output: (64 * 4, 8, 8)
        D_block(
            in_channels=critic_channels * 4,
            out_channels=critic_channels * 8,
        ),  # Output: (64 * 8, 4, 4)
        nn.Conv2d(
            in_channels=critic_channels * 8,
            out_channels=1,
            kernel_size=4,
            bias=False,
        ),
    )  # Output: (1, 1, 1)

    # WGAN-GP (gradient penalty) — the recommended default
    train(
        critic,
        generator,
        data_loader,
        NUM_EPOCHS,
        Z_DIM,
        OUT_DIR,
        lipschitz=LIPSCHITZ,
        learning_rate=LEARNING_RATE,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
        device=device,
    )
    # WGAN (weight clipping) — uncomment to compare (also change out_dir above)
    # train(critic, generator, data_loader, NUM_EPOCHS, Z_DIM, OUT_DIR,
    #       lipschitz="clip", n_critic=N_CRITIC, clip_value=0.01)


if __name__ == "__main__":
    main()
