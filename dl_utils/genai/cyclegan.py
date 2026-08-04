"""CycleGAN models and training helpers.

Adapted from the companion code of "Generative Deep Learning", 2nd edition
(book_repos/DGAI/utils/ch06util.py): unpaired image-to-image translation
with two generators and two discriminators, trained with adversarial +
cycle-consistency losses. Snapshot and training paths are parameterized so
outputs land under the project's output/ directory.
"""

import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from dl_utils.data.images import load_rgb_image


def save_translation_snapshot(
    epoch,
    batch_idx,
    A,
    B,
    fake_A,
    fake_B,
    out_dir,
    domain_A_label,
    domain_B_label,
):
    """Save a titled 2x2 grid for both CycleGAN translation directions."""
    images = [A[0], fake_B[0], B[0], fake_A[0]]
    titles = [
        f"Real: {domain_A_label}",
        f"Generated: {domain_B_label}",
        f"Real: {domain_B_label}",
        f"Generated: {domain_A_label}",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), dpi=150)
    for axis, image, title in zip(axes.flat, images, titles):
        image = image.detach().float().mul(0.5).add(0.5).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0).cpu().numpy())
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"CycleGAN translations — epoch {epoch}, step {batch_idx}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        out_dir / f"comparison_epoch_{epoch:02d}_step_{batch_idx:05d}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def train_epoch(disc_A, disc_B, gen_A, gen_B, loader, opt_disc,
        opt_gen, l1, mse, d_scaler, g_scaler, device, out_dir,
        epoch=1, loss_callback=None, loss_updates_per_epoch=20,
        snapshot_updates_per_epoch=10, domain_A_label="Domain A",
        domain_B_label="Domain B"):
    """Train for one epoch and optionally report window-averaged losses."""
    if loss_updates_per_epoch <= 0:
        raise ValueError("loss_updates_per_epoch must be positive.")
    if snapshot_updates_per_epoch <= 0:
        raise ValueError("snapshot_updates_per_epoch must be positive.")

    loop = tqdm(loader, leave=True)
    loss_sums = torch.zeros(2, device=device)
    window_loss_sums = [0.0, 0.0]
    num_examples = 0
    window_examples = 0
    num_batches = len(loader)
    update_interval = max(
        1,
        (num_batches + loss_updates_per_epoch - 1)
        // loss_updates_per_epoch,
    )
    snapshot_interval = max(
        1,
        (num_batches + snapshot_updates_per_epoch - 1)
        // snapshot_updates_per_epoch,
    )

    for i, (A,B) in enumerate(loop):
        batch_idx = i + 1
        A = A.to(device)
        B = B.to(device)
        batch_size = A.shape[0]

        # Train Discriminators A and B
        with torch.amp.autocast(device.type):
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # Average loss of the two discriminators
            D_loss = (D_A_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train the two generators
        with torch.amp.autocast(device.type):
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

            # NEW in Cycle GANs: cycle consistency loss
            cycle_B = gen_B(fake_A)
            cycle_A = gen_A(fake_B)
            cycle_B_loss = l1(B, cycle_B)
            cycle_A_loss = l1(A, cycle_A)

            # Total generator loss
            G_loss = (loss_G_A + loss_G_B + cycle_A_loss * 10
                + cycle_B_loss * 10)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if batch_idx % snapshot_interval == 0 or batch_idx == num_batches:
            save_translation_snapshot(
                epoch,
                batch_idx,
                A,
                B,
                fake_A,
                fake_B,
                out_dir,
                domain_A_label,
                domain_B_label,
            )

        loss_sums[0] += D_loss.detach() * batch_size
        loss_sums[1] += G_loss.detach() * batch_size
        num_examples += batch_size
        D_loss_value = D_loss.item()
        G_loss_value = G_loss.item()
        if loss_callback is not None:
            window_loss_sums[0] += D_loss_value * batch_size
            window_loss_sums[1] += G_loss_value * batch_size
            window_examples += batch_size
            if batch_idx % update_interval == 0 or batch_idx == num_batches:
                loss_callback(
                    batch_idx / num_batches,
                    window_loss_sums[0] / window_examples,
                    window_loss_sums[1] / window_examples,
                )
                window_loss_sums = [0.0, 0.0]
                window_examples = 0
        loop.set_postfix(D_loss=D_loss_value, G_loss=G_loss_value)

    if num_examples == 0:
        raise ValueError("Cannot train CycleGAN with an empty data loader.")
    loss_D_sum, loss_G_sum = loss_sums.tolist()
    return loss_D_sum / num_examples, loss_G_sum / num_examples


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels,
                                    out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity())

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels,
                      use_act=False, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64,
                 num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7,
                stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True))

        self.down_blocks = nn.ModuleList(
            [ConvBlock(num_features, num_features*2, kernel_size=3,
                       stride=2, padding=1),
            ConvBlock(num_features*2, num_features*4, kernel_size=3,
                stride=2, padding=1)])

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4)
            for _ in range(num_residuals)])

        self.up_blocks = nn.ModuleList(
            [ConvBlock(num_features * 4, num_features * 2,
                    down=False, kernel_size=3, stride=2,
                    padding=1, output_padding=1),
            ConvBlock(num_features * 2, num_features * 1,
                    down=False, kernel_size=3, stride=2,
                    padding=1, output_padding=1)])

        self.last = nn.Conv2d(num_features * 1, img_channels,
            kernel_size=7, stride=1,
            padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1,
                padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0],
                kernel_size=4, stride=2, padding=1,
                padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True))
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature,
                stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4,
                stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)  # PatchGAN

    def forward(self, x):
        out = self.model(self.initial(x))
        return torch.sigmoid(out)

class LoadData(Dataset):
    """Unpaired dataset: samples domain A and domain B independently."""
    def __init__(self, root_A, root_B, transform=None):
        super().__init__()
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        # Collect file paths for all images in domain A.
        self.A_images = []
        for r in root_A:
            # List the filenames in the current domain-A directory.
            files = os.listdir(r)
            # Safely combine the directory path with each filename.
            self.A_images += [os.path.join(r, i) for i in files]
        self.B_images = []
        for r in root_B:
            files = os.listdir(r)
            self.B_images += [os.path.join(r, i) for i in files]

        self.len_data = max(len(self.A_images),
                            len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]
        A_img = np.array(load_rgb_image(A_img))
        B_img = np.array(load_rgb_image(B_img))
        if self.transform:
            augmentations = self.transform(image=B_img,
                                           image0=A_img)
            B_img = augmentations["image"]
            A_img = augmentations["image0"]
        return A_img, B_img


def weights_init(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1 or name.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif name.find('Norm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
