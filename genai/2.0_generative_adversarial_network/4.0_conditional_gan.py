"""
Conditional GAN (cGAN) with WGAN-GP critic — faces with/without glasses.

Adapted from the companion code of "Generative Deep Learning", 2nd ed.
(book_repos/DGAI/ch05Conditional_GAN.ipynb); notebook code cells kept in
original order, markdown narrative removed.
Data: data/glasses (Kaggle eyeglasses dataset; assumes it has already been
downloaded and split into G/ and NoG/ — this script starts from data loading).
Outputs: output/cgan/.

Checkpoint evaluation moved to 4.1_conditional_gan_analysis.py.

Generator:  12.9 M params
Critic:      2.8 M params
Total:      15.7 M params
"""

# Data lives under <project_root>/data; generated images and checkpoints
# are written under <project_root>/output/cgan
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.filesystem.directories import reset_dir
from dl_utils.devices.selection import try_gpu
from dl_utils.genai.gan import gradient_penalty, Generator, Critic
from dl_utils.data.vision import image_folder_dataset
from dl_utils.plot.figures import Animator
from dl_utils.training.timing import Timer

import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
CACHED_DATA_DIR = DATA_DIR / 'glasses-256'
OUT_DIR = PROJECT_ROOT / 'output' / 'cgan'
reset_dir(str(OUT_DIR))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_epoch(gen, z_dim, epoch, animator, device, epoch_dir):
    fakes = []
    for label_idx, prefix in ((0, 'G'), (1, 'NoG')):
        noise = torch.randn(32, z_dim, 1, 1)
        labels = torch.zeros(32, 2, 1, 1)
        # use one-hot label so G knows what to generate:
        # [1,0] for glasses ('G'), [0,1] for no glasses ('NoG')
        labels[:,label_idx,:,:] = 1
        noise_and_labels = torch.cat([noise,labels],dim=1).to(device)
        fake = gen(noise_and_labels).cpu().detach()
        fakes.append(fake)
        with plt.ioff():
            fig = plt.figure(figsize=(20,10),dpi=100)
            for i in range(32):
                ax = plt.subplot(4, 8, i + 1)
                img = (fake[i]/2 + 0.5).permute(1,2,0)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
            plt.subplots_adjust(hspace=-0.6)
            plt.savefig(epoch_dir / f"{prefix}{epoch}.png")
            plt.close(fig)
    # live monitor grid: top 2 rows glasses, bottom 2 rows no glasses
    grid = torch.cat(
        [torch.cat([fakes[cls][i*8+j]/2 + 0.5 for j in range(8)], dim=2)
         for cls in (0, 1) for i in range(2)], dim=1)
    animator.axes[2].cla()
    animator.axes[2].imshow(grid.permute(1, 2, 0))

def update_critic(real_img_labels, noise, onehots, net_critic, net_gen, 
                  opt_critic, device, lambda_gp=10):
    """One critic update: push E[D(real)] - E[D(fake)] as high as possible,
    subject to the gradient penalty.

    real_img_labels is img_and_labels (image + 2 condition channels); noise is 
    raw noise and onehots the (B, 2) condition concatenated onto it for net_gen."""
    batch_size = noise.shape[0]
    opt_critic.zero_grad(set_to_none=True)
    real_critic = net_critic(real_img_labels).mean()       # E[D(real)]
    noise_and_labels = torch.cat(
        [noise, onehots.reshape(batch_size, 2, 1, 1)], dim=1)
    # The generator is not updated during a critic step, so avoid building
    # and immediately discarding its autograd graph.
    with torch.no_grad():
        fake_img = net_gen(noise_and_labels)
    fake_img_labels = torch.cat([fake_img,
                                 real_img_labels[:, 3:, :, :]], dim=1)
    fake_critic = net_critic(fake_img_labels).mean()       # E[D(fake)]
    wasserstein_gap = real_critic - fake_critic
    penalty = gradient_penalty(
        net_critic, real_img_labels, fake_img_labels, device, lambda_gp)
    # Minimize the negative Wasserstein gap plus the weighted penalty.
    loss_critic = -wasserstein_gap + penalty
    loss_critic.backward()
    opt_critic.step()   # Only update the net_critic
    return loss_critic, wasserstein_gap.detach(), penalty.detach()

def update_gen(noise, onehots, img_and_labels, net_critic, net_gen, opt_gen):
    """One generator update: maximize E[D(G(z))], i.e. fool the critic."""
    batch_size = noise.shape[0]
    opt_gen.zero_grad(set_to_none=True)
    net_critic.requires_grad_(False)
    try:
        noise_and_labels = torch.cat(
            [noise, onehots.reshape(batch_size, 2, 1, 1)], dim=1)
        fake_img = net_gen(noise_and_labels)
        # Gradients still flow through the critic to fake_img, but frozen
        # critic parameters do not accumulate gradients during this step.
        fake_img_labels = torch.cat(
            [fake_img, img_and_labels[:, 3:, :, :]], dim=1)
        loss_gen = -net_critic(fake_img_labels).mean()     # -E[D(fake)]
        loss_gen.backward()
        opt_gen.step()   # Only update net_gen
    finally:
        net_critic.requires_grad_(True)
    return loss_gen

def train(net_critic, net_gen, data_loader, num_epochs, z_dim, out_dir,
          n_critic=5, lambda_gp=10, lr_gen=1e-4, lr_critic=1e-4,
          device=try_gpu()):
    net_gen.apply(weights_init)
    net_critic.apply(weights_init)
    net_critic, net_gen = net_critic.to(device), net_gen.to(device)
    opt_gen = torch.optim.Adam(
        net_gen.parameters(), lr=lr_gen, betas=(0.0, 0.9))
    opt_critic = torch.optim.Adam(net_critic.parameters(), lr=lr_critic,
                                  betas=(0.0, 0.9))

    epoch_dir = out_dir / 'epochs'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    animator = Animator(xlabel='epoch', ylabel='critic diagnostics',
                        xlim=[1, num_epochs], nrows=3, figsize=(5, 7.5),
                        legend=['wasserstein gap', 'gradient penalty'])
    animator.fig.subplots_adjust(hspace=0.4)
    generator_epochs, generator_losses = [], []

    d_steps = 0  # critic updates so far; generator is updated once per n_critic
    timer = Timer()
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        # Keep diagnostics on the GPU and synchronize only once per epoch.
        # Critic and generator update counts are tracked separately.
        value_sums = torch.zeros(3, device=device)
        critic_examples = gen_examples = 0
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size, _, height, width = images.shape
            onehots = F.one_hot(labels, num_classes=2).to(images.dtype)
            label_channels = onehots[:, :, None, None].expand(
                -1, -1, height, width)
            img_and_labels = torch.cat([images, label_channels], dim=1)
            Z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            _, wasserstein_gap, penalty = update_critic(
                img_and_labels, Z, onehots, net_critic, net_gen, opt_critic,
                device, lambda_gp=lambda_gp)
            value_sums[0] += wasserstein_gap * batch_size
            value_sums[1] += penalty * batch_size
            critic_examples += batch_size
            d_steps += 1
            if d_steps % n_critic == 0:
                Z = torch.normal(0, 1, size=(batch_size, z_dim, 1, 1),
                                 device=device)
                loss_gen = update_gen(Z, onehots, img_and_labels,
                                      net_critic, net_gen, opt_gen)
                value_sums[2] += loss_gen.detach() * batch_size
                gen_examples += batch_size
        # Save generated examples (G / NoG grids) and update the live monitor
        if epoch == 1 or epoch % 6 == 0:
            plot_epoch(net_gen, z_dim, epoch, animator, device, epoch_dir)
        gap_sum, penalty_sum, loss_gen_sum = value_sums.tolist()
        wasserstein_gap = gap_sum / critic_examples
        penalty = penalty_sum / critic_examples
        loss_gen = loss_gen_sum / max(gen_examples, 1)
        generator_epochs.append(epoch)
        generator_losses.append(loss_gen)
        generator_axis = animator.axes[1]
        generator_axis.cla()
        generator_axis.plot(generator_epochs, generator_losses, 'm--')
        generator_axis.set_xlabel('epoch')
        generator_axis.set_ylabel('generator loss')
        generator_axis.set_xlim([1, num_epochs])
        generator_axis.grid()
        animator.add(epoch, (wasserstein_gap, penalty))
    total_time = timer.stop()
    total_tenths = round(total_time * 10)
    hours, remainder = divmod(total_tenths, 36000)
    minutes, second_tenths = divmod(remainder, 600)
    seconds = second_tenths / 10
    avg_tenths = round(total_time / num_epochs * 10)
    avg_minutes, avg_second_tenths = divmod(avg_tenths, 600)
    avg_seconds = avg_second_tenths / 10
    print(f'wasserstein_gap {wasserstein_gap:.3f}, '
          f'gradient_penalty {penalty:.3f}, loss_gen {loss_gen:.3f}, '
          f'training time {int(hours)} h {int(minutes)} min {seconds:.1f} s, '
          f'avg {avg_minutes} min {avg_seconds:.1f} s/epoch '
          f'on {str(device)}')
    torch.save(net_gen.state_dict(), out_dir / 'cgan.pth')
    animator.fig.savefig(out_dir / 'loss_curves.png', dpi=300)
    plt.close(animator.fig)

def main():
    z_dim, img_channels, features = 100, 3, 16
    net_gen = Generator(z_dim + 2, img_channels, features)
    net_critic = Critic(img_channels + 2, features)

    device = try_gpu()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    batch_size = 32
    if not CACHED_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Resized dataset cache not found: {CACHED_DATA_DIR}\n"
            "Create it with:\n"
            "python tool_scripts/resize_image_folder.py "
            "data/glasses data/glasses-256 --size 256")
    data_set = image_folder_dataset(
        CACHED_DATA_DIR, normalize=(0.5, 0.5))  # (x - mean) / std

    # Decode and transform images lazily in worker processes instead of
    # materializing duplicate float32 tensors for the entire dataset.
    num_workers = min(8, os.cpu_count() or 1)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True)

    num_epochs = 120
    lr_gen, lr_critic = 1e-4, 5e-5
    train(net_critic, net_gen, data_loader, num_epochs, z_dim, OUT_DIR,
          n_critic=5, lambda_gp=10, lr_gen=lr_gen, lr_critic=lr_critic,
          device=device)


if __name__ == '__main__':
    main()
