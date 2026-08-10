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
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dl_utils.data.images import load_rgb_image
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.filesystem.directories import reset_dir
from dl_utils.devices.selection import try_gpu
from dl_utils.genai.gan import gradient_penalty
from dl_utils.plot.figures import Animator
from dl_utils.training.timing import Timer, format_epoch_timing

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
OUT_DIR = PROJECT_ROOT / 'output' / 'wgan_gp' / 'anime_face'
reset_dir(str(OUT_DIR))

class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


class D_block(nn.Module):
    # No BatchNorm in the critic (see module docstring, point B).
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.conv2d(X))


def update_critic(X, Z, net_critic, net_gen, opt_critic, device,
                  lipschitz='gp', lambda_gp=10, clip_value=0.01):
    """One critic update: push E[D(real)] - E[D(fake)] as high as possible,
    subject to the chosen Lipschitz constraint."""
    opt_critic.zero_grad(set_to_none=True)
    real_Y = net_critic(X).mean()                # E[D(real)]
    # The generator is not updated during a critic step, so avoid building
    # and immediately discarding its autograd graph.
    with torch.no_grad():
        fake_X = net_gen(Z)
    fake_Y = net_critic(fake_X).mean()           # E[D(fake)]
    wasserstein_gap = real_Y - fake_Y
    penalty = torch.zeros((), device=device)
    loss_critic = -wasserstein_gap
    if lipschitz == 'gp':
        # Gradient penalty: soft constraint, part of the loss
        penalty = gradient_penalty(
            net_critic, X, fake_X, device, lambda_gp)
        loss_critic = loss_critic + penalty
    loss_critic.backward()
    opt_critic.step()   # Only update the net_critic
    if lipschitz == 'clip':
        # Weight clipping: hard constraint, applied after the update
        with torch.no_grad():
            for p in net_critic.parameters():
                p.clamp_(-clip_value, clip_value)
    return loss_critic, wasserstein_gap.detach(), penalty.detach()


def update_gen(Z, net_critic, net_gen, opt_gen):
    """One generator update: maximize E[D(G(z))], i.e. fool the critic."""
    opt_gen.zero_grad(set_to_none=True)
    net_critic.requires_grad_(False)
    try:
        # Gradients still flow through the critic to the generated images,
        # but frozen critic parameters do not accumulate gradients.
        fake_Y = net_critic(net_gen(Z))
        loss_gen = -fake_Y.mean()  # -E[D(fake)]
        loss_gen.backward()
        opt_gen.step()   # Only update net_gen
    finally:
        net_critic.requires_grad_(True)
    return loss_gen


def train(net_critic, net_gen, data_iter, num_epochs, latent_dim, out_dir,
          lipschitz='gp', lr=None, n_critic=5, lambda_gp=10, clip_value=0.01,
          device=try_gpu()):
    for w in net_critic.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_gen.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_critic, net_gen = net_critic.to(device), net_gen.to(device)
    # The choice of optimizer follows the Lipschitz enforcement method:
    # RMSProp (no momentum) for clipping, Adam for gradient penalty.
    if lipschitz == 'clip':
        # if not lr: lr = 5e-5 
        lr = lr or 5e-5  # WGAN paper's small shared learning rate
        opt_critic = torch.optim.RMSprop(net_critic.parameters(), lr=lr)
        opt_gen = torch.optim.RMSprop(net_gen.parameters(), lr=lr)
    else:
        lr = lr or 1e-4  # WGAN-GP paper's learning rate
        opt_critic = torch.optim.Adam(net_critic.parameters(), lr=lr,
                                          betas=(0.5, 0.9))
        opt_gen = torch.optim.Adam(net_gen.parameters(), lr=lr,
                                       betas=(0.5, 0.9))
    epoch_dir = out_dir / 'epochs'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    animator = Animator(xlabel='epoch', ylabel='critic diagnostics',
                        xlim=[1, num_epochs], nrows=3, figsize=(5, 7.5),
                        legend=['wasserstein gap', 'gradient penalty'])
    animator.fig.subplots_adjust(hspace=0.4)
    generator_epochs, generator_losses = [], []

    d_steps = 0  # critic updates so far; generator is updated once per n_critic
    fixed_noise = torch.randn(21, latent_dim, 1, 1, device=device)
    timer = Timer()
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        # Keep diagnostics on the GPU and synchronize only once per epoch.
        # Critic and generator update counts are tracked separately.
        value_sums = torch.zeros(3, device=device)
        critic_examples = gen_examples = 0
        for X, _ in data_iter:
            batch_size = X.shape[0]
            X = X.to(device, non_blocking=True)
            Z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            _, wasserstein_gap, penalty = update_critic(
                X, Z, net_critic, net_gen, opt_critic, device,
                lipschitz, lambda_gp, clip_value)
            value_sums[0] += wasserstein_gap * batch_size
            value_sums[1] += penalty * batch_size
            critic_examples += batch_size
            d_steps += 1
            if d_steps % n_critic == 0:
                Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1),
                                 device=device)
                loss_gen = update_gen(Z, net_critic, net_gen, opt_gen)
                value_sums[2] += loss_gen.detach() * batch_size
                gen_examples += batch_size
        # Show generated examples
        was_training = net_gen.training
        net_gen.eval()
        try:
            with torch.inference_mode():
                # Normalize generated images from [-1, 1] to [0, 1] and
                # transfer the complete batch to the CPU only once.
                fake_X = (
                    net_gen(fixed_noise).permute(0, 2, 3, 1) / 2 + 0.5
                ).cpu()
        finally:
            net_gen.train(was_training)
        imgs = torch.cat(  # → (3×H, 7×W, C)
            [torch.cat([   # → (H, 7×W, C)
                fake_X[i * 7 + j] for j in range(7)], dim=1)
             for i in range(len(fake_X)//7)], dim=0)
        if epoch == 1 or epoch % 6 == 0:
            plt.imsave(epoch_dir / f'epoch_{epoch:03d}.png', imgs.numpy())
        animator.axes[2].cla()
        animator.axes[2].imshow(imgs)
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
    timing = format_epoch_timing(timer.stop(), num_epochs)
    print(f'lipschitz={lipschitz}: wasserstein_gap {wasserstein_gap:.3f}, '
          f'gradient_penalty {penalty:.3f}, loss_gen {loss_gen:.3f}, '
          f'{timing} on {str(device)}')
    checkpoint_name = 'wgan_gp.pth' if lipschitz == 'gp' else 'wgan.pth'
    torch.save(net_gen.state_dict(), out_dir / checkpoint_name)
    animator.fig.savefig(out_dir / 'loss_curves.png', dpi=300)
    plt.close(animator.fig)


def main():
    device = try_gpu()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    batch_size = 128
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),          # int [0, 255] -> Pytorch Tensor FP32 [0.0, 1.0]
        torchvision.transforms.Normalize(0.5, 0.5)  # (x - 0.5)/0.5, [0, 1] -> [-1, 1]
    ])
    anime_face = torchvision.datasets.ImageFolder(
        DATA_DIR / 'anime_face',
        transform=transformer,
        loader=load_rgb_image)
    num_workers = min(8, os.cpu_count() or 1)
    data_iter = torch.utils.data.DataLoader(
        anime_face,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True)

    # Close the matplotlib / torch warning output
    warnings.filterwarnings('ignore')

    # Generator: identical to the anime-face DCGAN (BatchNorm is fine here —
    # the Lipschitz constraint only applies to the critic).
    n_G = 64
    net_gen = nn.Sequential(
        G_block(in_channels=100, out_channels=n_G*8,
                strides=1, padding=0),                  # Output: (64 * 8, 4, 4)
        G_block(in_channels=n_G*8, out_channels=n_G*4), # Output: (64 * 4, 8, 8)
        G_block(in_channels=n_G*4, out_channels=n_G*2), # Output: (64 * 2, 16, 16)
        G_block(in_channels=n_G*2, out_channels=n_G),   # Output: (64, 32, 32)
        nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh())  # Output: (3, 64, 64)

    # Critic: no sigmoid (raw score), no BatchNorm (see module docstring).
    n_D = 64
    net_critic = nn.Sequential(
        D_block(n_D),  # Output: (64, 32, 32)
        D_block(in_channels=n_D, out_channels=n_D*2),  # Output: (64 * 2, 16, 16)
        D_block(in_channels=n_D*2, out_channels=n_D*4),  # Output: (64 * 4, 8, 8)
        D_block(in_channels=n_D*4, out_channels=n_D*8),  # Output: (64 * 8, 4, 4)
        nn.Conv2d(in_channels=n_D*8, out_channels=1,
                  kernel_size=4, bias=False))  # Output: (1, 1, 1)

    lr, latent_dim, num_epochs = 1e-4, 100, 120
    # WGAN-GP (gradient penalty) — the recommended default
    train(net_critic, net_gen, data_iter, num_epochs, latent_dim, OUT_DIR,
          lipschitz='gp', lr=lr, n_critic=5, lambda_gp=10, device=device)
    # WGAN (weight clipping) — uncomment to compare (also change out_dir above)
    # train(net_critic, net_gen, data_iter, num_epochs, latent_dim, OUT_DIR,
    #       lipschitz='clip', n_critic=5, clip_value=0.01)


if __name__ == '__main__':
    main()
