"""
Deep Convolutional Generative Adversarial Networks

Generator:       3.6 M params
Discriminator:   2.8 M params
Total:           6.4 M params
"""

import warnings
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.filesystem.directories import reset_dir
from dl_utils.devices.selection import try_gpu
from dl_utils.genai.gan import update_D, update_G
from dl_utils.plot.figures import Animator, plot, set_figsize
from dl_utils.training.metrics import Accumulator
from dl_utils.training.timing import Timer


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
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, out_dir,
          device=try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                        legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    
    timer = Timer()
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        metric = Accumulator(3)  # loss_D, loss_G, num_examples
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D, real_label=0.9),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)
        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(  # → (3×H, 7×W, C)
            [torch.cat([   # → (H, 7×W, C)
                fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim=1)
             for i in range(len(fake_x)//7)], dim=0)
        if epoch == 1 or epoch % 2 == 0:
            plt.imsave(out_dir / f'epoch_{epoch:03d}.png', imgs.numpy())
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs)
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    total_time = timer.stop()
    total_tenths = round(total_time * 10)
    hours, remainder = divmod(total_tenths, 36000)
    minutes, second_tenths = divmod(remainder, 600)
    seconds = second_tenths / 10
    avg_tenths = round(total_time / num_epochs * 10)
    avg_minutes, avg_second_tenths = divmod(avg_tenths, 600)
    avg_seconds = avg_second_tenths / 10
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'training time {int(hours)} h {int(minutes)} min {seconds:.1f} s, '
          f'avg {avg_minutes} min {avg_seconds:.1f} s/epoch '
          f'on {str(device)}')
    animator.fig.savefig(out_dir / 'loss_curves.png', dpi=300)
    plt.close(animator.fig)


def main():
    out_dir = infer_project_root() / 'output' / 'dcgan' / 'pokemon'
    reset_dir(str(out_dir))

    pokemon = torchvision.datasets.ImageFolder(
        infer_project_root() / 'data' / 'pokemon')

    batch_size = 128
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),          # int [0, 255] -> Pytorch Tensor FP32 [0.0, 1.0]
        torchvision.transforms.Normalize(0.5, 0.5)  # (x - 0.5)/0.5, [0, 1] -> [-1, 1]
    ])
    pokemon.transform = transformer
    data_iter = torch.utils.data.DataLoader(
        pokemon, batch_size=batch_size,
        shuffle=True, num_workers=4)

    # Close the matplotlib / torch warning output
    warnings.filterwarnings('ignore')
    for X, y in data_iter:
        # [20, C, H, W] -> [20, H, W, C]; [-1, 1] -> [0, 1]
        imgs = X[:20,:,:,:].permute(0, 2, 3, 1)/2+0.5
        # set_figsize controls the figure size via rcParams;
        # plt.subplots() respects it (unlike show_images which overrides figsize)
        set_figsize((4, 5))
        num_rows, num_cols = 4, 5
        _, axes = plt.subplots(num_rows, num_cols)
        for ax, img in zip(axes.flatten(), imgs):
            ax.imshow(img.cpu().numpy())
            ax.axis('off')
        plt.savefig(out_dir / 'pokemon_samples.png', dpi=300)
        plt.close()
        break

    x = torch.zeros((2, 3, 16, 16))
    g_blk = G_block(20)
    print(g_blk(x).shape)  # torch.Size([2, 20, 32, 32])

    x = torch.zeros((2, 3, 1, 1))
    g_blk = G_block(20, strides=1, padding=0)
    print(g_blk(x).shape)  # torch.Size([2, 20, 4, 4])

    n_G = 64
    net_G = nn.Sequential(
        G_block(in_channels=100, out_channels=n_G*8,
                strides=1, padding=0),                  # Output: (64 * 8, 4, 4)
        G_block(in_channels=n_G*8, out_channels=n_G*4), # Output: (64 * 4, 8, 8)
        G_block(in_channels=n_G*4, out_channels=n_G*2), # Output: (64 * 2, 16, 16)
        G_block(in_channels=n_G*2, out_channels=n_G),   # Output: (64, 32, 32)
        nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh())  # Output: (3, 64, 64)

    x = torch.zeros((1, 100, 1, 1))
    print(net_G(x).shape)  # torch.Size([1, 3, 64, 64])

    alphas = [0, .2, .4, .6, .8, 1]
    x = torch.arange(-2, 1, 0.1)
    Y = [nn.LeakyReLU(alpha)(x).detach().numpy() for alpha in alphas]
    plot(x.detach().numpy(), Y, 'x', 'y', alphas)
    plt.savefig(out_dir / 'leaky_relu_curves.png', dpi=300)
    plt.close()

    x = torch.zeros((2, 3, 16, 16))
    d_blk = D_block(20)
    print(d_blk(x).shape)  # torch.Size([2, 20, 8, 8])

    n_D = 64
    net_D = nn.Sequential(
        D_block(n_D),  # Output: (64, 32, 32)
        D_block(in_channels=n_D, out_channels=n_D*2),  # Output: (64 * 2, 16, 16)
        D_block(in_channels=n_D*2, out_channels=n_D*4),  # Output: (64 * 4, 8, 8)
        D_block(in_channels=n_D*4, out_channels=n_D*8),  # Output: (64 * 8, 4, 4)
        nn.Conv2d(in_channels=n_D*8, out_channels=1,
                  kernel_size=4, bias=False))  # Output: (1, 1, 1)

    x = torch.zeros((1, 3, 64, 64))
    print(net_D(x).shape)  # torch.Size([1, 1, 1, 1])

    latent_dim, num_epochs = 100, 20
    lr_D, lr_G = 1e-4, 2e-4
    train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, out_dir)


if __name__ == '__main__':
    main()
