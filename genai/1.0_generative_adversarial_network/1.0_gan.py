"""
Generative Adversarial Networks
"""

import torch
from torch import nn
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.filesystem.directories import reset_dir
from dl_utils.plot.figures import Animator, set_figsize
from dl_utils.data.vision import load_array
from dl_utils.training.timing import Timer
from dl_utils.training.metrics import Accumulator
from dl_utils.gan.gan import update_D, update_G
from dl_utils.plot._backend import pyplot as plt

def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data, out_dir):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    # Parameter initialization
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=1, figsize=(5, 2.5),
                            legend=['discriminator', 'generator'])

    for epoch in range(num_epochs):
        # Train one epoch
        timer = Timer()
        metric = Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.3f} examples/sec')

    # --- Save loss curve ---
    animator.fig.savefig(out_dir / 'loss_curve.png', dpi=300)
    plt.close(animator.fig)

    # --- Save scatter comparison (final generated vs real) ---
    Z = torch.normal(0, 1, size=(100, latent_dim))
    fake_X = net_G(Z).detach().numpy()
    set_figsize((4, 4))
    plt.scatter(data[:, 0], data[:, 1], label='real', alpha=0.5)
    plt.scatter(fake_X[:, 0], fake_X[:, 1], label='generated', alpha=0.5)
    plt.legend()
    plt.savefig(out_dir / 'scatter_comparison.png', dpi=300)
    plt.close()

def main():
    X = torch.normal(0.0, 1, (1000, 2))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b  # (1000, 2)

    out_dir = infer_project_root() / "output" / "gan" / "gan"
    reset_dir(str(out_dir))

    set_figsize()
    plt.scatter(
        data[:100, (0)].detach().numpy(), 
        data[:100, (1)].detach().numpy())
    plt.savefig(
        out_dir / "data_distribution.png", 
        dpi=300, bbox_inches="tight")

    print(f'The covariance matrix is\n{torch.matmul(A.T, A)}')

    batch_size = 8
    data_iter = load_array((data,), batch_size)

    net_G = nn.Sequential(nn.Linear(2, 2))

    net_D = nn.Sequential(
        nn.Linear(2, 5), nn.Tanh(),
        nn.Linear(5, 3), nn.Tanh(),
        nn.Linear(3, 1))

    lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
    train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
          latent_dim, data[:100].detach().numpy(), out_dir)

if __name__ == "__main__":
    main()
