"""
AutoEncoder
"""

import os

import torch
import torch.nn.functional as F
from torch import nn
from dl_utils.plot.figures import Animator
import matplotlib.pyplot as plt
from dl_utils.data.vision import vision_loaders
from dl_utils.devices.selection import get_device
from dl_utils.filesystem.directories import reset_dir
from dl_utils.plot.images import show_images


class AE(nn.Module):
    def __init__(self,input_dim, z_dim, h_dim):
        super().__init__()
        # shared input layer: maps raw input to hidden space
        self.common = nn.Linear(input_dim, h_dim)
        # encoder output: compresses hidden representation to latent bottleneck
        self.encoded = nn.Linear(h_dim, z_dim)
        # decoder hidden layer: expands latent code back to hidden space
        self.l1 = nn.Linear(z_dim, h_dim)
        # decoder output: reconstructs hidden representation to original input space
        self.decode = nn.Linear(h_dim, input_dim)                
    def encoder(self, x):
        common = F.relu(self.common(x))
        mu = self.encoded(common)
        return mu
    def decoder(self, z):
        out=F.relu(self.l1(z))
        out=torch.sigmoid(self.decode(out))
        return out
    def forward(self, x):
        mu=self.encoder(x)
        out=self.decoder(mu)
        return out, mu


def plot_digits(originals, model, device, input_dim, out_dir, epoch=None):
    reconstructed = []
    for idx in range(10):
        with torch.no_grad():
            img = originals[idx].reshape(1, input_dim)
            out, _ = model(img.to(device))
        reconstructed.append(out.reshape(28, 28))
    originals_2d = [o.reshape(28, 28) for o in originals]
    show_images(originals_2d + reconstructed, 2, 10, cmap="binary")
    tag = f"_epoch_{epoch}" if epoch is not None else ""
    path = os.path.join(out_dir, f"ae_recon{tag}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved -> {path}")


def main():
    batch_size = 32

    train_loader, test_loader = vision_loaders(
        'mnist', data_dir='data/mnist', batch_size=batch_size, 
        num_workers=4, pin_memory=True
    )
    num_samples = len(train_loader.dataset)  # type: ignore[arg-type]

    out_dir = "output/ae_mnist"
    reset_dir(out_dir)

    device = get_device()
    input_dim = 784
    z_dim = 20   # dimension of latent variable of encoder
    h_dim = 200  # dimension of hidden layer of encoder/decoder

    model = AE(input_dim,z_dim,h_dim).to(device)
    lr = 0.00025
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pairs = []
    seen = set()
    for imgs, labels in test_loader:
        for img, label in zip(imgs, labels):
            lbl = label.item()
            if lbl not in seen:
                pairs.append((lbl, img))
                seen.add(lbl)
            if len(seen) == 10:
                break
        if len(seen) == 10:
            break
    pairs.sort(key=lambda x: x[0])
    originals = [img for _, img in pairs]

    plot_digits(originals, model, device, input_dim, out_dir, "init")

    animator = Animator(xlabel='epoch', ylabel='mean loss',
                        xlim=[1, 10], legend=['train loss'])

    for epoch in range(10):
        epoch_loss = 0.0  # the total loss
        for imgs, labels in train_loader:
            # reconstruct the images
            imgs = imgs.to(device).view(-1, input_dim)
            out, _ = model(imgs)
            # reconstruction loss (MSE), total loss of a mini-batch
            loss = ((out - imgs) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        animator.add(epoch + 1, epoch_loss / num_samples)
        plot_digits(originals, model, device, input_dim, out_dir, epoch)

    model_path = "output/ae_mnist/AEdigits.pt"

    # Export to TorchScript
    scripted = torch.jit.script(model) 
    scripted.save(model_path) 

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    plot_digits(originals, model, device, input_dim, out_dir, "final")


if __name__ == "__main__":
    main()
