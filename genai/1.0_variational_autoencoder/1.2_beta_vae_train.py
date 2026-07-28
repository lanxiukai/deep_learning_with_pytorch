"""
Train a beta-VAE and save its checkpoint.

Encoder:  31.7 M params
Decoder:  31.6 M params
Total:    63.3 M params

beta-VAE (Higgins et al., 2017) is a standard VAE whose ELBO weights the
KL-divergence term by a scalar beta:

    loss = reconstruction_loss + beta * kl

- beta = 1  -> exactly the standard VAE (see 1.0_vae_train.py)
- beta > 1  -> stronger pressure on q(z|x) to match the prior N(0, I),
               which encourages disentangled latent factors but blurs
               reconstructions (the disentanglement / fidelity trade-off)
- beta < 1  -> sharper reconstructions, weaker latent regularization

This script is identical to 1.0_vae_train.py except for the beta weighting,
a separate output directory, and per-component loss curves so you can watch
how beta reshapes the balance between the two terms.
"""

import torch
import torchvision

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.filesystem.directories import reset_dir
from dl_utils.genai.vae import VAE, device
from dl_utils.plot.figures import Animator
from dl_utils.plot.images import vae_sample_grid
from dl_utils.training.timing import Timer

BETA = 4.0  # the only change vs. the standard VAE objective

def main():
    root = infer_project_root()
    out_dir = root / "output" / "beta_vae"
    reset_dir(str(out_dir))
    (out_dir / "beta_vae_epoch").mkdir(exist_ok=True)

    loader = image_folder_loader(
        root / "data/glasses", batch_size=16, resize=256
        )

    data = loader.dataset
    latent_dims = 100
    assert isinstance(data, torchvision.datasets.ImageFolder)

    num_samples = len(data)
    vae = VAE().to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(vae.parameters(),
                                 lr=lr, weight_decay=1e-5)
    animator = Animator(xlabel='epoch', ylabel='mean loss',
                        xlim=[0, 10],
                        legend=['total loss', 'recon loss', f'{BETA:g}*kl'])

    def train_epoch(epoch):
        vae.train()
        # window accumulators reset after every plotted point, so each point
        # is the mean loss since the previous point. (A cumulative mean that
        # resets every epoch instead produces a sawtooth drop at each integer
        # epoch: full-epoch mean -> first-few-batches mean.)
        window_loss, window_recon, window_kl = 0.0, 0.0, 0.0
        window_samples = 0
        seen_samples, num_batches = 0, len(loader)
        update_interval = max(1, (num_batches + 19) // 20)
        for batch_idx, (imgs, _) in enumerate(loader, start=1):
            imgs = imgs.to(device)
            batch_samples = imgs.shape[0]
            seen_samples += batch_samples
            window_samples += batch_samples
            mu, std, out = vae(imgs)
            # reconstruction loss
            reconstruction_loss = ((imgs-out)**2).sum()
            # kl loss
            kl = ((std**2)/2 + (mu**2)/2 - torch.log(std) - 0.5).sum()
            # total loss of a mini-batch: the only line beta-VAE changes
            loss = reconstruction_loss + BETA * kl
            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            window_loss += loss.item()
            window_recon += reconstruction_loss.item()
            window_kl += kl.item()
            if batch_idx % update_interval == 0 or batch_idx == num_batches:
                animator.add(epoch - 1 + seen_samples / num_samples,
                             (window_loss / window_samples,
                              window_recon / window_samples,
                              BETA * window_kl / window_samples))
                window_loss, window_recon, window_kl = 0.0, 0.0, 0.0
                window_samples = 0

    num_epochs = 10
    timer = Timer()
    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch)
        vae_sample_grid(
            vae.decoder, latent_dims, device,
            out_dir / "beta_vae_epoch" / f"beta_vae_samples_epoch_{epoch:02d}.png"
        )
    elapsed = timer.stop()
    animator.fig.savefig(out_dir / 'loss_curve.png', dpi=300)
    torch.save(vae.state_dict(), out_dir / "BetaVAEglasses.pth")
    total_tenths = round(elapsed * 10)
    hours, remainder = divmod(total_tenths, 36000)
    minutes, second_tenths = divmod(remainder, 600)
    seconds = second_tenths / 10
    avg_tenths = round(elapsed / num_epochs * 10)
    avg_minutes, avg_second_tenths = divmod(avg_tenths, 600)
    avg_seconds = avg_second_tenths / 10
    print(f'training time {hours} h {minutes} min {seconds:.1f} s, '
          f'avg {avg_minutes} min {avg_seconds:.1f} s/epoch')


if __name__ == "__main__":
    main()
