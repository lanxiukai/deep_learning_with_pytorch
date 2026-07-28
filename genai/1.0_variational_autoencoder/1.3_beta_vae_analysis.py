"""
Analyze a previously trained beta-VAE checkpoint (run 1.2_beta_vae_train.py first).

Besides the standard VAE checks (reconstruction fidelity, random samples),
this script focuses on what beta-VAE is designed for — disentanglement:

1. original vs. reconstructed: expect visibly blurrier reconstructions than
   the beta=1 model (the fidelity cost of the disentanglement trade-off)
2. per-dimension KL: beta>1 forces most latent dims to match the prior
   (KL ~ 0, "inactive"), concentrating information in a few active dims
3. latent traversal of the most active dims: the signature beta-VAE plot —
   if disentanglement worked, each row varies a single interpretable factor
   (pose, lighting, glasses, ...) while everything else stays fixed
4. interpolation between two images: check latent-space smoothness

Compare these figures side by side with output/vae/vae_analysis/ (beta=1)
to see the effect of beta.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.filesystem.directories import reset_dir
from dl_utils.genai.vae import VAE, device
from dl_utils.plot.images import vae_sample_grid

def main():
    root = infer_project_root()
    out_dir = root / "output" / "beta_vae" / "beta_vae_analysis"
    reset_dir(str(out_dir))

    checkpoint_path = root / "output/beta_vae/BetaVAEglasses.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run 1.2_beta_vae_train.py first."
        )

    latent_dims = 100
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True))
    vae.eval()
    loader = image_folder_loader(
        root / "data/glasses", batch_size=16, resize=256, shuffle=False
        )

    # ---- compare original images with reconstructions ----
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    mu, std, out = vae(imgs) # reconstruct imgs
    # ---- interleave original and reconstruction blocks ----
    nrow = 8
    images_list = []
    for i in range(0, imgs.shape[0], nrow):
        images_list.append(imgs[i:i+nrow])   # original blocks
        images_list.append(out[i:i+nrow])    # reconstruction blocks（under the original blocks）
    images = torch.cat(images_list, dim=0).detach().cpu()
    images = torchvision.utils.make_grid(images, nrow, 4)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.axis("off")
    fig.savefig(
        out_dir / "beta_vae_original_vs_reconstructed.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- generate random faces from latent space ----
    vae_sample_grid(
        vae.decoder, latent_dims, device,
        out_dir / "beta_vae_samples_grid.png",
    )

    # ============================================================
    # Disentanglement analysis — what beta-VAE is actually for
    # ============================================================

    # ---- per-dimension KL: which latent dims carry information? ----
    # A dim with mean KL ~ 0 has posterior == prior, i.e. beta "switched it
    # off". beta>1 models typically keep only a handful of dims active.
    with torch.no_grad():
        kl_sum = torch.zeros(latent_dims).to(device)
        num_samples = 0
        for imgs_kl, _ in loader:
            imgs_kl = imgs_kl.to(device)
            mu_kl, std_kl, _ = vae.encoder(imgs_kl)
            kl_sum += ((std_kl**2)/2 + (mu_kl**2)/2
                       - torch.log(std_kl) - 0.5).sum(dim=0)
            num_samples += imgs_kl.shape[0]
    kl_per_dim = kl_sum / num_samples          # (100,) mean KL per dim, in nats
    active_dims = torch.argsort(kl_per_dim, descending=True)
    top10 = active_dims[:10].tolist()
    print("top-10 most active dims (by mean KL):", top10)
    print("their mean KL (nats):",
          [round(float(kl_per_dim[d]), 3) for d in top10])
    print("dims with mean KL > 0.1 nats:",
          int((kl_per_dim > 0.1).sum()), "/", latent_dims)

    # ---- latent traversal of the 6 most active dims ----
    # Fix z at the encoded mean of one image, then sweep one dim at a time
    # across the prior's typical range [-3, 3]. Rows = dims, cols = values.
    traversal_dims = active_dims[:6]
    z0 = mu[0]                                  # (100,) deterministic point estimate
    values = torch.linspace(-3, 3, 9).to(device)
    with torch.no_grad():
        traversed = []
        for d in traversal_dims:
            for v in values:
                z = z0.clone()
                z[d] = v
                traversed.append(vae.decoder(z.unsqueeze(0)))
    grid = torch.cat(traversed, dim=0).detach().cpu()
    grid = torchvision.utils.make_grid(grid, nrow=len(values), padding=1)
    grid = np.transpose(grid, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    plt.imshow(grid)
    plt.title("latent traversal: 6 most active dims (rows) "
              "swept over [-3, 3] (cols)", fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "beta_vae_latent_traversal.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- the base image being traversed, for reference ----
    pair = torch.cat((imgs[0].unsqueeze(0), out[0].unsqueeze(0)),
                     dim=0).detach().cpu()
    pair = torchvision.utils.make_grid(pair, 2, 1)
    pair = np.transpose(pair, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    plt.imshow(pair)
    plt.title("traversal base: original (left) vs. reconstruction (right)",
              fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "beta_vae_traversal_base.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- interpolation between two images ----
    results = []
    for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        z = w * mu[0] + (1 - w) * mu[1]
        results.append(vae.decoder(z.unsqueeze(0)))
    imgs_interp = torch.cat(results, dim=0).detach().cpu()
    imgs_interp = torchvision.utils.make_grid(imgs_interp, 6, 1).numpy()
    imgs_interp = np.transpose(imgs_interp, (1, 2, 0))
    fig, ax = plt.subplots(dpi=300)
    plt.imshow(imgs_interp)
    plt.axis("off")
    plt.title("interpolation between two encoded images")
    fig.savefig(
        out_dir / "beta_vae_interpolation.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()


if __name__ == "__main__":
    main()
