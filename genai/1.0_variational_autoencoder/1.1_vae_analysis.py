"""
Analyze a previously trained Variational AutoEncoder checkpoint.
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
    out_dir = root / "output" / "vae" / "vae_analysis"
    reset_dir(str(out_dir))

    checkpoint_path = root / "output/vae/VAEglasses.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run 1.0_vae_train.py first."
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
        out_dir / "vae_original_vs_reconstructed.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- generate random faces from latent space ----
    vae_sample_grid(
        vae.decoder, latent_dims, device,
        out_dir / "vae_samples_grid.png",
    )

    # ============================================================
    # Encoding arithmetic — manipulate latent vectors for attribute control
    # ============================================================

    # access the underlying dataset for individual image lookup
    data = loader.dataset

    # ---- take the first 25 samples (glasses) and the last 25 (noglasses),
    #       then manually pick three men / three women from each group ----
    torch.manual_seed(0)  # set the cpu seed
    glasses = []
    plt.figure(figsize=(12, 12))
    for i in range(25):
        img, label = data[i]
        glasses.append(img)
        plt.subplot(5, 5, i + 1)  # place each image in a 5×5 subplot grid
        plt.imshow(img.numpy().transpose((1, 2, 0)))  # -> (H, W, C)
        plt.axis("off")
    plt.savefig(out_dir / "glasses_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # select three men's images with glasses
    men_g = [glasses[0], glasses[3], glasses[14]]
    # select three women's images with glasses
    women_g = [glasses[2], glasses[15], glasses[23]]

    noglasses = []
    plt.figure(figsize=(12, 12))
    for i in range(25):
        img, label = data[-i - 1]
        noglasses.append(img)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img.numpy().transpose((1, 2, 0)))
        plt.axis("off")
    plt.savefig(out_dir / "noglasses_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # select three men's images without glasses
    men_ng = [noglasses[1], noglasses[3], noglasses[11]]
    # select three women's images without glasses
    women_ng = [noglasses[7], noglasses[9], noglasses[14]]

    # ---- encode each group and obtain average latent vector ----
    men_g_batch = torch.cat((
        men_g[0].unsqueeze(0), men_g[1].unsqueeze(0), men_g[2].unsqueeze(0)),
        dim=0).to(device)  # -> (3, 3, 256, 256)
    _, _, men_g_encodings = vae.encoder(men_g_batch)  # (3, 100)
    men_g_encoding = men_g_encodings.mean(dim=0)      # (100,)
    men_g_recon = vae.decoder(men_g_encoding.unsqueeze(0))  # (1, 100) -> (1, 3, 256, 256)

    women_g_batch = torch.cat((
        women_g[0].unsqueeze(0), women_g[1].unsqueeze(0), women_g[2].unsqueeze(0)),
        dim=0).to(device)
    _, _, women_g_encodings = vae.encoder(women_g_batch)
    women_g_encoding = women_g_encodings.mean(dim=0)
    women_g_recon = vae.decoder(women_g_encoding.unsqueeze(0))

    men_ng_batch = torch.cat((
        men_ng[0].unsqueeze(0), men_ng[1].unsqueeze(0), men_ng[2].unsqueeze(0)),
        dim=0).to(device)
    _, _, men_ng_encodings = vae.encoder(men_ng_batch)
    men_ng_encoding = men_ng_encodings.mean(dim=0)
    men_ng_recon = vae.decoder(men_ng_encoding.unsqueeze(0))

    women_ng_batch = torch.cat((
        women_ng[0].unsqueeze(0), women_ng[1].unsqueeze(0), women_ng[2].unsqueeze(0)),
        dim=0).to(device)
    _, _, women_ng_encodings = vae.encoder(women_ng_batch)
    women_ng_encoding = women_ng_encodings.mean(dim=0)
    women_ng_recon = vae.decoder(women_ng_encoding.unsqueeze(0))

    # ---- display the four group reconstructions ----
    imgs = torch.cat(
        (men_g_recon, women_g_recon, men_ng_recon, women_ng_recon),
        dim=0)  # (4, 3, 256, 256)
    # make_grid (nrow=4, padding=1), -> (3, H+2, W*4+5)
    imgs = torchvision.utils.make_grid(imgs, 4, 1).cpu().numpy()
    # C,H,W → H,W,C for imshow
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)
    plt.imshow(imgs)
    plt.title("encode each group and obtain average latent vector",
          fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "vae_average_latent_vector.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- encoding arithmetic: man without glasses ----
    z = men_g_encoding - women_g_encoding + women_ng_encoding
    out = vae.decoder(z.unsqueeze(0))
    imgs = torch.cat((men_g_recon, women_g_recon, women_ng_recon, out), dim=0)
    imgs = torchvision.utils.make_grid(imgs, 4, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)
    plt.imshow(imgs)
    plt.title("man with glasses - woman with glasses "
              "+ woman without glasses = man without glasses",
              fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "vae_exercise_7_1_0.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- Exercise 7.1 (i): woman with glasses ----
    z = men_g_encoding - men_ng_encoding + women_ng_encoding
    out = vae.decoder(z.unsqueeze(0))
    imgs = torch.cat((men_g_recon, men_ng_recon, women_ng_recon, out), dim=0)
    imgs = torchvision.utils.make_grid(imgs, 4, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)
    plt.imshow(imgs)
    plt.title("man with glasses - man without glasses "
              "+ woman without glasses = woman with glasses",
              fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "vae_exercise_7_1_i.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- Exercise 7.1 (ii): man with glasses ----
    z = men_ng_encoding - women_ng_encoding + women_g_encoding
    out = vae.decoder(z.unsqueeze(0))
    imgs = torch.cat((men_ng_recon, women_ng_recon, women_g_recon, out), dim=0)
    imgs = torchvision.utils.make_grid(imgs, 4, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)
    plt.imshow(imgs)
    plt.title("man without glasses - woman without glasses "
              "+ woman with glasses = man with glasses",
              fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "vae_exercise_7_1_ii.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- Exercise 7.1 (iii): woman with glasses ----
    z = women_ng_encoding - men_ng_encoding + men_g_encoding
    out = vae.decoder(z.unsqueeze(0))
    imgs = torch.cat((women_ng_recon, men_ng_recon, men_g_recon, out), dim=0)
    imgs = torchvision.utils.make_grid(imgs, 4, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(8, 2), dpi=300)
    plt.imshow(imgs)
    plt.title("woman without glasses - man without glasses "
              "+ man with glasses = woman with glasses",
              fontsize=10, c="r")
    plt.axis("off")
    fig.savefig(
        out_dir / "vae_exercise_7_1_iii.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- interpolation: woman with glasses → woman without glasses ----
    results = []
    for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        z = w * women_ng_encoding + (1 - w) * women_g_encoding
        out = vae.decoder(z.unsqueeze(0))  # (1, 100) -> (1, 3, H, W)
        results.append(out)
    imgs = torch.cat(results, dim=0)  # -> (6, 3, H, W)
    imgs = torchvision.utils.make_grid(imgs, 6, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(dpi=300)
    plt.imshow(imgs)
    plt.axis("off")
    plt.title("woman with glasses → woman without glasses")
    fig.savefig(
        out_dir / "vae_exercise_7_2_0.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- Exercise 7.2 (i): man with glasses → man without glasses ----
    results = []
    for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        z = w * men_ng_encoding + (1 - w) * men_g_encoding
        out = vae.decoder(z.unsqueeze(0))
        results.append(out)
    imgs = torch.cat(results, dim=0)
    imgs = torchvision.utils.make_grid(imgs, 6, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(dpi=300)
    plt.imshow(imgs)
    plt.axis("off")
    plt.title("man with glasses → man without glasses")
    fig.savefig(
        out_dir / "vae_exercise_7_2_i.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- Exercise 7.2 (ii): woman without glasses → man without glasses ----
    results = []
    for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        z = w * men_ng_encoding + (1 - w) * women_ng_encoding
        out = vae.decoder(z.unsqueeze(0))
        results.append(out)
    imgs = torch.cat(results, dim=0)
    imgs = torchvision.utils.make_grid(imgs, 6, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(dpi=300)
    plt.imshow(imgs)
    plt.axis("off")
    plt.title("woman without glasses → man without glasses")
    fig.savefig(
        out_dir / "vae_exercise_7_2_ii.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ---- Exercise 7.2 (iii): woman with glasses → man with glasses ----
    results = []
    for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        z = w * men_g_encoding + (1 - w) * women_g_encoding
        out = vae.decoder(z.unsqueeze(0))
        results.append(out)
    imgs = torch.cat(results, dim=0)
    imgs = torchvision.utils.make_grid(imgs, 6, 1).cpu().numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    fig, ax = plt.subplots(dpi=300)
    plt.imshow(imgs)
    plt.axis("off")
    plt.title("woman with glasses → man with glasses")
    fig.savefig(
        out_dir / "vae_exercise_7_2_iii.png",
        bbox_inches="tight")
    plt.close(fig)
    # plt.show()


if __name__ == "__main__":
    main()
