"""Train the introductory 256x256 VAE on ``glasses-256`` images."""

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae import (
    reconstruction_and_kl,
    train_glasses_vae,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "vae"
MODEL_CONFIG = {"z_dim": 100}


def vae_loss(images, reconstructions, mu, std):
    """Return the standard VAE objective: reconstruction + KL."""
    reconstruction_loss, kl_loss = reconstruction_and_kl(
        images, reconstructions, mu, std
    )
    return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss


def main():
    train_glasses_vae(
        loss_function=vae_loss,
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        checkpoint_name="vae.pth",
        model_name="vae",
        model_config=MODEL_CONFIG,
    )


if __name__ == "__main__":
    main()
