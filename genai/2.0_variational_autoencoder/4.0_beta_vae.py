"""Train a 256x256 beta-VAE on the same setup as ``1.0_vae.py``.

The model, data, reconstruction term, optimizer, and training loop are shared
with the standard VAE lesson. Only beta changes the weight of the complete
per-image KL term.
"""

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae import (
    reconstruction_and_kl,
    train_glasses_vae,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "beta_vae"
MODEL_CONFIG = {"z_dim": 100}
BETA = 4.0


def beta_vae_loss(images, reconstructions, mu, std):
    """Return the beta-VAE objective: reconstruction + beta * KL."""
    reconstruction_loss, kl_loss = reconstruction_and_kl(
        images, reconstructions, mu, std
    )
    return (
        reconstruction_loss + BETA * kl_loss,
        reconstruction_loss,
        kl_loss,
    )


def main():
    train_glasses_vae(
        loss_function=beta_vae_loss,
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        checkpoint_name="beta_vae.pth",
        model_name="beta_vae",
        model_config=MODEL_CONFIG,
        metadata={"beta": BETA},
    )


if __name__ == "__main__":
    main()
