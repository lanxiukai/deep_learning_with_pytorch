"""Train the introductory 256x256 variational autoencoder.

Training uses the clean two-class ``data/glasses-256`` cache. One fixed prior
batch is decoded after every epoch, so image changes reflect model training
rather than new random inputs. The final metadata-rich checkpoint stores only
model weights; this short FP32 run intentionally has no resume machinery.

Outputs:
    output/vae/vae/training/epoch_*.png: fixed-z prior samples
    output/vae/vae/vae.pth: final model checkpoint
    output/vae/vae/loss_curves.png: independent total, reconstruction, and KL panels
"""

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.figures import save_loss_panels
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.vae import VAE, diagonal_gaussian_kl

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "vae"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_PATH = OUT_DIR / "vae.pth"

NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
Z_DIM = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_FIXED_SAMPLES = 18
SAMPLE_GRID_COLUMNS = 6
SAMPLE_EVERY_EPOCHS = 10
SEED = 42

MODEL_CONFIG = {"z_dim": Z_DIM}
METRIC_NAMES = ("total", "reconstruction", "kl")


def vae_loss(images, reconstructions, mu, std):
    """Return per-image mean total, reconstruction, and KL objectives."""
    reconstruction_loss = (
        (reconstructions - images).square().flatten(1).sum(dim=1).mean()
    )
    kl_loss = diagonal_gaussian_kl(mu, std).sum(dim=1).mean()
    return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss


@torch.inference_mode()
def save_epoch_samples(model, fixed_z, output_path):
    """Decode the fixed prior batch and save one epoch comparison grid."""
    was_training = model.training
    model.eval()
    samples = model.decoder(fixed_z).cpu()
    save_image(samples, output_path, nrow=SAMPLE_GRID_COLUMNS)
    model.train(was_training)


def train_epoch(model, loader, optimizer, device, progress_bar):
    """Train one FP32 epoch and return example-weighted loss means."""
    model.train()
    metrics = MetricAccumulator(METRIC_NAMES, device=device)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        mu, std, decoded_images = model(images)
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            images, decoded_images, mu, std
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        metrics.update(
            (total_loss, reconstruction_loss, kl_loss),
            num_examples=images.shape[0],
        )
        progress_bar.update(1)
    return metrics.compute_finite()


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Training data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    reset_dir(str(OUT_DIR))
    reset_dir(str(TRAINING_DIR))
    set_seed(SEED)
    device = try_gpu()
    loader = image_folder_loader(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = VAE(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    fixed_z = torch.randn(NUM_FIXED_SAMPLES, Z_DIM, device=device)
    loss_history = {
        "epoch": [],
        "total": [],
        "reconstruction": [],
        "kl": [],
    }

    save_epoch_samples(model, fixed_z, TRAINING_DIR / "epoch_000.png")

    with tqdm(
        total=NUM_EPOCHS * len(loader),
        desc=f"Epoch 1/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, NUM_EPOCHS + 1):
            progress_bar.set_description(f"Epoch {epoch}/{NUM_EPOCHS}", refresh=False)
            metrics = train_epoch(
                model,
                loader,
                optimizer,
                device,
                progress_bar,
            )
            loss_history["epoch"].append(epoch)
            for name in METRIC_NAMES:
                loss_history[name].append(metrics[name])
            progress_bar.set_postfix(
                total=f"{metrics['total']:.3f}",
                reconstruction=f"{metrics['reconstruction']:.3f}",
                kl=f"{metrics['kl']:.3f}",
                refresh=False,
            )

            if epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_epoch_samples(
                    model,
                    fixed_z,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                )

    save_model_weights(
        model,
        CHECKPOINT_PATH,
        metadata={
            "model_name": "vae",
            "model_config": MODEL_CONFIG,
            "dataset": "glasses-256",
            "value_range": [0.0, 1.0],
        },
    )
    save_loss_panels(
        loss_history["epoch"],
        {
            "Total negative ELBO": {
                "Total loss": loss_history["total"],
            },
            "Reconstruction objective": {
                "Summed-pixel MSE per image": loss_history["reconstruction"],
            },
            "Prior regularization": {
                "KL nats per image": loss_history["kl"],
            },
        },
        OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
