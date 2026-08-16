"""Train beta-VAE while keeping beta as the visible algorithmic change.

For one image x, the beta-VAE rate-distortion objective (Higgins et al., 2017)
is

    distortion(x) + beta * rate(x)

    distortion = sum_pixels (x - decoder(z)) ** 2
    rate       = KL[q(z | x) || N(0, I)]

``beta=1`` recovers the standard VAE objective.  A value above one spends less
latent-channel capacity in exchange for pressure toward factorized codes.  It
does not guarantee disentanglement: the data distribution, random seed, and
decoder likelihood still matter.

This lesson deliberately reuses ``dl_utils.vae.vae.VAE``, the 256x256
glasses dataset, and the optimizer from ``1.0_vae_train.py``.  The weighted KL
is therefore the only learning-algorithm increment.  No mixed precision,
compiled kernels, distributed training, or hyperparameter search is hidden in
the script.  The 63.3 M-parameter model with a batch of 16 is the intended
12 GB single-card configuration.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py.

Outputs:
    output/beta_vae/training/epoch_*.png: fixed-prior samples by epoch
    output/beta_vae/BetaVAEglasses.pth: model plus beta/config metadata
    output/beta_vae/loss_curves.png: distortion, rate, and weighted objective

Examples:
    python genai/2.0_variational_autoencoder/1.2_beta_vae_train.py
    python genai/2.0_variational_autoencoder/1.2_beta_vae_train.py --beta 6

Training data -- glasses-256:
Training images:         4,500
Samples per epoch:       4,496 (281 full batches; drop_last=True)
Training epochs:         10
Optimizer updates:       2,810

Encoder:                 31.70 M params
Decoder:                 31.63 M params
Total:                   63.33 M params
"""

import argparse

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from dl_utils.data.vision import image_folder_loader
from dl_utils.runtime.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae import VAE, device, diagonal_gaussian_kl
from dl_utils.plot.figures import save_loss_panels
from dl_utils.plot.images import save_grid


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "beta_vae"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_PATH = OUT_DIR / "BetaVAEglasses.pth"

NUM_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
LATENT_DIMS = 100
BETA = 4.0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_FIXED_SAMPLES = 18
SAMPLE_GRID_COLUMNS = 6
SEED = 42

MODEL_CONFIG = {"latent_dims": LATENT_DIMS}
METRIC_NAMES = ("loss", "reconstruction", "kl", "weighted_kl")


def beta_vae_loss(reconstructed, images, mu, std, beta):
    """Return the batch-summed objective and per-image mean diagnostics."""
    if reconstructed.shape != images.shape:
        raise ValueError("reconstructed images must match the input shape.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    distortion = F.mse_loss(
        reconstructed,
        images,
        reduction="none",
    ).flatten(1).sum(dim=1)
    # Per-dimension formula lives in the shared VAE lesson primitive:
    # 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2)).
    rate = diagonal_gaussian_kl(mu, std).sum(dim=1)
    per_image_loss = distortion + float(beta) * rate
    mean_distortion = distortion.mean()
    mean_rate = rate.mean()
    weighted_rate = float(beta) * mean_rate
    # Sum over the mini-batch, matching 1.0_vae_train.py exactly.  Curves
    # remain per-image means, so batch size does not change their scale.
    loss = per_image_loss.sum()
    terms = {
        "loss": per_image_loss.mean().detach(),
        "reconstruction": mean_distortion.detach(),
        "kl": mean_rate.detach(),
        "weighted_kl": weighted_rate.detach(),
    }
    return loss, terms


def train_epoch(vae, loader, optimizer, beta, progress_bar=None):
    """Run one beta-VAE epoch with the rate-distortion terms explicit."""
    vae.train()
    metric_sums = torch.zeros(len(METRIC_NAMES), device=device)
    num_examples = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)  # [B, 3, 256, 256]
        mu, std, reconstructed = vae(images)  # [B, 100], [B, 100], image
        loss, terms = beta_vae_loss(
            reconstructed,
            images,
            mu,
            std,
            beta,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        metric_sums += torch.stack(
            [terms[name] for name in METRIC_NAMES]
        ) * batch_size
        num_examples += batch_size
        if progress_bar is not None:
            progress_bar.update(1)

    return dict(
        zip(
            METRIC_NAMES,
            (value / num_examples for value in metric_sums.tolist()),
            strict=True,
        )
    )


@torch.inference_mode()
def save_fixed_prior_samples(vae, fixed_z, output_path, epoch):
    """Decode the same prior draws each epoch so learning is comparable."""
    was_training = vae.training
    vae.eval()
    try:
        samples = vae.decoder(fixed_z).cpu()
    finally:
        vae.train(was_training)
    save_grid(
        samples,
        output_path,
        nrow=SAMPLE_GRID_COLUMNS,
        title=f"beta-VAE fixed prior samples - epoch {epoch:02d}",
    )


def main(beta=BETA, num_epochs=NUM_EPOCHS):
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Training data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset.py first."
        )

    reset_dir(str(OUT_DIR))
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    loader = image_folder_loader(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    if not isinstance(loader.dataset, torchvision.datasets.ImageFolder):
        raise TypeError("Expected glasses-256 to use ImageFolder layout.")

    vae = VAE(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    fixed_z = torch.randn(NUM_FIXED_SAMPLES, LATENT_DIMS, device=device)
    history = {
        "epoch": [],
        **{name: [] for name in METRIC_NAMES},
    }

    with tqdm(
        total=num_epochs * len(loader),
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(
                f"beta-VAE (beta={beta:g}) | Epoch {epoch}/{num_epochs}",
                refresh=False,
            )
            metrics = train_epoch(
                vae,
                loader,
                optimizer,
                beta,
                progress_bar,
            )
            history["epoch"].append(epoch)
            for name, value in metrics.items():
                history[name].append(value)
            save_fixed_prior_samples(
                vae,
                fixed_z,
                TRAINING_DIR / f"epoch_{epoch:02d}.png",
                epoch,
            )

    save_loss_panels(
        history["epoch"],
        {
            "Rate-distortion objective": {
                "total": history["loss"],
                "distortion": history["reconstruction"],
            },
            "Latent rate": {
                "KL": history["kl"],
                f"beta * KL (beta={beta:g})": history["weighted_kl"],
            },
        },
        OUT_DIR / "loss_curves.png",
    )
    torch.save(
        {
            "format_version": 2,
            "model_name": "beta_vae",
            "model_config": MODEL_CONFIG,
            "state_dict": vae.state_dict(),
            "beta": float(beta),
            "objective": "reconstruction_plus_beta_kl",
            "reconstruction": "summed_squared_error_per_image",
            "dataset": "glasses-256",
            "epoch": int(num_epochs),
        },
        CHECKPOINT_PATH,
    )
    print(
        f"Saved beta-VAE (beta={beta:g}, epochs={num_epochs}) to "
        f"{CHECKPOINT_PATH}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(beta=args.beta, num_epochs=args.epochs)
