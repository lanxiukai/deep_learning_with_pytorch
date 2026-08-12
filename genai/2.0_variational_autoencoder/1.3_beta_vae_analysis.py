"""Analyze beta-VAE rate, reconstructions, and latent-coordinate behavior.

Run ``1.2_beta_vae_train.py`` first.  The analysis asks four teaching
questions without treating a visually pleasing traversal as proof:

1. How much KL rate does every latent coordinate use?
2. What changes when one high-rate coordinate is traversed alone?
3. Does any coordinate align with the known glasses/no-glasses label that was
   not used for training?
4. Are posterior-mean reconstructions and interpolations coherent?

Mean per-coordinate KL above 0.1 nat is reported as an "active" heuristic,
not a theorem.  Label alignment is also diagnostic only: this face dataset has
uncontrolled pose, identity, lighting, and demographic correlations.  A
factor-controlled dataset and a multi-seed metric are needed for a defensible
disentanglement comparison.

Outputs:
    output/beta_vae/analysis/reconstructions.png
    output/beta_vae/analysis/prior_samples.png
    output/beta_vae/analysis/kl_per_dimension.png
    output/beta_vae/analysis/known_attribute_alignment.png
    output/beta_vae/analysis/latent_traversal.png
    output/beta_vae/analysis/traversal_base.png
    output/beta_vae/analysis/interpolation.png
"""

import torch
from torch.utils.data import DataLoader

from dl_utils.data.vision import image_folder_dataset
from dl_utils.devices.randomness import set_seed
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.vae import VAE, device, diagonal_gaussian_kl
from dl_utils.plot import _backend as _  # select a backend before pyplot
from dl_utils.plot.images import save_image_row_grid
from matplotlib import pyplot as plt


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "beta_vae" / "analysis"
CHECKPOINT_PATH = PROJECT_ROOT / "output" / "beta_vae" / "BetaVAEglasses.pth"

BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_REPRESENTATIVES_PER_CLASS = 4
NUM_PRIOR_SAMPLES = 18
NUM_TRAVERSAL_DIMS = 6
TRAVERSAL_VALUES = tuple(float(value) for value in range(-3, 4))
NUM_INTERPOLATION_STEPS = 7
ACTIVE_KL_THRESHOLD = 0.1
SEED = 42


def load_beta_vae(checkpoint_path):
    """Load the metadata-rich checkpoint made by the training lesson."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run 1.2_beta_vae_train.py first."
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    expected = {
        "format_version": 2,
        "model_name": "beta_vae",
        "dataset": "glasses-256",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint has {key}={checkpoint.get(key)!r}; "
                f"expected {value!r}. Retrain it with 1.2_beta_vae_train.py."
            )

    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    beta = checkpoint.get("beta")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model metadata is incomplete.")
    if not isinstance(beta, (int, float)) or beta < 0:
        raise ValueError("Checkpoint beta must be a non-negative number.")

    vae = VAE(**model_config).to(device)
    vae.load_state_dict(state_dict, strict=True)
    return vae.eval(), int(model_config["latent_dims"]), float(beta)


def make_dataset_and_loader():
    """Read every example once, without stochastic training augmentation."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Analysis data not found: {DATA_DIR}. "
            "Run tool_scripts/download_dataset_test.py first."
        )
    dataset = image_folder_dataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )
    return dataset, loader


def representative_examples(dataset):
    """Choose the first few examples from each class for fixed visual checks."""
    if len(dataset.classes) != 2:
        raise ValueError("This analysis expects exactly two ImageFolder classes.")
    targets = torch.as_tensor(dataset.targets)
    indices = []
    for class_index in range(len(dataset.classes)):
        class_indices = torch.nonzero(
            targets == class_index,
            as_tuple=False,
        ).flatten()
        if len(class_indices) < NUM_REPRESENTATIVES_PER_CLASS:
            raise ValueError("Each class needs enough representative examples.")
        indices.extend(
            class_indices[:NUM_REPRESENTATIVES_PER_CLASS].tolist()
        )
    images = torch.stack([dataset[index][0] for index in indices])
    labels = targets[indices]
    return images, labels


@torch.inference_mode()
def encode_dataset(vae, loader):
    """Collect posterior means, per-unit KL, and evaluation-only labels."""
    all_mu = []
    all_kl = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        mu, std = vae.encoder.statistics(images)
        all_mu.append(mu.cpu())
        all_kl.append(diagonal_gaussian_kl(mu, std).cpu())
        all_labels.append(labels)
    return (
        torch.cat(all_mu),
        torch.cat(all_kl),
        torch.cat(all_labels),
    )


def standardized_label_alignment(mu, labels):
    """Compute absolute standardized mean differences for the two labels."""
    classes = torch.unique(labels, sorted=True)
    if len(classes) != 2:
        raise ValueError("Label alignment requires exactly two classes.")
    first = mu[labels == classes[0]]
    second = mu[labels == classes[1]]
    pooled_scale = torch.sqrt(
        0.5
        * (
            first.var(dim=0, unbiased=False)
            + second.var(dim=0, unbiased=False)
        )
        + 1e-8
    )
    return (first.mean(dim=0) - second.mean(dim=0)).abs() / pooled_scale


def display_range(images):
    """Convert VAE [0, 1] images for the GAN-oriented row-grid helper."""
    return images.mul(2).sub(1)


@torch.inference_mode()
def save_reconstructions(vae, images, labels, class_names):
    """Use posterior means so reconstruction noise is not a confounder."""
    images = images.to(device)
    mu, _, reconstructed = vae.reconstruct(images, sample=False)
    save_image_row_grid(
        [display_range(images), display_range(reconstructed)],
        ["Original", "Decoder(mu)"],
        OUT_DIR / "reconstructions.png",
        title="Deterministic posterior-mean reconstructions",
        column_labels=[class_names[int(label)] for label in labels],
    )
    return mu.cpu()


@torch.inference_mode()
def save_prior_samples(vae, latent_dims):
    """Show unconditional generation from z sampled from N(0, I)."""
    torch.manual_seed(SEED + 1)
    z = torch.randn(NUM_PRIOR_SAMPLES, latent_dims, device=device)
    samples = vae.decoder(z).cpu()
    samples = samples.reshape(3, 6, *samples.shape[1:])
    save_image_row_grid(
        [display_range(row) for row in samples],
        ["Prior 1-6", "Prior 7-12", "Prior 13-18"],
        OUT_DIR / "prior_samples.png",
        title="Independent samples z ~ N(0, I)",
        column_labels=[f"Sample {index + 1}" for index in range(6)],
    )


def save_kl_profile(mean_kl):
    """Plot the coordinates using the most posterior rate."""
    count = min(20, len(mean_kl))
    order = torch.argsort(mean_kl, descending=True)[:count]
    values = mean_kl[order]
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
        try:
            positions = torch.arange(count).numpy()
            ax.bar(positions, values.numpy())
            ax.axhline(
                ACTIVE_KL_THRESHOLD,
                color="tab:red",
                linestyle="--",
                label=f"active heuristic ({ACTIVE_KL_THRESHOLD:g} nat)",
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [str(index) for index in order.tolist()],
                rotation=45,
            )
            ax.set_xlabel("latent coordinate (sorted by mean KL)")
            ax.set_ylabel("mean KL [nats / image]")
            ax.set_title("Top posterior-rate coordinates")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_DIR / "kl_per_dimension.png")
        finally:
            plt.close(fig)


def save_label_alignment(alignment, class_names):
    """Plot label association as an evaluation-only latent diagnostic."""
    count = min(20, len(alignment))
    order = torch.argsort(alignment, descending=True)[:count]
    values = alignment[order]
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
        try:
            positions = torch.arange(count).numpy()
            ax.bar(positions, values.numpy(), color="tab:orange")
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [str(index) for index in order.tolist()],
                rotation=45,
            )
            ax.set_xlabel("latent coordinate (sorted by effect size)")
            ax.set_ylabel("absolute standardized mean difference")
            ax.set_title(
                f"Evaluation-only alignment: {class_names[0]} vs "
                f"{class_names[1]} (not a disentanglement score)"
            )
            fig.tight_layout()
            fig.savefig(OUT_DIR / "known_attribute_alignment.png")
        finally:
            plt.close(fig)


@torch.inference_mode()
def save_latent_traversal(
    vae,
    base_image,
    base_mu,
    traversal_dims,
    mean_kl,
):
    """Change one high-rate coordinate at a time over the prior's core."""
    values = torch.tensor(TRAVERSAL_VALUES, device=device)
    codes = base_mu.to(device).repeat(len(traversal_dims) * len(values), 1)
    for row, dimension in enumerate(traversal_dims.tolist()):
        start = row * len(values)
        codes[start : start + len(values), dimension] = values
    traversed = vae.decoder(codes).cpu().reshape(
        len(traversal_dims),
        len(values),
        *base_image.shape,
    )
    save_image_row_grid(
        [display_range(row) for row in traversed],
        [
            f"z[{dimension}] KL={mean_kl[dimension]:.2f}"
            for dimension in traversal_dims.tolist()
        ],
        OUT_DIR / "latent_traversal.png",
        title="One-coordinate traversals; all other coordinates fixed",
        column_labels=[f"{value:g}" for value in TRAVERSAL_VALUES],
    )

    base_reconstruction = vae.decoder(base_mu[None].to(device)).cpu()
    save_image_row_grid(
        [
            display_range(base_image[None]),
            display_range(base_reconstruction),
        ],
        ["Original", "Decoder(mu)"],
        OUT_DIR / "traversal_base.png",
        title="Reference image for the latent traversal",
        column_labels=["Fixed base"],
    )


@torch.inference_mode()
def save_interpolation(vae, endpoint_mu, endpoint_labels, class_names):
    """Linearly interpolate two posterior means from different labels."""
    fractions = torch.linspace(
        0,
        1,
        NUM_INTERPOLATION_STEPS,
        device=device,
    )[:, None]
    codes = torch.lerp(
        endpoint_mu[:1].to(device),
        endpoint_mu[1:].to(device),
        fractions,
    )
    images = vae.decoder(codes).cpu()
    start_name = class_names[int(endpoint_labels[0])]
    end_name = class_names[int(endpoint_labels[1])]
    save_image_row_grid(
        [display_range(images)],
        [f"{start_name} to {end_name}"],
        OUT_DIR / "interpolation.png",
        title="Linear interpolation between two posterior means",
        column_labels=[
            f"t={index / (NUM_INTERPOLATION_STEPS - 1):.2f}"
            for index in range(NUM_INTERPOLATION_STEPS)
        ],
    )


def main():
    set_seed(SEED)
    vae, latent_dims, beta = load_beta_vae(CHECKPOINT_PATH)
    dataset, loader = make_dataset_and_loader()
    reset_dir(str(OUT_DIR))
    representative_images, representative_labels = representative_examples(
        dataset
    )
    representative_mu = save_reconstructions(
        vae,
        representative_images,
        representative_labels,
        dataset.classes,
    )
    save_prior_samples(vae, latent_dims)

    posterior_mu, per_sample_kl, labels = encode_dataset(vae, loader)
    mean_kl = per_sample_kl.mean(dim=0)
    alignment = standardized_label_alignment(posterior_mu, labels)
    active = mean_kl > ACTIVE_KL_THRESHOLD
    traversal_dims = torch.argsort(mean_kl, descending=True)[
        : min(NUM_TRAVERSAL_DIMS, latent_dims)
    ]

    save_kl_profile(mean_kl)
    save_label_alignment(alignment, dataset.classes)
    save_latent_traversal(
        vae,
        representative_images[0],
        representative_mu[0],
        traversal_dims,
        mean_kl,
    )

    first_label = representative_labels[0]
    second_index = torch.nonzero(
        representative_labels != first_label,
        as_tuple=False,
    ).flatten()[0].item()
    endpoint_indices = torch.tensor([0, second_index])
    save_interpolation(
        vae,
        representative_mu[endpoint_indices],
        representative_labels[endpoint_indices],
        dataset.classes,
    )

    top_kl = torch.argsort(mean_kl, descending=True)[:10]
    top_alignment = torch.argsort(alignment, descending=True)[:10]
    print(f"beta: {beta:g}")
    print(f"examples analyzed: {len(posterior_mu):,}")
    print(f"aggregate mean KL rate: {mean_kl.sum():.3f} nats/image")
    print(
        f"active dimensions (mean KL > {ACTIVE_KL_THRESHOLD:g}): "
        f"{int(active.sum())}/{latent_dims}"
    )
    print("top-10 dimensions by mean KL:", top_kl.tolist())
    print("their mean KL:", [round(float(mean_kl[d]), 3) for d in top_kl])
    print(
        "top-10 dimensions by glasses-label alignment:",
        top_alignment.tolist(),
    )
    print(
        "their standardized differences:",
        [round(float(alignment[d]), 3) for d in top_alignment],
    )


if __name__ == "__main__":
    main()
