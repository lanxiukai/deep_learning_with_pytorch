"""Image-grid rendering primitives."""

import math
import os
from pathlib import Path
from typing import Any

from dl_utils.plot._backend import pyplot as _plt

import numpy as np
import torch
import torchvision

from dl_utils.gan.inference import generate_in_batches


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, cmap=None):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = _plt.subplots(num_rows, num_cols, figsize=figsize)
    axes: Any = axes
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor image
            ax.imshow(img.cpu().numpy(), cmap=cmap)
        else:
            # PIL image
            ax.imshow(img, cmap=cmap)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def save_grid(
    images: torch.Tensor,
    path: str,
    nrow: int,
    title: str | None = None,
    *,
    hw: tuple[int, int] | None = None,
) -> None:
    """
    Save a grid image.

    images:
      - [N, V] flattened (if hw is None, V must be a perfect square, e.g. 28*28)
      - [N, 1, H, W] image tensor
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if images.dim() == 2:
        V = images.size(1)
        if hw is not None:
            H, W = int(hw[0]), int(hw[1])
            if H * W != V:
                raise ValueError(f"save_grid: hw={hw} implies V={H*W}, but got V={V}.")
            images = images.reshape(-1, 1, H, W)
        else:
            side = int(math.isqrt(V))
            if side * side != V:
                raise ValueError(f"save_grid: cannot infer square image shape from V={V}. Pass hw=(H,W).")
            images = images.reshape(-1, 1, side, side)

    # Make a grid using torchvision then plot with matplotlib (portable)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid = grid.detach().cpu()

    _plt.figure(figsize=(8, 8))
    if title is not None:
        _plt.title(title)
    _plt.axis("off")
    _plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    _plt.tight_layout()
    _plt.savefig(path, dpi=200)
    _plt.close()


def save_image_row_grid(
    image_rows,
    row_labels,
    output_path,
    *,
    title=None,
    column_labels=None,
    dpi=300,
) -> None:
    """Save a titled row grid of ``C x H x W`` tensors in [-1, 1]."""
    if len(image_rows) == 0 or len(image_rows[0]) == 0:
        raise ValueError("image_rows must contain at least one image.")
    if len(image_rows) != len(row_labels):
        raise ValueError(
            "image_rows and row_labels must have the same length."
        )

    num_columns = len(image_rows[0])
    if any(len(images) != num_columns for images in image_rows):
        raise ValueError("All image rows must have the same length.")
    if column_labels is not None and len(column_labels) != num_columns:
        raise ValueError(
            "column_labels must have one label for every image column."
        )

    has_decorations = title is not None or column_labels is not None
    figure_height = (
        2.0 * len(image_rows) + 0.5
        if has_decorations
        else 2.1 * len(image_rows)
    )
    # ``Animator`` enables pyplot's global interactive mode.  Temporarily turn
    # it off so this save-only figure never creates a visible GUI window.
    with _plt.ioff():
        fig, axes = _plt.subplots(
            len(image_rows),
            num_columns,
            figsize=(1.8 * num_columns, figure_height),
            squeeze=False,
        )
        try:
            for row_index, (images, label) in enumerate(
                zip(image_rows, row_labels)
            ):
                axes[row_index, 0].set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=12,
                )
                for column_index, image in enumerate(images):
                    display_image = (
                        image.detach()
                        .cpu()
                        .mul(0.5)
                        .add(0.5)
                        .clamp(0, 1)
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    axes[row_index, column_index].imshow(display_image)
                    axes[row_index, column_index].set_xticks([])
                    axes[row_index, column_index].set_yticks([])
                    if has_decorations:
                        for spine in (
                            axes[row_index, column_index].spines.values()
                        ):
                            spine.set_visible(False)
                    if row_index == 0 and column_labels is not None:
                        axes[row_index, column_index].set_title(
                            column_labels[column_index],
                            fontsize=10,
                        )

            if title is not None:
                fig.suptitle(title, fontsize=14)
            if has_decorations:
                top = 0.94 if title is not None else 0.98
                fig.tight_layout(rect=(0, 0, 1, top), pad=0.6)
                fig.subplots_adjust(wspace=0.03, hspace=0.12)
            else:
                fig.subplots_adjust(wspace=0.02, hspace=0.08)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        finally:
            _plt.close(fig)


@torch.inference_mode()
def save_fixed_noise_samples(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    output_path: str | Path,
    *,
    columns: int,
    title: str = "Fixed-noise samples",
    epoch: int | None = None,
    dpi: int = 200,
    inference_batch_size: int | None = None,
) -> None:
    """Generate and save a fixed-noise grid for an unconditional model."""
    if columns < 1:
        raise ValueError("columns must be positive.")
    if noise.ndim < 1 or len(noise) == 0:
        raise ValueError("noise must contain at least one sample.")
    if len(noise) % columns != 0:
        raise ValueError("The number of noise samples must divide into rows.")

    sample_batch_size = (
        len(noise) if inference_batch_size is None else inference_batch_size
    )
    samples = generate_in_batches(
        noise,
        sample_batch_size,
        lambda batch: generator(batch).float(),
        module=generator,
    )
    if samples.ndim != 4 or len(samples) != len(noise):
        raise ValueError("generator must return an N x C x H x W tensor.")

    image_rows = samples.reshape(-1, columns, *samples.shape[1:])
    row_labels = [
        f"z {row * columns + 1:02d}-{(row + 1) * columns:02d}"
        for row in range(len(image_rows))
    ]
    display_title = (
        title if epoch is None else f"{title} - epoch {epoch:03d}"
    )
    save_image_row_grid(
        image_rows,
        row_labels,
        output_path,
        title=display_title,
        column_labels=[f"Sample {index + 1}" for index in range(columns)],
        dpi=dpi,
    )


def save_training_samples(
    generator,
    noise,
    labels,
    output_path,
    *,
    class_names,
    title,
    dpi=200,
    shared_latents_across_classes=False,
    inference_batch_size: int | None = None,
) -> None:
    """Generate and save fixed class-conditional samples grouped by class."""
    class_names = tuple(class_names)
    sample_batch_size = (
        len(noise) if inference_batch_size is None else inference_batch_size
    )
    samples = generate_in_batches(
        (noise, labels),
        sample_batch_size,
        lambda noise_batch, label_batch: generator(
            noise_batch,
            label_batch,
        ).float(),
        module=generator,
    )

    cpu_labels = labels.cpu()
    image_rows = [
        samples[cpu_labels == class_index]
        for class_index in range(len(class_names))
    ]
    save_image_row_grid(
        image_rows,
        [name.title() for name in class_names],
        output_path,
        title=title,
        column_labels=[
            (
                f"Shared z {index + 1}"
                if shared_latents_across_classes
                else f"Sample {index + 1}"
            )
            for index in range(len(image_rows[0]) if image_rows else 0)
        ],
        dpi=dpi,
    )


def vae_sample_grid(
    decoder: torch.nn.Module,
    latent_dims: int,
    device: torch.device,
    output_path: str | Path,
    show: bool = False,
) -> None:
    """Sample 18 vectors from N(0, 1) on ``device``, decode, render grid."""
    with torch.no_grad():
        noise = torch.randn(18, latent_dims).to(device)
        imgs = decoder(noise).cpu()
        grid = torchvision.utils.make_grid(imgs, 6, 3)
        fig, ax = _plt.subplots(figsize=(6, 3), dpi=300)
        _plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        _plt.axis("off")
        fig.savefig(output_path)
        if show:
            _plt.show()
        _plt.close(fig)
