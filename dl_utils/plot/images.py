"""Image-grid rendering primitives."""

import math
import os
from pathlib import Path
from typing import Any

from dl_utils.plot import _backend as _  # one-shot backend selection

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as _plt


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
