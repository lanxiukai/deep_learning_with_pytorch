"""Generic figure and curve-rendering primitives."""

import csv
import os
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from dl_utils.plot import _backend as _  # one-shot backend selection

import matplotlib_inline.backend_inline as backend_inline
from IPython import get_ipython

from matplotlib import pyplot as _plt

from dl_utils.training.metrics import MetricHistory, as_list, has_any_finite


def use_svg_display():
    """Use SVG display in Jupyter (no-op outside IPython)."""
    if get_ipython() is not None:
        backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size in Matplotlib."""
    use_svg_display()
    _plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes in Matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot the data in Matplotlib."""
    if legend is None:
        legend = []
    
    set_figsize(figsize)
    axes = axes if axes else _plt.gca()
    
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
        
    x_values = [X] if has_one_axis(X) else X
    y_values = x_values if Y is None else ([Y] if has_one_axis(Y) else Y)
    if Y is None:
        x_values = [[]] * len(x_values)
    if len(x_values) != len(y_values):
        x_values = x_values * len(y_values)
    axes.cla()
    for x, y, fmt in zip(x_values, y_values, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def annotate(text, xy, xytext):
    _plt.gca().annotate(text, xy=xy, xytext=xytext,
                        arrowprops=dict(arrowstyle='->'))


class Animator:
    """Animate data curves using Matplotlib interactive mode."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = _plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        if _plt.get_backend().lower() != "agg":
            _plt.ion()
            self.fig.show()
        self._closed = False

    def add(self, x, y):
        """Add the data to the animator."""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        x_data = self.X
        if not x_data:
            x_data = [[] for _ in range(n)]
            setattr(self, 'X', x_data)
        y_data = self.Y
        if not y_data:
            y_data = [[] for _ in range(n)]
            setattr(self, 'Y', y_data)
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                x_data[i].append(a)
                y_data[i].append(b)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(x_data, y_data, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.fig.canvas.draw_idle()
        if _plt.get_backend().lower() != "agg":
            self.fig.canvas.flush_events()
            _plt.pause(0.001)


def heatmap(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
            cmap='Reds'):
    """
    Show heatmaps of matrices.
    Args:
        matrices: (number of rows for display, number of columns for display, number of queries, number of keys)
        xlabel: x-axis label
        ylabel: y-axis label
        titles: titles for each subplot
        figsize: figure size
        cmap: color map
    """
    use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = _plt.subplots(num_rows, num_cols, figsize=figsize,
                              sharex=True, sharey=True, squeeze=False)
    pcm = None
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    if pcm is not None:
        fig.colorbar(pcm, ax=axes, shrink=0.6)


def trace2d(f, results):
    """Show the trace of 2D variables during optimization"""
    import torch

    set_figsize()
    _plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    _plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    _plt.xlabel('x1')
    _plt.ylabel('x2')


def seq_len_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot a histogram of sequence length pairs."""
    set_figsize()
    _, _, patches = _plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    patch_groups: Any = patches
    for patch in patch_groups[1].patches:
        patch.set_hatch('/')
    _plt.legend(legend)


def save_curve(
    x: Sequence[float],
    curves: Mapping[str, Sequence[float]],
    path: str | PathLike[str],
    *,
    xlabel: str = "Step",
    ylabel: str = "Value",
    title: str | None = None,
    csv_path: str | PathLike[str] | None = None,
) -> None:
    """
    Plot one or more curves and optionally save the raw values to CSV.

    Args:
        x: x-axis values (e.g., steps or epochs).
        curves: mapping from curve name to y values.
        path: output image path.
        xlabel, ylabel, title: labeling for the figure.
        csv_path: optional path to save the underlying data.
    """
    if not curves:
        raise ValueError("save_curve: curves is empty.")

    x_list = list(x)
    n = len(x_list)

    # Normalize inputs to lists of floats and validate lengths
    norm_curves: dict[str, list[float]] = {}
    for name, y in curves.items():
        y_list = list(map(float, y))
        if len(y_list) != n:
            raise ValueError(
                f"save_curve: length mismatch for '{name}', expected {n} got {len(y_list)}."
            )
        norm_curves[name] = y_list

    path_str = os.fspath(path)
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    _plt.figure(figsize=(8, 5))
    for name, y in norm_curves.items():
        _plt.plot(x_list, y, label=name)
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    if title:
        _plt.title(title)
    if len(norm_curves) > 1:
        _plt.legend()
    _plt.grid(True, alpha=0.3)
    _plt.tight_layout()
    _plt.savefig(path_str, dpi=200)
    _plt.close()

    if csv_path:
        csv_path_str = os.fspath(csv_path)
        os.makedirs(os.path.dirname(csv_path_str), exist_ok=True)
        header = [xlabel] + list(norm_curves.keys())
        rows = zip(x_list, *norm_curves.values())
        with open(csv_path_str, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


def save_loss_curves(
    x: Sequence[float],
    discriminator_losses: Sequence[float],
    generator_losses: Sequence[float],
    path: str | PathLike[str],
    *,
    xlabel: str = "epoch",
) -> None:
    """Save adversarial losses on separate subplots with independent y-axes."""
    x_values = list(x)
    discriminator_values = list(map(float, discriminator_losses))
    generator_values = list(map(float, generator_losses))
    if len(discriminator_values) != len(x_values):
        raise ValueError(
            "save_loss_curves: discriminator loss length does not match x."
        )
    if len(generator_values) != len(x_values):
        raise ValueError(
            "save_loss_curves: generator loss length does not match x."
        )

    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, (disc_axis, gen_axis) = _plt.subplots(
        2,
        1,
        figsize=(6, 6),
        sharex=True,
    )
    disc_axis.plot(x_values, discriminator_values, color="tab:blue")
    disc_axis.set_ylabel("discriminator loss")
    disc_axis.grid()
    gen_axis.plot(x_values, generator_values, color="tab:orange")
    gen_axis.set_xlabel(xlabel)
    gen_axis.set_ylabel("generator loss")
    gen_axis.grid()
    fig.tight_layout()
    fig.savefig(path_str, dpi=300)
    _plt.close(fig)


def maybe_save_curve(
    x: Sequence[float],
    metrics: MetricHistory,
    series: Mapping[str, str],
    path: str | PathLike[str],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    verbose: bool = False,
) -> None:
    """
    Plot curve(s) only when the needed metric keys exist and contain valid data.
    Automatically truncates to a common length to avoid length-mismatch errors.
    """
    curves: dict[str, list[float]] = {}
    lengths: list[int] = []
    for label, key in series.items():
        if key not in metrics:
            continue
        y = as_list(metrics.get(key))
        if not y:
            continue
        if not has_any_finite(y):
            continue
        curves[label] = [float(value) for value in y]
        lengths.append(len(y))

    if not curves:
        return

    n = min([len(x)] + lengths) if lengths else len(x)
    if n <= 0:
        return

    x_use = x[:n]
    curves_use = {key: values[:n] for key, values in curves.items()}
    try:
        save_curve(x_use, curves_use, path=path, xlabel=xlabel, ylabel=ylabel, title=title)
    except Exception as err:
        if verbose:
            print(f"[genai] skip plot {path!r}: {err}")
