"""
Attention Weights Visualization
"""

import torch
from d2l_importer import d2l_save

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
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
    d2l_save.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l_save.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

# (number of rows for display, number of columns for display, number of queries, number of keys)
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
d2l_save.plt.show()
