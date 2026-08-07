"""Neural-network parameter inspection utilities."""

from torch import nn


__all__ = ["count_parameters"]


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return a module's parameter count, excluding registered buffers.

    Set ``trainable_only=True`` to include only parameters whose gradients are
    enabled.
    """
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
