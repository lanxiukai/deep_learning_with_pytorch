"""Bounded inference helpers for GAN generators."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


@torch.inference_mode()
def generate_in_batches(
    inputs: torch.Tensor,
    batch_size: int,
    generate: Callable[[torch.Tensor], torch.Tensor],
    *,
    module: nn.Module | None = None,
    output_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate non-empty tensor inputs in bounded batches and concatenate."""
    if inputs.ndim == 0 or inputs.shape[0] == 0:
        raise ValueError("batched inference requires at least one input.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    was_training = module.training if module is not None else None
    if module is not None:
        module.eval()
    try:
        outputs = []
        for batch in inputs.split(batch_size):
            generated = generate(batch)
            if not isinstance(generated, torch.Tensor):
                raise TypeError("generate must return a tensor.")
            outputs.append(generated.to(output_device))
        return torch.cat(outputs)
    finally:
        if module is not None:
            module.train(was_training)


__all__ = ["generate_in_batches"]
