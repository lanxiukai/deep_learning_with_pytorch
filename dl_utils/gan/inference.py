"""Bounded inference helpers for GAN generators."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


def make_fixed_class_latent_grid(
    class_indices: Sequence[int],
    samples_per_class: int,
    z_dim: int,
    device: torch.device,
    *,
    base_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat the same latent columns for each requested class row."""
    if not class_indices:
        raise ValueError("class_indices must not be empty.")
    if base_noise is None:
        base_noise = torch.randn(samples_per_class, z_dim, device=device)
    elif tuple(base_noise.shape) != (samples_per_class, z_dim):
        raise ValueError(
            "base_noise must have shape "
            f"({samples_per_class}, {z_dim}), got {tuple(base_noise.shape)}."
        )
    else:
        base_noise = base_noise.to(device)

    classes = torch.tensor(class_indices, dtype=torch.long, device=device)
    labels = classes.repeat_interleave(samples_per_class)
    noise = base_noise.repeat(len(class_indices), 1)
    return noise, labels


@torch.no_grad()
def refresh_generator_statistics(
    generator: nn.Module,
    *,
    z_dim: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 64,
    num_batches: int = 32,
) -> None:
    """Refresh an EMA generator's BatchNorm and spectral-norm buffers.

    Averaged parameters can disagree with buffers copied from the live model.
    Forward-only calibration updates those buffers without changing parameters
    or consuming the training loop's random-number stream.
    """
    random_generator = torch.Generator(device=device).manual_seed(771)
    was_training = generator.training
    generator.train()
    try:
        for _ in range(num_batches):
            noise = torch.randn(
                batch_size, z_dim, device=device, generator=random_generator
            )
            labels = torch.randint(
                num_classes,
                (batch_size,),
                device=device,
                generator=random_generator,
            )
            generator(noise, labels)
    finally:
        generator.train(was_training)


@torch.inference_mode()
def generate_in_batches(
    inputs: torch.Tensor | Sequence[torch.Tensor],
    batch_size: int,
    generate: Callable[..., torch.Tensor],
    *,
    module: nn.Module | None = None,
    output_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate aligned tensor inputs in bounded batches and concatenate."""
    input_tensors = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not input_tensors:
        raise ValueError("batched inference requires at least one input tensor.")
    if any(not isinstance(value, torch.Tensor) for value in input_tensors):
        raise TypeError("all batched inference inputs must be tensors.")
    if input_tensors[0].ndim == 0 or input_tensors[0].shape[0] == 0:
        raise ValueError("batched inference requires at least one input.")
    num_inputs = input_tensors[0].shape[0]
    if any(
        value.ndim == 0 or value.shape[0] != num_inputs for value in input_tensors[1:]
    ):
        raise ValueError("batched inference inputs must have equal lengths.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    was_training = module.training if module is not None else False
    if module is not None:
        module.eval()
    try:
        outputs = []
        split_inputs = [value.split(batch_size) for value in input_tensors]
        for batches in zip(*split_inputs, strict=True):
            generated = generate(*batches)
            if not isinstance(generated, torch.Tensor):
                raise TypeError("generate must return a tensor.")
            outputs.append(generated.to(output_device))
        return torch.cat(outputs)
    finally:
        if module is not None:
            module.train(was_training)


__all__ = [
    "generate_in_batches",
    "make_fixed_class_latent_grid",
    "refresh_generator_statistics",
]
