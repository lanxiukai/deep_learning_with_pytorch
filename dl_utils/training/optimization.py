"""Generic optimization primitives."""

import torch
from torch import nn


def ema_decay_for_batch(batch_size: int, half_life_kimg: float) -> float:
    """Convert an image-count half-life into one batch's EMA decay."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if half_life_kimg <= 0:
        raise ValueError("half_life_kimg must be positive.")
    return 0.5 ** (batch_size / (half_life_kimg * 1_000))


@torch.no_grad()
def update_ema(
    target: nn.Module,
    source: nn.Module,
    decay: float,
    *,
    copy_buffers: bool = True,
) -> None:
    """Move target parameters toward source and optionally copy buffers."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be within [0, 1).")
    for target_parameter, source_parameter in zip(
        target.parameters(),
        source.parameters(),
        strict=True,
    ):
        target_parameter.lerp_(source_parameter, 1 - decay)
    if copy_buffers:
        for target_buffer, source_buffer in zip(
            target.buffers(),
            source.buffers(),
            strict=True,
        ):
            target_buffer.copy_(source_buffer)


def update_ema_by_images(
    target: nn.Module,
    source: nn.Module,
    batch_size: int,
    half_life_kimg: float,
    *,
    copy_buffers: bool = True,
) -> None:
    """Update an EMA using a half-life expressed in thousands of images."""
    update_ema(
        target,
        source,
        ema_decay_for_batch(batch_size, half_life_kimg),
        copy_buffers=copy_buffers,
    )


def sgd(params, lr, batch_size):
    """
    Batch stochastic gradient descent.
    
    Args:
        params: the parameters
        lr: the learning rate
    """
    with torch.no_grad():  # no need to track gradient, just update the parameters
        for param in params:  # traverse the parameters
            param -= lr * param.grad / batch_size  # update the parameter
            param.grad.zero_()                     # reset the gradient


def grad_clipping(net, theta):
    """
    Clip gradients (global norm clipping).
    
    Args:
        net: the network
        theta: the threshold of the gradient
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.sqrt(sum(
        (torch.sum(grad ** 2) for grad in grads),
        torch.zeros_like(grads[0].sum()),
    ))
    if norm > theta:
        for grad in grads:
            grad[:] *= theta / norm
