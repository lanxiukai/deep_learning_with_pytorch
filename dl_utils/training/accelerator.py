"""Single-GPU BF16 helpers for long-running training lessons."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BF16Precision:
    """Own the BF16 autocast and optimizer-step behavior."""

    device: torch.device

    @property
    def name(self) -> str:
        return "bf16"

    def autocast(self):
        """Return a fresh forward-pass autocast context."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

    def backward_step(self, loss, optimizer) -> None:
        """Finish one CUDA backward pass before updating its parameters."""
        loss.backward()
        # PyTorch 2.13 + CUDA 13 needs an explicit completion boundary here for
        # stable BF16 GAN updates that include higher-order regularization.
        torch.cuda.synchronize(self.device)
        optimizer.step()


def configure_device(device: torch.device) -> None:
    """Enable safe throughput-oriented CUDA backend settings."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def resolve_bf16_precision(device: torch.device) -> BF16Precision:
    """Require CUDA BF16 support and construct the training context."""
    if device.type != "cuda":
        raise ValueError("BF16 training requires a CUDA device.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("the selected CUDA device does not support BF16.")
    return BF16Precision(device)


def make_fused_adam(parameters, *, device: torch.device, **kwargs):
    """Create Adam with its fused CUDA implementation when available."""
    return torch.optim.Adam(
        parameters,
        fused=device.type == "cuda",
        **kwargs,
    )


__all__ = [
    "BF16Precision",
    "configure_device",
    "make_fused_adam",
    "resolve_bf16_precision",
]
