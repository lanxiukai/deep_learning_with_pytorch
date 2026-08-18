"""Device-aware DataLoader construction shared by image lessons."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def make_device_aware_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with device-appropriate memory settings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


__all__ = ["make_device_aware_loader"]
