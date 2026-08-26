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
    prefetch_factor: int = 2,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader with device-appropriate memory settings."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive.")
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        **loader_kwargs,
    )


__all__ = ["make_device_aware_loader"]
