"""CIFAR-10 dataset and loader construction for GAN lessons."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dl_utils.data.loading import make_device_aware_loader


def normalized_cifar10_transform(*, horizontal_flip: bool):
    """Map CIFAR-10 images to [-1, 1] with optional training augmentation."""
    operations = []
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return transforms.Compose(operations)


def make_cifar10_dataset(
    root: str | Path,
    *,
    train: bool,
    transform: Callable | None,
) -> datasets.CIFAR10:
    """Open an already prepared CIFAR-10 split without network access."""
    return datasets.CIFAR10(
        root,
        train=train,
        download=False,
        transform=transform,
    )


def make_cifar10_loader(
    root: str | Path,
    batch_size: int,
    device: torch.device,
    *,
    train: bool,
    transform: Callable | None,
    num_workers: int = 0,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """Create one device-aware loader for an already prepared split."""
    dataset = make_cifar10_dataset(
        root,
        train=train,
        transform=transform,
    )
    return make_device_aware_loader(
        dataset,
        batch_size,
        device,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        drop_last=train if drop_last is None else drop_last,
    )


__all__ = [
    "make_cifar10_dataset",
    "make_cifar10_loader",
    "normalized_cifar10_transform",
]
