"""Aligned CelebA dataset access for unconditional image generation."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dl_utils.data.images import load_rgb_image


CELEBA_PARTITIONS = {"train": 0, "validation": 1, "test": 2}
CELEBA_ALIGNED_CROP_SIZE = 178


class CelebAAlignedDataset(Dataset):
    """Read one official split from the locally prepared aligned CelebA."""

    def __init__(self, root, split="train", transform=None):
        super().__init__()
        root = Path(root)
        if split not in CELEBA_PARTITIONS:
            choices = ", ".join(sorted(CELEBA_PARTITIONS))
            raise ValueError(f"split must be one of {{{choices}}}.")

        partition_path = root / "list_eval_partition.csv"
        image_candidates = (
            root / "img_align_celeba" / "img_align_celeba",
            root / "img_align_celeba",
        )
        image_dir = next(
            (candidate for candidate in image_candidates if candidate.is_dir()),
            None,
        )
        missing = [
            path
            for path in (partition_path, image_dir)
            if path is None or not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"aligned CelebA files are incomplete under {root}: {missing}"
            )

        partition = CELEBA_PARTITIONS[split]
        with partition_path.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            self.image_paths = [
                image_dir / row["image_id"]
                for row in rows
                if int(row["partition"]) == partition
            ]
        if not self.image_paths:
            raise ValueError(f"CelebA split {split!r} contains no images.")

        missing_images = [
            path for path in self.image_paths if not path.is_file()
        ]
        if missing_images:
            preview = ", ".join(str(path) for path in missing_images[:3])
            raise FileNotFoundError(
                f"CelebA split {split!r} is missing {len(missing_images)} "
                f"images; first missing paths: {preview}"
            )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_rgb_image(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def make_aligned_celeba_loader(
    root,
    resolution: int,
    batch_size: int,
    device: torch.device,
    *,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """Create a normalized, resolution-specific aligned CelebA loader."""
    if resolution < 1 or batch_size < 1:
        raise ValueError("resolution and batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    transform = transforms.Compose(
        [
            transforms.CenterCrop(CELEBA_ALIGNED_CROP_SIZE),
            transforms.Resize(
                (resolution, resolution),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = CelebAAlignedDataset(
        root,
        split=split,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


__all__ = [
    "CELEBA_ALIGNED_CROP_SIZE",
    "CELEBA_PARTITIONS",
    "CelebAAlignedDataset",
    "make_aligned_celeba_loader",
]
