"""Aligned CelebA dataset access for unconditional image generation."""

from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import Dataset

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


__all__ = [
    "CELEBA_ALIGNED_CROP_SIZE",
    "CELEBA_PARTITIONS",
    "CelebAAlignedDataset",
]
