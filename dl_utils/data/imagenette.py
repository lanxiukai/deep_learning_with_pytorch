"""Imagenette-320 download, 128-pixel caching, and split-aware data loading."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path, PurePosixPath

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from dl_utils.data.images import flatten_to_rgb
from dl_utils.data.loading import make_device_aware_loader

IMAGENETTE_IMAGE_SIZE = 128
IMAGENETTE_NUM_CLASSES = 10
IMAGENETTE_TRAIN_IMAGE_COUNT = 9_469
IMAGENETTE_VALID_IMAGE_COUNT = 3_925
IMAGENETTE_320_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_320_ARCHIVE = "imagenette2-320.tgz"
IMAGENETTE_320_ARCHIVE_ROOT = "imagenette2-320"
IMAGENETTE_320_MARKER = ".imagenette-320-complete.json"
IMAGENETTE_128_MARKER = ".imagenette-128-complete.json"
IMAGENETTE_128_EXCLUSIONS = ".imagenette-128-excluded.jsonl"
IMAGENETTE_128_FORMAT_VERSION = 2
IMAGENETTE_128_JPEG_QUALITY = 95
IMAGENETTE_128_MIN_SOURCE_EDGE = 128
IMAGENETTE_128_MAX_SOURCE_PIXELS = 50_000_000
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT = (10, 120)
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

IMAGENETTE_CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
IMAGENETTE_WNIDS = tuple(IMAGENETTE_CLASS_NAMES)


def uniform_dequantize_uint8(pixels, *, generator=None):
    """Map uint8 pixels to [-1, 1) after adding uniform pixel noise."""
    if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
        raise TypeError("pixels must be a torch.uint8 tensor.")
    noise = torch.rand(
        pixels.shape,
        dtype=torch.float32,
        device=pixels.device,
        generator=generator,
    )
    return (pixels.to(torch.float32) + noise) / 128.0 - 1.0


def center_crop_long_edge(image):
    """Crop the longer image edge to form the largest centered square."""
    return transform_functional.center_crop(image, min(image.size))


def imagenette_128_transform(
    *,
    image_size: int = IMAGENETTE_IMAGE_SIZE,
    dequantize: bool = False,
    horizontal_flip: bool = False,
    preprocessed: bool = False,
):
    """Map raw or cached Imagenette images to a square tensor in [-1, 1]."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    pixel_transform: list[Callable] = []
    if not preprocessed:
        pixel_transform.append(transforms.Lambda(center_crop_long_edge))
    if not preprocessed or image_size != IMAGENETTE_IMAGE_SIZE:
        pixel_transform.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    if horizontal_flip:
        pixel_transform.append(transforms.RandomHorizontalFlip())
    if dequantize:
        pixel_transform.extend(
            [
                transforms.PILToTensor(),
                transforms.Lambda(uniform_dequantize_uint8),
            ]
        )
    else:
        pixel_transform.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
    return transforms.Compose(pixel_transform)


def resolve_imagenette_split_dir(root: str | Path, split: str) -> Path:
    """Resolve an Imagenette root or one ImageFolder-compatible split."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    root = Path(root)
    split_dir = root / split if (root / split).is_dir() else root
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Imagenette {split} directory not found: {split_dir}")
    return split_dir


def resolve_imagenette_train_dir(root: str | Path) -> Path:
    """Resolve either an Imagenette root or its ImageFolder train directory."""
    return resolve_imagenette_split_dir(root, "train")


def _class_directories_are_valid(split_dir: Path) -> bool:
    class_names = tuple(
        sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    )
    return class_names == IMAGENETTE_WNIDS


def _count_images(split_dir: Path) -> int:
    return sum(
        1
        for class_dir in split_dir.iterdir()
        if class_dir.is_dir()
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def make_imagenette_dataset(
    root: str | Path,
    *,
    split: str = "train",
    image_size: int = IMAGENETTE_IMAGE_SIZE,
    dequantize: bool = False,
    horizontal_flip: bool = False,
    preprocessed: bool = False,
) -> datasets.ImageFolder:
    """Open one 10-class split without performing network access."""
    if preprocessed and not imagenette_128_is_ready(root):
        raise RuntimeError(
            f"Imagenette-128 cache is incomplete: {Path(root)}. "
            "Run tool_scripts/download_dataset.py --dataset imagenette."
        )
    split_dir = resolve_imagenette_split_dir(root, split)
    dataset = datasets.ImageFolder(
        split_dir,
        transform=imagenette_128_transform(
            image_size=image_size,
            dequantize=dequantize,
            horizontal_flip=horizontal_flip,
            preprocessed=preprocessed,
        ),
    )
    if tuple(dataset.classes) != IMAGENETTE_WNIDS:
        raise ValueError(
            f"Expected the {IMAGENETTE_NUM_CLASSES} Imagenette WNID directories "
            f"under {split_dir}, found {dataset.classes}."
        )
    return dataset


def make_imagenette_loader(
    root: str | Path,
    batch_size: int,
    device: torch.device,
    *,
    split: str = "train",
    image_size: int = IMAGENETTE_IMAGE_SIZE,
    dequantize: bool = False,
    horizontal_flip: bool = False,
    num_workers: int = 0,
    shuffle: bool = True,
    preprocessed: bool = False,
    drop_last: bool = True,
) -> DataLoader:
    """Create a device-aware Imagenette loader from one cached split."""
    dataset = make_imagenette_dataset(
        root,
        split=split,
        image_size=image_size,
        dequantize=dequantize,
        horizontal_flip=horizontal_flip,
        preprocessed=preprocessed,
    )
    return make_device_aware_loader(
        dataset,
        batch_size,
        device,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def make_imagenette_train_validation_loaders(
    root: str | Path,
    batch_size: int,
    device: torch.device,
    *,
    image_size: int = IMAGENETTE_IMAGE_SIZE,
    horizontal_flip: bool = False,
    num_workers: int = 0,
    preprocessed: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create the matched train/validation pair used by tokenizer lessons."""
    train_loader = make_imagenette_loader(
        root,
        batch_size,
        device,
        split="train",
        image_size=image_size,
        horizontal_flip=horizontal_flip,
        num_workers=num_workers,
        preprocessed=preprocessed,
    )
    validation_loader = make_imagenette_loader(
        root,
        batch_size,
        device,
        split="val",
        image_size=image_size,
        num_workers=num_workers,
        shuffle=False,
        preprocessed=preprocessed,
        drop_last=False,
    )
    return train_loader, validation_loader


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def imagenette_320_is_ready(root: str | Path) -> bool:
    """Return whether the official 320-pixel train and validation trees exist."""
    root = Path(root)
    train_dir = root / "train"
    valid_dir = root / "val"
    marker = _read_json(root / IMAGENETTE_320_MARKER)
    if marker is None or not train_dir.is_dir() or not valid_dir.is_dir():
        return False
    return (
        marker.get("train_image_count") == IMAGENETTE_TRAIN_IMAGE_COUNT
        and marker.get("valid_image_count") == IMAGENETTE_VALID_IMAGE_COUNT
        and _class_directories_are_valid(train_dir)
        and _class_directories_are_valid(valid_dir)
        and _count_images(train_dir) == IMAGENETTE_TRAIN_IMAGE_COUNT
        and _count_images(valid_dir) == IMAGENETTE_VALID_IMAGE_COUNT
    )


def imagenette_128_is_ready(
    root: str | Path,
    *,
    min_source_edge: int | None = None,
    max_source_pixels: int | None = None,
) -> bool:
    """Return whether a complete, compatible Imagenette-128 cache exists."""
    root = Path(root)
    train_dir = root / "train"
    valid_dir = root / "val"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        return False
    if not _class_directories_are_valid(train_dir) or not _class_directories_are_valid(
        valid_dir
    ):
        return False
    marker = _read_json(root / IMAGENETTE_128_MARKER)
    if marker is None:
        return False
    retained = marker.get("image_counts")
    excluded = marker.get("excluded_image_counts")
    source_counts = marker.get("source_image_counts")
    if not isinstance(retained, dict) or not isinstance(excluded, dict):
        return False
    expected_source_counts = {
        "train": IMAGENETTE_TRAIN_IMAGE_COUNT,
        "val": IMAGENETTE_VALID_IMAGE_COUNT,
    }
    compatible = (
        marker.get("format_version") == IMAGENETTE_128_FORMAT_VERSION
        and source_counts == expected_source_counts
        and marker.get("image_size") == IMAGENETTE_IMAGE_SIZE
        and marker.get("jpeg_quality") == IMAGENETTE_128_JPEG_QUALITY
        and all(isinstance(retained.get(split), int) for split in ("train", "val"))
        and all(isinstance(excluded.get(split), int) for split in ("train", "val"))
        and all(
            retained[split] + excluded[split] == expected_source_counts[split]
            for split in ("train", "val")
        )
        and all(retained[split] >= IMAGENETTE_NUM_CLASSES for split in ("train", "val"))
    )
    if min_source_edge is not None:
        compatible = compatible and marker.get("min_source_edge") == min_source_edge
    if max_source_pixels is not None:
        compatible = compatible and marker.get("max_source_pixels") == max_source_pixels
    return (
        compatible
        and _count_images(train_dir) == retained["train"]
        and _count_images(valid_dir) == retained["val"]
    )


def _iter_imagenette_128_jobs(
    source_split_dir: Path,
    target_split_dir: Path,
    split: str,
    min_source_edge: int,
    max_source_pixels: int,
) -> Iterator[tuple[Path, Path, str, int, int]]:
    if not _class_directories_are_valid(source_split_dir):
        raise ValueError(
            f"Expected the {IMAGENETTE_NUM_CLASSES} Imagenette WNID directories "
            f"under {source_split_dir}."
        )
    for class_name in IMAGENETTE_WNIDS:
        class_dir = source_split_dir / class_name
        target_class_dir = target_split_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(class_dir.iterdir()):
            if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
                relative_path = f"{split}/{class_name}/{source.name}"
                yield (
                    source,
                    target_class_dir / f"{source.stem}.JPEG",
                    relative_path,
                    min_source_edge,
                    max_source_pixels,
                )


def _cached_image_is_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.size == (
                IMAGENETTE_IMAGE_SIZE,
                IMAGENETTE_IMAGE_SIZE,
            ) and (image.mode == "RGB" and image.format == "JPEG")
    except OSError:
        return False


def _prepare_imagenette_128_image(
    job: tuple[Path, Path, str, int, int],
) -> tuple[str, str, int | None, int | None]:
    source, target, relative_path, min_source_edge, max_source_pixels = job
    temporary = target.with_name(f".{target.name}.part")
    try:
        try:
            image = Image.open(source)
        except Image.DecompressionBombError:
            target.unlink(missing_ok=True)
            return "too-large", relative_path, None, None
        with image:
            width, height = image.size
            if min(width, height) < min_source_edge:
                target.unlink(missing_ok=True)
                return "too-small", relative_path, width, height
            if width * height > max_source_pixels:
                target.unlink(missing_ok=True)
                return "too-large", relative_path, width, height
            if _cached_image_is_ready(target):
                return "reused", relative_path, width, height
            square = center_crop_long_edge(flatten_to_rgb(image))
            resized = square.resize(
                (IMAGENETTE_IMAGE_SIZE, IMAGENETTE_IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )
            resized.save(
                temporary,
                format="JPEG",
                quality=IMAGENETTE_128_JPEG_QUALITY,
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return "created", relative_path, width, height


def prepare_imagenette_128_cache(
    source: str | Path,
    destination: str | Path,
    *,
    workers: int,
    min_source_edge: int = IMAGENETTE_128_MIN_SOURCE_EDGE,
    max_source_pixels: int = IMAGENETTE_128_MAX_SOURCE_PIXELS,
) -> Path:
    """Build or resume filtered Imagenette-128 train and validation caches."""
    if min(workers, min_source_edge, max_source_pixels) < 1:
        raise ValueError("workers and source-size limits must be positive")
    source = Path(source)
    source_dirs = {
        split: resolve_imagenette_split_dir(source, split) for split in ("train", "val")
    }
    if source_dirs["train"].resolve() == source_dirs["val"].resolve():
        raise ValueError("Imagenette-320 source must contain train and val splits")
    destination = Path(destination)
    if imagenette_128_is_ready(
        destination,
        min_source_edge=min_source_edge,
        max_source_pixels=max_source_pixels,
    ):
        return resolve_imagenette_train_dir(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Imagenette-128 destination must be outside the source tree")
    target_dirs = {split: destination / split for split in ("train", "val")}
    for target_dir in target_dirs.values():
        target_dir.mkdir(parents=True, exist_ok=True)
        unexpected_classes = sorted(
            path.name
            for path in target_dir.iterdir()
            if path.is_dir() and path.name not in IMAGENETTE_WNIDS
        )
        if unexpected_classes:
            raise ValueError(
                f"Unexpected class directories under {target_dir}: {unexpected_classes}"
            )

    expected_source_counts = {
        "train": IMAGENETTE_TRAIN_IMAGE_COUNT,
        "val": IMAGENETTE_VALID_IMAGE_COUNT,
    }
    source_counts = {"train": 0, "val": 0}
    created_counts = {"train": 0, "val": 0}
    reused_counts = {"train": 0, "val": 0}
    excluded_counts = {"train": 0, "val": 0}
    jobs = chain.from_iterable(
        _iter_imagenette_128_jobs(
            source_dirs[split],
            target_dirs[split],
            split,
            min_source_edge,
            max_source_pixels,
        )
        for split in ("train", "val")
    )
    marker = destination / IMAGENETTE_128_MARKER
    marker.unlink(missing_ok=True)
    exclusions_path = destination / IMAGENETTE_128_EXCLUSIONS
    temporary_exclusions = exclusions_path.with_suffix(".part")
    try:
        with (
            temporary_exclusions.open("w", encoding="utf-8") as exclusions_file,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            results = executor.map(
                _prepare_imagenette_128_image,
                jobs,
                buffersize=workers * 4,
            )
            for total_scanned, result in enumerate(results, start=1):
                status, relative_path, width, height = result
                split = relative_path.partition("/")[0]
                source_counts[split] += 1
                if status == "created":
                    created_counts[split] += 1
                elif status == "reused":
                    reused_counts[split] += 1
                else:
                    excluded_counts[split] += 1
                    exclusions_file.write(
                        json.dumps(
                            {
                                "path": relative_path,
                                "width": width,
                                "height": height,
                                "reason": status,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if total_scanned % 1_000 == 0:
                    retained = sum(created_counts.values()) + sum(
                        reused_counts.values()
                    )
                    excluded = sum(excluded_counts.values())
                    print(
                        f"Scanned {total_scanned:,}/"
                        f"{sum(expected_source_counts.values()):,} source images; "
                        f"retained {retained:,}, excluded {excluded:,}",
                        flush=True,
                    )
        if source_counts != expected_source_counts:
            raise RuntimeError(
                f"Expected source image counts {expected_source_counts}, "
                f"found {source_counts}."
            )
        for split, target_dir in target_dirs.items():
            empty_classes = [
                class_name
                for class_name in IMAGENETTE_WNIDS
                if not any((target_dir / class_name).glob("*.JPEG"))
            ]
            if empty_classes:
                raise RuntimeError(
                    f"Imagenette-128 {split} filtering left empty classes: "
                    f"{empty_classes}"
                )
            retained = created_counts[split] + reused_counts[split]
            if _count_images(target_dir) != retained:
                raise RuntimeError(
                    f"Imagenette-128 {split} cache contains stale or duplicate images"
                )
        temporary_exclusions.replace(exclusions_path)
    finally:
        temporary_exclusions.unlink(missing_ok=True)

    metadata = {
        "excluded_image_counts": excluded_counts,
        "format_version": IMAGENETTE_128_FORMAT_VERSION,
        "image_counts": {
            split: created_counts[split] + reused_counts[split]
            for split in ("train", "val")
        },
        "image_size": IMAGENETTE_IMAGE_SIZE,
        "jpeg_quality": IMAGENETTE_128_JPEG_QUALITY,
        "max_source_pixels": max_source_pixels,
        "min_source_edge": min_source_edge,
        "source_image_counts": source_counts,
    }
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(marker)
    created = sum(created_counts.values())
    reused = sum(reused_counts.values())
    excluded = sum(excluded_counts.values())
    print(
        f"Imagenette-128 cache is ready: {destination} "
        f"({created:,} created, {reused:,} reused, {excluded:,} excluded)"
    )
    return target_dirs["train"]


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe Imagenette archive member: {name!r}")
    return path


def _download_imagenette_archive(destination: Path) -> Path:
    downloads_dir = destination / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive = downloads_dir / IMAGENETTE_320_ARCHIVE
    if archive.is_file():
        return archive

    temporary = archive.with_suffix(f"{archive.suffix}.part")
    offset = temporary.stat().st_size if temporary.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(
        IMAGENETTE_320_URL,
        headers=headers,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    ) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == requests.codes.partial_content
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        expected = response.headers.get("Content-Length")
        expected_bytes = None if expected is None else offset + int(expected)
        downloaded = offset
        next_report = ((downloaded // (32 * 1024 * 1024)) + 1) * (32 * 1024 * 1024)
        print(
            f"Downloading Imagenette-320 to {archive}"
            + (f" (resuming at {offset:,} bytes)" if append else ""),
            flush=True,
        )
        with temporary.open(mode) as output:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    total = f"/{expected_bytes:,}" if expected_bytes is not None else ""
                    print(f"Downloaded {downloaded:,}{total} bytes", flush=True)
                    next_report += 32 * 1024 * 1024
    if expected_bytes is not None and downloaded != expected_bytes:
        raise RuntimeError(
            f"Incomplete Imagenette download: expected {expected_bytes:,} bytes, "
            f"received {downloaded:,}."
        )
    temporary.replace(archive)
    return archive


def prepare_imagenette_320_archive(
    archive_path: str | Path,
    destination: str | Path,
) -> Path:
    """Safely expand or resume the official Imagenette-320 archive."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Imagenette archive not found: {archive_path}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if imagenette_320_is_ready(destination):
        return destination / "train"

    extracted = 0
    reused = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            archive_path_name = _safe_archive_name(member.name)
            if not archive_path_name.parts:
                continue
            if archive_path_name.parts[0] != IMAGENETTE_320_ARCHIVE_ROOT:
                raise ValueError(f"Unexpected Imagenette archive root: {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in the archive: {member.name}")
            if not member.isfile() or len(archive_path_name.parts) == 1:
                continue
            relative = Path(*archive_path_name.parts[1:])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.size:
                reused += 1
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name} from archive")
            temporary = target.with_name(f".{target.name}.{os.getpid()}.part")
            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, DOWNLOAD_CHUNK_SIZE)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted += 1
            if (extracted + reused) % 2_000 == 0:
                print(
                    f"Prepared {extracted + reused:,} Imagenette archive files",
                    flush=True,
                )

    train_dir = destination / "train"
    valid_dir = destination / "val"
    if not train_dir.is_dir() or not _class_directories_are_valid(train_dir):
        raise RuntimeError("The Imagenette-320 train class tree is incomplete.")
    if not valid_dir.is_dir() or not _class_directories_are_valid(valid_dir):
        raise RuntimeError("The Imagenette-320 validation class tree is incomplete.")
    train_count = _count_images(train_dir)
    valid_count = _count_images(valid_dir)
    if train_count != IMAGENETTE_TRAIN_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {IMAGENETTE_TRAIN_IMAGE_COUNT:,} training images, "
            f"found {train_count:,}."
        )
    if valid_count != IMAGENETTE_VALID_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {IMAGENETTE_VALID_IMAGE_COUNT:,} validation images, "
            f"found {valid_count:,}."
        )

    marker = destination / IMAGENETTE_320_MARKER
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(
        json.dumps(
            {
                "source_url": IMAGENETTE_320_URL,
                "train_image_count": train_count,
                "valid_image_count": valid_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(marker)
    print(
        f"Imagenette-320 is ready: {destination} "
        f"({extracted:,} extracted, {reused:,} reused)"
    )
    return train_dir


def download_imagenette_320(destination: str | Path) -> Path:
    """Download and safely prepare the official Imagenette-320 archive."""
    destination = Path(destination)
    if imagenette_320_is_ready(destination):
        return destination / "train"
    archive = _download_imagenette_archive(destination)
    return prepare_imagenette_320_archive(archive, destination)


__all__ = [
    "IMAGENETTE_128_EXCLUSIONS",
    "IMAGENETTE_128_FORMAT_VERSION",
    "IMAGENETTE_128_JPEG_QUALITY",
    "IMAGENETTE_128_MARKER",
    "IMAGENETTE_128_MAX_SOURCE_PIXELS",
    "IMAGENETTE_128_MIN_SOURCE_EDGE",
    "IMAGENETTE_320_ARCHIVE",
    "IMAGENETTE_320_MARKER",
    "IMAGENETTE_320_URL",
    "IMAGENETTE_CLASS_NAMES",
    "IMAGENETTE_IMAGE_SIZE",
    "IMAGENETTE_NUM_CLASSES",
    "IMAGENETTE_TRAIN_IMAGE_COUNT",
    "IMAGENETTE_VALID_IMAGE_COUNT",
    "IMAGENETTE_WNIDS",
    "center_crop_long_edge",
    "download_imagenette_320",
    "imagenette_128_is_ready",
    "imagenette_128_transform",
    "imagenette_320_is_ready",
    "make_imagenette_dataset",
    "make_imagenette_loader",
    "make_imagenette_train_validation_loaders",
    "prepare_imagenette_128_cache",
    "prepare_imagenette_320_archive",
    "resolve_imagenette_split_dir",
    "resolve_imagenette_train_dir",
    "uniform_dequantize_uint8",
]
