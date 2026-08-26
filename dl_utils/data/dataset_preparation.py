"""Shared download, classification, correction, and resize-cache helpers."""

import csv
import filecmp
import json
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from dl_utils.data.images import flatten_to_rgb

CORRECTIONS_PATH = Path(__file__).with_name("glasses_label_corrections.json")
CELEBA_IMAGE_COUNT = 202_599
EXPECTED_TRAINING_IMAGES = 4500
CORRECTED_CLASS_COUNTS = {"G": 2543, "NoG": 1957}
RAW_IMAGE_DIRS = (
    Path("faces-spring-2020") / "faces-spring-2020",
    Path("faces-spring-2020"),
)
IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}


def celeba_dataset_is_ready(
    celeba_dir: Path,
    *,
    expected_images: int = CELEBA_IMAGE_COUNT,
) -> bool:
    """Return whether the official CelebA metadata and images are complete."""
    celeba_dir = Path(celeba_dir)
    required_files = (
        celeba_dir / "list_attr_celeba.csv",
        celeba_dir / "list_eval_partition.csv",
    )
    image_dir_candidates = (
        celeba_dir / "img_align_celeba" / "img_align_celeba",
        celeba_dir / "img_align_celeba",
    )
    image_dir = next(
        (path for path in image_dir_candidates if path.is_dir()),
        None,
    )
    if image_dir is None or not all(path.is_file() for path in required_files):
        return False
    return sum(1 for path in image_dir.glob("*.jpg") if path.is_file()) == (
        expected_images
    )


def download_kaggle_dataset(
    owner_dataset: str,
    destination: Path,
    *,
    is_ready: Callable[[Path], bool] | None = None,
) -> bool:
    """Download a Kaggle dataset unless its destination is already ready."""
    destination = Path(destination)
    populated = destination.is_dir() and any(destination.iterdir())
    if populated and (is_ready is None or is_ready(destination)):
        print(f"  Already exists: {destination}")
        return True
    if populated:
        print(f"  Existing download is incomplete; retrying: {destination}")

    try:
        import kagglehub
    except ImportError:
        print(
            "  SKIP: kagglehub not installed. "
            "Install via: uv sync --no-dev --extra celeba"
        )
        return False

    kagglehub.dataset_download(
        owner_dataset,
        output_dir=os.fspath(destination),
    )
    if not destination.is_dir() or not any(destination.iterdir()):
        raise RuntimeError(
            f"Kaggle download produced no files under {destination}"
        )
    if is_ready is not None and not is_ready(destination):
        raise RuntimeError(
            f"Kaggle download is incomplete under {destination}"
        )
    print(f"  Downloaded to: {destination}")
    return True


def prepare_celeba_cyclegan_splits(celeba_dir: Path) -> bool:
    """Create black- and blond-hair splits without changing source images."""
    celeba_dir = Path(celeba_dir)
    attributes_file = celeba_dir / "list_attr_celeba.csv"
    image_dir_candidates = (
        celeba_dir / "img_align_celeba" / "img_align_celeba",
        celeba_dir / "img_align_celeba",
    )
    image_dir = next(
        (path for path in image_dir_candidates if path.is_dir()),
        None,
    )
    if not attributes_file.is_file() or image_dir is None:
        print(
            "  SKIP: CelebA files are incomplete; "
            "cannot prepare CycleGAN splits."
        )
        return False

    black_dir = celeba_dir / "black"
    blond_dir = celeba_dir / "blond"
    black_dir.mkdir(exist_ok=True)
    blond_dir.mkdir(exist_ok=True)

    linked = 0
    with attributes_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target_dir = (
                black_dir
                if int(row["Black_Hair"]) == 1
                else blond_dir
                if int(row["Blond_Hair"]) == 1
                else None
            )
            if target_dir is None:
                continue
            source = image_dir / row["image_id"]
            target = target_dir / row["image_id"]
            if target.exists() or not source.is_file():
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            linked += 1

    black_count = sum(1 for path in black_dir.iterdir() if path.is_file())
    blond_count = sum(1 for path in blond_dir.iterdir() if path.is_file())
    print(
        f"  CycleGAN local splits ready: {black_dir} "
        f"({black_count} images), {blond_dir} ({blond_count} images); "
        f"added {linked}."
    )
    return True


def resize_image(source: Path, target: Path, size: int) -> bool:
    """Resize one image atomically, flattening transparency onto white."""
    if target.exists():
        with Image.open(target) as cached:
            if cached.size != (size, size):
                raise RuntimeError(
                    f"Existing cache image has size {cached.size}, expected "
                    f"{(size, size)}: {target}"
                )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with Image.open(source) as image:
            image_format = (
                image.format or source.suffix.removeprefix(".").upper()
            )
            resized = flatten_to_rgb(image).resize(
                (size, size),
                Image.Resampling.BILINEAR,
            )
            save_options = (
                {"compress_level": 1} if image_format == "PNG" else {}
            )
            resized.save(temporary, format=image_format, **save_options)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def build_image_folder_cache(
    source_root: Path,
    target_root: Path,
    size: int,
    workers: int,
) -> None:
    """Build or resume a resized copy of an ImageFolder directory."""
    source_root = Path(source_root)
    target_root = Path(target_root)
    source_resolved = source_root.resolve()
    target_resolved = target_root.resolve()
    if (
        source_resolved == target_resolved
        or target_resolved.is_relative_to(source_resolved)
    ):
        raise ValueError("Target directory must be outside the source directory")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")

    sources = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not sources:
        raise RuntimeError(f"No supported images found under {source_root}")

    jobs = [
        (source, target_root / source.relative_to(source_root), size)
        for source in sources
    ]
    created = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(resize_image, *job) for job in jobs]
        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            created += int(future.result())
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"\rProcessed {completed}/{len(futures)} images",
                    end="",
                    flush=True,
                )
    print()

    missing = [
        source
        for source in sources
        if not (target_root / source.relative_to(source_root)).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Cache is incomplete: {len(missing)} images missing"
        )
    print(
        f"Cache ready: {target_root} ({created} created, "
        f"{len(sources) - created} already present)"
    )


def load_corrections() -> tuple[list[int], list[int]]:
    """Load and validate the static Vision-LLM review results."""
    corrections = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    g_to_nog = corrections["G_to_NoG"]
    nog_to_g = corrections["NoG_to_G"]

    if corrections.get("total_reviewed") != 4500:
        raise ValueError("Expected corrections reviewed against 4500 images")
    if len(g_to_nog) != 415 or len(nog_to_g) != 102:
        raise ValueError("Expected 415 G→NoG and 102 NoG→G corrections")
    if len(set(g_to_nog)) != len(g_to_nog):
        raise ValueError("Duplicate image IDs in G_to_NoG")
    if len(set(nog_to_g)) != len(nog_to_g):
        raise ValueError("Duplicate image IDs in NoG_to_G")
    if set(g_to_nog) & set(nog_to_g):
        raise ValueError("The correction directions contain overlapping IDs")
    return g_to_nog, nog_to_g


def _same_image(first: Path, second: Path) -> bool:
    """Return whether two files have identical bytes or decoded pixels."""
    if filecmp.cmp(first, second, shallow=False):
        return True
    with Image.open(first) as first_image, Image.open(second) as second_image:
        return (
            first_image.size == second_image.size
            and first_image.mode == second_image.mode
            and first_image.tobytes() == second_image.tobytes()
        )


def _load_training_labels(data_dir: Path) -> dict[str, str]:
    """Return the CSV-defined destination class for every training image."""
    train_csv = data_dir / "train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training labels not found: {train_csv}")

    labels = {}
    with train_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            filename = f"face-{int(row['id'])}.png"
            label = "G" if int(row["glasses"]) == 1 else "NoG"
            if filename in labels:
                raise ValueError(f"Duplicate training image ID: {filename}")
            labels[filename] = label
    if len(labels) != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES} training labels, "
            f"found {len(labels)} in {train_csv}"
        )
    return labels


def ensure_glasses_classification(data_dir: Path) -> tuple[int, int]:
    """Create or complete G/NoG from raw images; return (copied, existing)."""
    data_dir = Path(data_dir)
    labels = _load_training_labels(data_dir)
    class_dirs = {name: data_dir / name for name in ("G", "NoG")}
    for class_dir in class_dirs.values():
        class_dir.mkdir(parents=True, exist_ok=True)

    present = {
        path.name
        for class_dir in class_dirs.values()
        for path in class_dir.glob("*.png")
    }
    unexpected = sorted(present - labels.keys())
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise ValueError(
            f"{len(unexpected)} unexpected classified images under "
            f"{data_dir}: {preview}"
        )

    missing = labels.keys() - present
    if not missing:
        print(
            f"  Glasses classification already complete: "
            f"{len(present)} images."
        )
        return 0, len(present)

    raw_dir = next(
        (
            data_dir / relative
            for relative in RAW_IMAGE_DIRS
            if (data_dir / relative).is_dir()
        ),
        None,
    )
    if raw_dir is None:
        raise FileNotFoundError(
            f"{len(missing)} classified images are missing and no raw image "
            f"directory was found under {data_dir}"
        )

    missing_sources = sorted(
        filename for filename in missing if not (raw_dir / filename).is_file()
    )
    if missing_sources:
        preview = ", ".join(missing_sources[:5])
        raise FileNotFoundError(
            f"{len(missing_sources)} raw training images are missing from "
            f"{raw_dir}: {preview}"
        )

    for filename in sorted(missing, key=lambda name: int(name[5:-4])):
        shutil.copy2(raw_dir / filename, class_dirs[labels[filename]] / filename)
    print(
        f"  Glasses classification completed: {len(missing)} copied, "
        f"{len(present)} already classified."
    )
    return len(missing), len(present)


def validate_glasses_classification(
    data_dir: Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Validate unique G/NoG membership and return the two class counts."""
    data_dir = Path(data_dir)
    files = {
        name: {path.name for path in (data_dir / name).glob("*.png")}
        for name in ("G", "NoG")
    }
    duplicates = sorted(files["G"] & files["NoG"])
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"{len(duplicates)} images occur in both G/ and NoG/ under "
            f"{data_dir}: {preview}"
        )
    total = len(files["G"] | files["NoG"])
    if total != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES} classified images under "
            f"{data_dir}, found {total}"
        )

    counts = {name: len(paths) for name, paths in files.items()}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(
            f"Expected corrected class counts {expected_counts}, found "
            f"{counts} under {data_dir}"
        )
    return counts


def apply_glasses_label_corrections(data_dir: Path) -> tuple[int, int]:
    """Move reviewed images to the correct class; return (moved, unchanged)."""
    data_dir = Path(data_dir)
    g_dir = data_dir / "G"
    nog_dir = data_dir / "NoG"
    if not g_dir.is_dir() or not nog_dir.is_dir():
        raise FileNotFoundError(
            f"Expected G/ and NoG/ class directories under {data_dir}"
        )

    g_to_nog, nog_to_g = load_corrections()
    planned = [
        (g_dir / f"face-{image_id}.png", nog_dir / f"face-{image_id}.png")
        for image_id in g_to_nog
    ]
    planned.extend(
        (nog_dir / f"face-{image_id}.png", g_dir / f"face-{image_id}.png")
        for image_id in nog_to_g
    )

    missing = [
        source.name
        for source, target in planned
        if not source.is_file() and not target.is_file()
    ]
    conflicts = [
        source.name
        for source, target in planned
        if source.is_file()
        and target.is_file()
        and not _same_image(source, target)
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} reviewed images are missing under {data_dir}: "
            f"{preview}"
        )
    if conflicts:
        preview = ", ".join(conflicts[:5])
        raise FileExistsError(
            f"{len(conflicts)} corrections have different source and target "
            f"files under {data_dir}: {preview}"
        )

    moved = 0
    unchanged = 0
    for source, target in planned:
        if not source.is_file():
            unchanged += 1
            continue
        if target.is_file():
            source.unlink()
        else:
            source.replace(target)
        moved += 1

    print(
        f"  Glasses labels corrected: {moved} moved, "
        f"{unchanged} already correct."
    )
    validate_glasses_classification(data_dir, CORRECTED_CLASS_COUNTS)
    return moved, unchanged
