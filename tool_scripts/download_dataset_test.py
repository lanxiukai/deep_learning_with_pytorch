'''
Download Dataset Test

Downloads datasets referenced in book_repos/DGAI ch1-7 (and extras) into data/.
'''

import os
import shutil
from pathlib import Path

import pandas as pd
from dl_utils.data.downloads import download, download_extract
from dl_utils.data.vision import vision_loaders
from dl_utils.d2l.time_machine import read_time_machine


# ------------------------------------------------------------
# Helper for kagglehub downloads
# ------------------------------------------------------------
def _download_kaggle(owner_dataset: str, dest_subdir: str):
    """Download a Kaggle dataset via kagglehub, move to data/<dest_subdir>."""
    dest = os.path.join(DATA_DIR, dest_subdir)
    if os.path.exists(dest) and os.listdir(dest):
        print(f"  Already exists: {dest}")
        return True

    try:
        import kagglehub
    except ImportError:
        print(f"  SKIP: kagglehub not installed. Install via: pip install kagglehub")
        return False

    cache_path = kagglehub.dataset_download(owner_dataset)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(cache_path, dest)
    print(f"  Downloaded to: {dest}")
    return True


DATA_DIR = "./data"


def _prepare_celeba_cyclegan_splits(celeba_dir: Path):
    """Create local black- and blond-hair image folders for the CycleGAN example.

    The source images remain untouched.  Hard links avoid duplicating the
    downloaded images; filesystems without hard-link support fall back to
    copying the individual image.
    """
    attributes_file = celeba_dir / "list_attr_celeba.csv"
    image_dir_candidates = (
        celeba_dir / "img_align_celeba" / "img_align_celeba",
        celeba_dir / "img_align_celeba",
    )
    image_dir = next((path for path in image_dir_candidates if path.is_dir()), None)
    if not attributes_file.is_file() or image_dir is None:
        print("  SKIP: CelebA files are incomplete; cannot prepare CycleGAN splits.")
        return False

    black_dir = celeba_dir / "black"
    blond_dir = celeba_dir / "blond"
    black_dir.mkdir(exist_ok=True)
    blond_dir.mkdir(exist_ok=True)
    attributes = pd.read_csv(attributes_file)
    linked = 0
    for row in attributes.itertuples(index=False):
        image_id = row.image_id
        target_dir = black_dir if row.Black_Hair == 1 else blond_dir if row.Blond_Hair == 1 else None
        if target_dir is None:
            continue
        source = image_dir / image_id
        target = target_dir / image_id
        if target.exists() or not source.is_file():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        linked += 1

    print(
        f"  CycleGAN local splits ready: {black_dir} ({len(list(black_dir.iterdir()))} images), "
        f"{blond_dir} ({len(list(blond_dir.iterdir()))} images); added {linked}."
    )
    return True


# ============================================================
# 1. MNIST
#    Source: torchvision.datasets.MNIST
#    Format: 70K grayscale (1ch) images, 28×28 px, uint8, 10 digit classes (0-9)
#    Download: http://yann.lecun.com/exdb/mnist/
# ============================================================
try:
    train_iter, test_iter = vision_loaders(
        dataset="mnist", data_dir="data/mnist", batch_size=256)
    print("The MNIST Dataset has been downloaded.")
except Exception as e:
    print(f"Failed to download the MNIST Dataset: {e}")


# ============================================================
# 2. Fashion-MNIST
#    Source: torchvision.datasets.FashionMNIST
#    Format: 70K grayscale (1ch) images, 28×28 px, uint8, 10 clothing categories
#    Download: https://github.com/zalandoresearch/fashion-mnist
# ============================================================
try:
    train_iter, test_iter = vision_loaders(
        dataset="fashion_mnist", data_dir="data/fashion_mnist", batch_size=256)
    print("The Fashion-MNIST Dataset has been downloaded.")
except Exception as e:
    print(f"Failed to download the Fashion-MNIST Dataset: {e}")


# ============================================================
# 3. CIFAR-10
#    Source: torchvision.datasets.CIFAR10
#    Format: 60K RGB (3ch) images, 32×32 px, uint8, 10 object categories
#    Download: https://www.cs.toronto.edu/~kriz/cifar.html
# ============================================================
try:
    train_iter, test_iter = vision_loaders(
        dataset="cifar10", data_dir="data/cifar10", batch_size=256)
    print("The CIFAR-10 Dataset has been downloaded.")
except Exception as e:
    print(f"Failed to download the CIFAR-10 Dataset: {e}")


# ============================================================
# 4. Kaggle House Price Prediction
#    Source: d2l.download (D2L data hub)
#    Format: tabular CSV (1460 train / 1459 test rows, 79 features)
#    Contains: kaggle_house_pred_train.csv, kaggle_house_pred_test.csv
#    Download: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# ============================================================
try:
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    print("The Kaggle House Price Prediction Dataset has been downloaded.")
except Exception as e:
    print(f"Failed to download the Kaggle House Price Prediction Dataset: {e}")


# ============================================================
# 5. Time Machine
#    Source: d2l.read_time_machine (D2L data hub)
#    Format: plain text (~30K words, character-level language model)
#    Contains: timemachine.txt (raw text by H. G. Wells, public domain)
#    Download: bundled in D2L data hub
# ============================================================
try:
    lines = read_time_machine()
    print(f"The Time Machine Dataset has been downloaded: {lines[0]}")
except Exception as e:
    print(f"Failed to download the Time Machine Dataset: {e}")


# ============================================================
# 6. CelebA (CycleGAN celebrity-face dataset)
#    Source: Kaggle (kagglehub) — jessicali9530/celeba-dataset
#    Format: 202,599 aligned RGB (3ch) face images, 178×218 px, uint8, 40 binary attributes
#    Contains: img_align_celeba/, list_attr_celeba.csv, list_bbox_celeba.csv,
#              list_landmarks_align_celeba.csv, list_eval_partition.csv
#    CycleGAN splits: black/ (48,472 images), blond/ (29,980 images)
#    Note: torchvision Google Drive link is unreliable → use kagglehub
#    Download: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
# ============================================================
try:
    ok = _download_kaggle("jessicali9530/celeba-dataset", "celeba")
    if ok:
        _prepare_celeba_cyclegan_splits(Path(DATA_DIR) / "celeba")
        print("The CelebA Dataset for CycleGAN has been downloaded.")
except Exception as e:
    print(f"Failed to download the CelebA Dataset: {e}")


# ============================================================
# 7. Anime Face Dataset
#    Source: Kaggle (kagglehub) — splcher/animefacedataset
#    Format: 63.6K RGB (3ch) square anime face images, 25–220 px, uint8
#    Download: https://www.kaggle.com/datasets/splcher/animefacedataset
# ============================================================
try:
    ok = _download_kaggle("splcher/animefacedataset", "anime_face")
    if ok:
        print("The Anime Face Dataset has been downloaded.")
except Exception as e:
    print(f"Failed to download the Anime Face Dataset: {e}")


# ============================================================
# 8. Glasses or No Glasses
#    Source: Kaggle (kagglehub) — jeffheaton/glasses-or-no-glasses
#    Format: 5000 RGB (3ch) face images, 1024×1024 px, uint8, binary label
#    Contains: train.csv (labels), test.csv (labels), faces-spring-2020/ (images)
#    Note: ~11.5% (517/4500) of labels are wrong (415 in G/ without glasses,
#    102 in NoG/ with glasses). Before training, manually or via a vision model
#    identify mislabeled images in G/ NoG/ and move them to the correct folder.
#    See batch_verify_glasses.py for automated verification via OpenRouter (qwen3.7-plus).
#    Download: https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses
# ============================================================
try:
    ok = _download_kaggle("jeffheaton/glasses-or-no-glasses", "glasses")
    if ok:
        print("The Glasses or No Glasses Dataset has been downloaded.")
        img_dir   = Path(DATA_DIR) / "glasses" / "faces-spring-2020" / "faces-spring-2020"
        train_csv = Path(DATA_DIR) / "glasses" / "train.csv"
        out_root  = Path(DATA_DIR) / "glasses"
        if img_dir.exists():
            train = pd.read_csv(train_csv).set_index("id")
            for dirname in ("G", "NoG"):
                (out_root / dirname).mkdir(parents=True, exist_ok=True)
            for img_id in train.index:
                src = img_dir / f"face-{img_id}.png"
                dst = (out_root / "G" if train.loc[img_id, "glasses"] == 1 else out_root / "NoG") / f"face-{img_id}.png"
                shutil.copy2(src, dst)
            print(f"  Reorganized {len(train)} images → {out_root} (G/ NoG/)")
        else:
            print(f"  Already reorganized: {out_root}")
except Exception as e:
    print(f"Failed to download the Glasses or No Glasses Dataset: {e}")


# ============================================================
# 9. Airfoil Self-Noise
#    Source: D2L data hub — download('airfoil')
#    Format: tabular regression (1504 rows, 5 features, 1 target)
#    Used by: deep_learning/4.0_optimization/0.5_minibatch_sgd_0.py
#    Download: http://d2l-data.s3-accelerate.amazonaws.com/airfoil_self_noise.dat
# ============================================================
try:
    import numpy as np
    data = np.genfromtxt(download('airfoil'), dtype=np.float32, delimiter='\t')
    print(f"The Airfoil Self-Noise Dataset has been downloaded: {data.shape}")
except Exception as e:
    print(f"Failed to download the Airfoil Self-Noise Dataset: {e}")


# ============================================================
# 10. English-French Translation (fra-eng)
#     Source: D2L data hub — download_extract('fra-eng')
#     Format: ZIP containing fra.txt (tab-separated EN-FR sentence pairs)
#     Used by: deep_learning/2.0_recurrent_neural_network/5.0_seq2seq.py
#              deep_learning/3.0_attention_mechanisms/0.4_bahdanau_attention.py
#              deep_learning/3.0_attention_mechanisms/0.7_transformer.py
#     Download: http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip
# ============================================================
try:
    data_dir = download_extract('fra-eng')
    print(f"The English-French Translation Dataset has been downloaded: {data_dir}")
except Exception as e:
    print(f"Failed to download the English-French Translation Dataset: {e}")


# ============================================================
# 11. Pokemon
#     Source: D2L data hub — download_extract('pokemon')
#    Format: ~81k paletted PNG images, variable sizes (~50–120 px), 722 classes
#            (ImageFolder layout: one subdirectory per Pokédex number)
#            ToTensor() auto-converts palette → 3ch RGB; reshape to 64×64 for training
#     Cache: extracts to data/pokemon/ with class subdirectories (e.g., bulbasaur/, pikachu/)
#     Used by: genai/2.0_generative_adversarial_network/ (DCGAN training)
#     Download: http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip
# ============================================================
data_dir = download_extract('pokemon')
print(f"The Pokemon Dataset has been downloaded: {data_dir}")
