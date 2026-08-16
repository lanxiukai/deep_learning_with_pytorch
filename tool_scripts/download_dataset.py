'''
Download Datasets

Downloads and prepares datasets used by the project examples under DATA_DIR.
'''

import os

import pandas as pd
from dl_utils.data.dataset_preparation import (
    apply_glasses_label_corrections,
    build_image_folder_cache,
    download_kaggle_dataset,
    ensure_glasses_classification,
    prepare_celeba_cyclegan_splits,
)
from dl_utils.data.downloads import download, download_extract
from dl_utils.data.vision import vision_loaders
from dl_utils.d2l.time_machine import read_time_machine
from dl_utils.filesystem.project_root import infer_project_root


PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# 1. MNIST
#    Source: torchvision.datasets.MNIST
#    Format: 70K grayscale (1ch) images, 28×28 px, uint8, 10 digit classes (0-9)
#    Reference: https://yann.lecun.com/exdb/mnist/
# ============================================================
try:
    train_iter, test_iter = vision_loaders(
        dataset="mnist", data_dir=DATA_DIR / "mnist", batch_size=256)
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
        dataset="fashion_mnist",
        data_dir=DATA_DIR / "fashion_mnist",
        batch_size=256,
    )
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
        dataset="cifar10", data_dir=DATA_DIR / "cifar10", batch_size=256)
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
    train_data = pd.read_csv(
        download('kaggle_house_train', data_root=DATA_DIR)
    )
    test_data = pd.read_csv(
        download('kaggle_house_test', data_root=DATA_DIR)
    )
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
    lines = read_time_machine(data_root=DATA_DIR)
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
    ok = download_kaggle_dataset(
        "jessicali9530/celeba-dataset",
        DATA_DIR / "celeba",
    )
    if ok:
        prepare_celeba_cyclegan_splits(DATA_DIR / "celeba")
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
    ok = download_kaggle_dataset(
        "splcher/animefacedataset",
        DATA_DIR / "anime_face",
    )
    if ok:
        print("The Anime Face Dataset has been downloaded.")
except Exception as e:
    print(f"Failed to download the Anime Face Dataset: {e}")


# ============================================================
# 8. Glasses or No Glasses
#    Source: Kaggle (kagglehub) — jeffheaton/glasses-or-no-glasses
#    Format: 5000 RGB (3ch) face images, 1024×1024 px, uint8, binary label
#    Contains: train.csv (labels), test.csv (labels), faces-spring-2020/ (images)
#    Label corrections: a repository-tracked Vision-LLM review moves 415 images
#    from G/ to NoG/ and 102 from NoG/ to G/ after the initial CSV split.
#    Download: https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses
# ============================================================
try:
    ok = download_kaggle_dataset(
        "jeffheaton/glasses-or-no-glasses",
        DATA_DIR / "glasses",
    )
    if ok:
        print("The Glasses or No Glasses Dataset has been downloaded.")
        out_root = DATA_DIR / "glasses"
        ensure_glasses_classification(out_root)
        apply_glasses_label_corrections(out_root)
        resized_root = DATA_DIR / "glasses-256"
        resized_root.mkdir(parents=True, exist_ok=True)
        unexpected_classes = sorted(
            path.name
            for path in resized_root.iterdir()
            if path.is_dir() and path.name not in {"G", "NoG"}
        )
        if unexpected_classes:
            raise RuntimeError(
                f"Unexpected class directories in {resized_root}: "
                f"{unexpected_classes}"
            )
        workers = min(16, os.cpu_count() or 1)
        for class_name in ("G", "NoG"):
            build_image_folder_cache(
                out_root / class_name,
                resized_root / class_name,
                size=256,
                workers=workers,
            )
        # Also normalize an existing cache that may have been created before
        # the reviewed label corrections were integrated into this workflow.
        apply_glasses_label_corrections(resized_root)
        print(f"  Glasses training dataset ready: {resized_root}")
except Exception as e:
    print(f"Failed to download the Glasses or No Glasses Dataset: {e}")


# ============================================================
# 9. Airfoil Self-Noise
#    Source: D2L data hub — download('airfoil')
#    Format: tabular regression (1504 rows, 5 features, 1 target)
#    Used by: deep_learning/4.0_optimization/0.5_minibatch_sgd_0.py
#    Download: https://d2l-data.s3-accelerate.amazonaws.com/airfoil_self_noise.dat
# ============================================================
try:
    import numpy as np
    data = np.genfromtxt(
        download('airfoil', data_root=DATA_DIR),
        dtype=np.float32,
        delimiter='\t',
    )
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
#     Download: https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip
# ============================================================
try:
    data_dir = download_extract('fra-eng', data_root=DATA_DIR)
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
#     Used by: genai/1.0_generative_adversarial_network/ (DCGAN training)
#     Download: https://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip
# ============================================================
data_dir = download_extract('pokemon', data_root=DATA_DIR)
print(f"The Pokemon Dataset has been downloaded: {data_dir}")
