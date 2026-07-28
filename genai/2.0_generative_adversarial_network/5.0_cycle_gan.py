"""
CycleGAN — unpaired image-to-image translation
(black <-> blond hair; with <-> without glasses).

Adapted from the companion code of "Generative Deep Learning", 2nd ed.
(book_repos/DGAI/ch06CycleGAN.ipynb); notebook code cells kept in
original order, markdown narrative removed.
Models and training helpers live in dl_utils/genai/cyclegan.py.
Data: local data/celeba/{black,blond} prepared by
tool_scripts/download_dataset_test.py; data/glasses.
Outputs: output/cyclegan/.

Training data — hair-color task:
Black hair:              48,472 images
Blond hair:              29,980 images
Available total:         78,452 unique images
Samples per epoch:       48,472 unpaired image pairs
Note: LoadData uses the larger domain as its length, so the smaller blond-hair
domain cycles and 18,492 of its images are reused once per epoch.

Training data — glasses task:
With glasses (G):         2,543 images
Without glasses (NoG):    1,957 images
Available total:          4,500 unique images
Samples per epoch:        2,543 unpaired image pairs
Note: Counts include 517 repository-tracked label corrections. The smaller NoG
domain cycles, so 586 of its images are reused once per epoch.

Generator (each):      11.4 M params
Discriminator (each):   2.8 M params
Total (2 of each):     28.3 M params
"""

import random

import albumentations
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.genai.cyclegan import (
    Discriminator,
    Generator,
    LoadData,
    train_epoch,
    weights_init,
)


# installation
# !pip install pandas albumentations


# Data lives under <project_root>/data; generated images and checkpoints
# are written under <project_root>/output/cyclegan
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
OUT_DIR = PROJECT_ROOT / 'output' / 'cyclegan'
OUT_DIR.mkdir(parents=True, exist_ok=True)


CELEBA_DIR = DATA_DIR / "celeba"
BLACK_DIR = CELEBA_DIR / "black"
BLOND_DIR = CELEBA_DIR / "blond"
if not BLACK_DIR.is_dir() or not BLOND_DIR.is_dir():
    raise FileNotFoundError(
        "CycleGAN requires the local CelebA splits at data/celeba/black and "
        "data/celeba/blond. Run: python tool_scripts/download_dataset_test.py"
    )

def save_dataset_samples(image_dirs, output_path, samples_per_dir=8, seed=42):
    """Save a reproducible sample grid from each image directory."""
    rng = random.Random(seed)
    sampled_paths = []
    for image_dir in image_dirs:
        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
        if len(image_paths) < samples_per_dir:
            raise ValueError(
                f"{image_dir} contains {len(image_paths)} images; "
                f"at least {samples_per_dir} are required."
            )
        sampled_paths.append(rng.sample(image_paths, samples_per_dir))

    fig, axes = plt.subplots(
        len(image_dirs),
        samples_per_dir,
        dpi=100,
        figsize=(1.78 * samples_per_dir, 2.18 * len(image_dirs)),
        squeeze=False,
    )
    for row, paths in enumerate(sampled_paths):
        for column, image_path in enumerate(paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.subplots_adjust(wspace=-0.01, hspace=-0.1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


save_dataset_samples(
    [BLACK_DIR, BLOND_DIR],
    OUT_DIR / "celeba_hair_samples.png",
)


transforms = albumentations.Compose(
    [albumentations.Resize(width=256, height=256),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()],  # -> PyTorch Tensor: (C, H, W)
    additional_targets={"image0": "image"})  # Apply the same transforms to image0.
dataset = LoadData(root_A=[str(BLACK_DIR)],
                   root_B=[str(BLOND_DIR)],
                   transform=transforms)
loader = DataLoader(dataset, batch_size=1,
                    shuffle=True, pin_memory=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)
weights_init(disc_A)
weights_init(disc_B)


gen_A = Generator(img_channels=3, num_residuals=9).to(device)
gen_B = Generator(img_channels=3, num_residuals=9).to(device)
weights_init(gen_A)
weights_init(gen_B)


l1 = nn.L1Loss()
mse = nn.MSELoss()
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()


lr = 0.00001
opt_disc = torch.optim.Adam(list(disc_A.parameters()) + 
  list(disc_B.parameters()),lr=lr,betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(list(gen_A.parameters()) + 
  list(gen_B.parameters()),lr=lr,betas=(0.5, 0.999))


def test(i,A,B,fake_A,fake_B):
    save_image(A*0.5+0.5, OUT_DIR / f"A{i}.png")
    save_image(B*0.5+0.5, OUT_DIR / f"B{i}.png")    #A
    save_image(fake_A*0.5+0.5, OUT_DIR / f"fakeA{i}.png")
    save_image(fake_B*0.5+0.5, OUT_DIR / f"fakeB{i}.png")    #B


for epoch in range(1):
    train_epoch(disc_A, disc_B, gen_A, gen_B, loader, opt_disc,
    opt_gen, l1, mse, d_scaler, g_scaler, device, OUT_DIR)    #A
torch.save(gen_A.state_dict(), OUT_DIR / "gen_black.pth")
torch.save(gen_B.state_dict(), OUT_DIR / "gen_blond.pth")    #B


# answer to exercise 6.1
dataset = LoadData(root_A=[str(BLOND_DIR)],
    root_B=[str(BLACK_DIR)],
    transform=transforms)


gen_A.load_state_dict(torch.load(OUT_DIR / "gen_black.pth"))
gen_B.load_state_dict(torch.load(OUT_DIR / "gen_blond.pth"))
i=1
for black,blond in loader:
    fake_blond=gen_B(black.to(device))
    save_image(black*0.5+0.5, OUT_DIR / f"black{i}.png")
    save_image(fake_blond*0.5+0.5, OUT_DIR / f"fakeblond{i}.png")   
    fake2black=gen_A(fake_blond)
    save_image(fake2black*0.5+0.5, OUT_DIR / f"fake2black{i}.png")    
    fake_black=gen_A(blond.to(device))
    save_image(blond*0.5+0.5, OUT_DIR / f"blond{i}.png")
    save_image(fake_black*0.5+0.5, OUT_DIR / f"fakeblack{i}.png")
    fake2blond=gen_B(fake_black)
    save_image(fake2blond*0.5+0.5, OUT_DIR / f"fake2blond{i}.png")  
    i=i+1
    if i>10:
        break


# solution to exercise 6.2

disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)
gen_A = Generator(img_channels=3, num_residuals=9).to(device)
gen_B = Generator(img_channels=3, num_residuals=9).to(device)
weights_init(gen_A)
weights_init(disc_A)
weights_init(gen_B)
weights_init(disc_B)
opt_disc = torch.optim.Adam(list(disc_A.parameters()) + 
  list(disc_B.parameters()),lr=lr,betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(list(gen_A.parameters()) + 
  list(gen_B.parameters()),lr=lr,betas=(0.5, 0.999))

# The download workflow applies the reviewed G/NoG label corrections.
dataset = LoadData(root_A=[f"{DATA_DIR}/glasses/G/"],
    root_B=[f"{DATA_DIR}/glasses/NoG/"],
    transform=transforms)
loader=DataLoader(dataset,batch_size=1,
    shuffle=True, pin_memory=True)
for epoch in range(1):
    train_epoch(disc_A, disc_B, gen_A, gen_B, loader, opt_disc,
    opt_gen, l1, mse, d_scaler, g_scaler, device, OUT_DIR)
torch.save(gen_A.state_dict(), OUT_DIR / "add_glasses.pth")
torch.save(gen_B.state_dict(), OUT_DIR / "remove_glasses.pth")
