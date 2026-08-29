"""
Deep Convolutional Generative Adversarial Networks

Training data:
Anime faces:      63,565 images
Samples per epoch: 63,565
Note: The dataset has one class folder; its label is ignored by the GAN.

Generator:       3.6 M params
Discriminator:   2.8 M params
Total:           6.4 M params
"""

import warnings

import torch
import torchvision
from torch import nn
from tqdm import tqdm

from dl_utils.data.images import load_rgb_image
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.gan import update_D, update_G
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import plot, save_loss_panels, set_figsize
from dl_utils.runtime.devices import try_gpu
from dl_utils.training.metrics import MetricAccumulator

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "anime_face"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "dcgan" / "anime_face"

NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_WORKERS = 8
Z_DIM = 100
GENERATOR_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 64
LEARNING_RATE_D = 1e-4
LEARNING_RATE_G = 2e-4
SAMPLES_TO_DISPLAY = 18
SAMPLE_GRID_COLUMNS = 6
SAMPLE_EVERY_EPOCHS = 2


class G_block(nn.Module):
    def __init__(
        self, out_channels, in_channels=3, kernel_size=4, strides=2, padding=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        return self.activation(self.batch_norm(self.conv2d_trans(inputs)))


class D_block(nn.Module):
    def __init__(
        self,
        out_channels,
        in_channels=3,
        kernel_size=4,
        strides=2,
        padding=1,
        alpha=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, inputs):
        return self.activation(self.batch_norm(self.conv2d(inputs)))


def train(
    discriminator,
    generator,
    data_loader,
    num_epochs,
    discriminator_lr,
    generator_lr,
    z_dim,
    output_dir,
    device=None,
):
    device = try_gpu() if device is None else device
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for parameter in discriminator.parameters():
        nn.init.normal_(parameter, 0, 0.02)
    for parameter in generator.parameters():
        nn.init.normal_(parameter, 0, 0.02)
    discriminator, generator = discriminator.to(device), generator.to(device)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=(0.5, 0.999),
    )
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_lr,
        betas=(0.5, 0.999),
    )
    epochs, discriminator_losses, generator_losses = [], [], []

    with tqdm(
        total=num_epochs * len(data_loader),
        desc=f"Epoch 1/{num_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs}", refresh=False)
            # Train one epoch
            metric = MetricAccumulator(
                ("discriminator", "generator"),
                device=device,
            )
            for real_images, _ in data_loader:
                batch_size = real_images.shape[0]
                noise = torch.normal(0, 1, size=(batch_size, z_dim, 1, 1))
                real_images = real_images.to(device)
                noise = noise.to(device)
                discriminator_loss = update_D(
                    real_images,
                    noise,
                    discriminator,
                    generator,
                    loss,
                    discriminator_optimizer,
                    real_label=0.9,
                )
                generator_loss = update_G(
                    noise,
                    discriminator,
                    generator,
                    loss,
                    generator_optimizer,
                )
                metric.update(
                    (
                        discriminator_loss / batch_size,
                        generator_loss / batch_size,
                    ),
                    num_examples=batch_size,
                )
                progress_bar.update(1)
            # Show generated examples
            noise = torch.normal(
                0, 1, size=(SAMPLES_TO_DISPLAY, z_dim, 1, 1), device=device
            )
            # Normalize the synthetic data to N(0, 1)
            generated_images = generator(noise).permute(0, 2, 3, 1) / 2 + 0.5
            image_grid = torch.cat(  # → (3×H, 7×W, C)
                [
                    torch.cat(
                        [  # → (H, 7×W, C)
                            generated_images[
                                row_index * SAMPLE_GRID_COLUMNS + column_index
                            ]
                            .cpu()
                            .detach()
                            for column_index in range(SAMPLE_GRID_COLUMNS)
                        ],
                        dim=1,
                    )
                    for row_index in range(len(generated_images) // SAMPLE_GRID_COLUMNS)
                ],
                dim=0,
            )
            if epoch == 1 or epoch % SAMPLE_EVERY_EPOCHS == 0:
                plt.imsave(
                    output_dir / f"epoch_{epoch:03d}.png",
                    image_grid.numpy(),
                )
            # Show the losses
            epoch_metrics = metric.compute()
            discriminator_loss = epoch_metrics["discriminator"]
            generator_loss = epoch_metrics["generator"]
            epochs.append(epoch)
            discriminator_losses.append(discriminator_loss)
            generator_losses.append(generator_loss)
            progress_bar.set_postfix(
                loss_D=f"{discriminator_loss:.3f}",
                loss_G=f"{generator_loss:.3f}",
            )
    print(f"loss_D {discriminator_loss:.3f}, loss_G {generator_loss:.3f} on {device}")
    save_loss_panels(
        epochs,
        {
            "Discriminator loss": {"D loss": discriminator_losses},
            "Generator loss": {"G loss": generator_losses},
        },
        output_dir / "loss_curves.png",
    )
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pth")
    torch.save(generator.state_dict(), output_dir / "generator.pth")


def main():
    reset_dir(str(OUT_DIR))

    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),  # int [0, 255] -> Pytorch Tensor FP32 [0.0, 1.0]
            torchvision.transforms.Normalize(
                0.5, 0.5
            ),  # (pixel - 0.5) / 0.5, [0, 1] -> [-1, 1]
        ]
    )
    anime_face = torchvision.datasets.ImageFolder(
        DATA_DIR, transform=transformer, loader=load_rgb_image
    )
    data_loader = torch.utils.data.DataLoader(
        anime_face, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # Close the matplotlib / torch warning output
    warnings.filterwarnings("ignore")
    for images, _ in data_loader:
        # [20, C, H, W] -> [20, H, W, C]; [-1, 1] -> [0, 1]
        sample_images = images[:20].permute(0, 2, 3, 1) / 2 + 0.5
        # set_figsize controls the figure size via rcParams;
        # plt.subplots() respects it (unlike show_images which overrides figsize)
        set_figsize((4, 5))
        num_rows, num_cols = 4, 5
        _, axes = plt.subplots(num_rows, num_cols)
        for axis, image in zip(axes.flatten(), sample_images):
            axis.imshow(image.cpu().numpy())
            axis.axis("off")
        plt.savefig(OUT_DIR / "anime_face_samples.png", dpi=300)
        plt.close()
        break

    example_inputs = torch.zeros((2, 3, 16, 16))
    generator_block = G_block(20)
    print(generator_block(example_inputs).shape)  # torch.Size([2, 20, 32, 32])

    example_inputs = torch.zeros((2, 3, 1, 1))
    generator_block = G_block(20, strides=1, padding=0)
    print(generator_block(example_inputs).shape)  # torch.Size([2, 20, 4, 4])

    generator = nn.Sequential(
        G_block(
            in_channels=Z_DIM,
            out_channels=GENERATOR_BASE_CHANNELS * 8,
            strides=1,
            padding=0,
        ),  # Output: (64 * 8, 4, 4)
        G_block(
            in_channels=GENERATOR_BASE_CHANNELS * 8,
            out_channels=GENERATOR_BASE_CHANNELS * 4,
        ),  # Output: (64 * 4, 8, 8)
        G_block(
            in_channels=GENERATOR_BASE_CHANNELS * 4,
            out_channels=GENERATOR_BASE_CHANNELS * 2,
        ),  # Output: (64 * 2, 16, 16)
        G_block(
            in_channels=GENERATOR_BASE_CHANNELS * 2,
            out_channels=GENERATOR_BASE_CHANNELS,
        ),  # Output: (64, 32, 32)
        nn.ConvTranspose2d(
            in_channels=GENERATOR_BASE_CHANNELS,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.Tanh(),
    )  # Output: (3, 64, 64)

    example_noise = torch.zeros((1, Z_DIM, 1, 1))
    print(generator(example_noise).shape)  # torch.Size([1, 3, 64, 64])

    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    activation_inputs = torch.arange(-2, 1, 0.1)
    activation_outputs = [
        nn.LeakyReLU(alpha)(activation_inputs).detach().numpy() for alpha in alphas
    ]
    plot(
        activation_inputs.detach().numpy(),
        activation_outputs,
        "x",
        "y",
        alphas,
    )
    plt.savefig(OUT_DIR / "leaky_relu_curves.png", dpi=300)
    plt.close()

    example_inputs = torch.zeros((2, 3, 16, 16))
    discriminator_block = D_block(20)
    print(discriminator_block(example_inputs).shape)  # torch.Size([2, 20, 8, 8])

    discriminator = nn.Sequential(
        D_block(DISCRIMINATOR_BASE_CHANNELS),  # Output: (64, 32, 32)
        D_block(
            in_channels=DISCRIMINATOR_BASE_CHANNELS,
            out_channels=DISCRIMINATOR_BASE_CHANNELS * 2,
        ),  # Output: (64 * 2, 16, 16)
        D_block(
            in_channels=DISCRIMINATOR_BASE_CHANNELS * 2,
            out_channels=DISCRIMINATOR_BASE_CHANNELS * 4,
        ),  # Output: (64 * 4, 8, 8)
        D_block(
            in_channels=DISCRIMINATOR_BASE_CHANNELS * 4,
            out_channels=DISCRIMINATOR_BASE_CHANNELS * 8,
        ),  # Output: (64 * 8, 4, 4)
        nn.Conv2d(
            in_channels=DISCRIMINATOR_BASE_CHANNELS * 8,
            out_channels=1,
            kernel_size=4,
            bias=False,
        ),
    )  # Output: (1, 1, 1)

    example_inputs = torch.zeros((1, 3, 64, 64))
    print(discriminator(example_inputs).shape)  # torch.Size([1, 1, 1, 1])

    train(
        discriminator,
        generator,
        data_loader,
        NUM_EPOCHS,
        LEARNING_RATE_D,
        LEARNING_RATE_G,
        Z_DIM,
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
