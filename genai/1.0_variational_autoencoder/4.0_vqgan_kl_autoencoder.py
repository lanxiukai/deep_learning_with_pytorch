"""
Perceptual autoencoders: VQGAN and KL-regularized autoencoder.

Both variants share an encoder, decoder, fixed VGG feature loss, and a local
PatchGAN discriminator.  They differ only at the bottleneck:

* vq: nearest code + codebook/commitment losses (the VQGAN route)
* kl: Gaussian reparameterization + a light KL term (the latent-diffusion route)

The two optimizer steps make the gradient boundary explicit:

1. Autoencoder step: freeze D parameters, but keep D(fake)'s gradient to fake.
2. Discriminator step: detach fake, then update D with hinge loss.

Adversarial training starts after a warm-up.  Its weight is balanced against
the reconstruction objective using gradient norms on the decoder's last layer.

Run:
    python genai/1.0_variational_autoencoder/4.0_vqgan_kl_autoencoder.py \
        --smoke-test
    python genai/1.0_variational_autoencoder/4.0_vqgan_kl_autoencoder.py \
        --bottleneck vq
    python genai/1.0_variational_autoencoder/4.0_vqgan_kl_autoencoder.py \
        --bottleneck kl
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels // 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @property
    def last_layer(self) -> nn.Parameter:
        final_convolution = self.net[-2]
        assert isinstance(final_convolution, nn.ConvTranspose2d)
        return final_convolution.weight

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    def __init__(
        self, codebook_size: int, embedding_dim: int, commitment: float = 0.25
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment = commitment
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(
        self, z_e: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        distances = (
            flat.square().sum(1, keepdim=True)
            + self.embedding.weight.square().sum(1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(
            z_e.shape[0], z_e.shape[2], z_e.shape[3], self.embedding_dim
        )
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment * F.mse_loss(z_e, z_q.detach())
        z_st = z_e + (z_q - z_e).detach()

        counts = torch.bincount(indices, minlength=self.codebook_size).float()
        probabilities = counts / counts.sum()
        perplexity = torch.exp(
            -(probabilities * torch.log(probabilities + 1e-10)).sum()
        )
        grid = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        diagnostics = {
            "perplexity": perplexity.detach(),
            "active_codes": (counts > 0).sum().detach(),
        }
        return (
            z_st,
            grid,
            codebook_loss + commitment_loss,
            diagnostics,
        )

    def lookup(self, indices: Tensor) -> Tensor:
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()


class VQAutoencoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 64,
        codebook_size: int = 512,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(latent_channels, hidden_channels)
        self.quantizer = VectorQuantizer(codebook_size, latent_channels)
        self.decoder = Decoder(latent_channels, hidden_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        z_st, _, latent_loss, diagnostics = self.quantizer(self.encoder(x))
        return self.decoder(z_st), latent_loss, diagnostics

    @torch.inference_mode()
    def reconstruct(self, x: Tensor) -> Tensor:
        z_st, _, _, _ = self.quantizer(self.encoder(x))
        return self.decoder(z_st)


class KLAutoencoder(nn.Module):
    def __init__(
        self, latent_channels: int = 4, hidden_channels: int = 128
    ) -> None:
        super().__init__()
        self.encoder = Encoder(2 * latent_channels, hidden_channels)
        self.decoder = Decoder(latent_channels, hidden_channels)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-12.0, 12.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        # Per-sample latent KL is summed first, then averaged over the batch.
        kl = 0.5 * (
            mu.square() + logvar.exp() - 1.0 - logvar
        ).flatten(1).sum(dim=1).mean()
        diagnostics = {
            "latent_mean": mu.mean().detach(),
            "latent_std": mu.std().detach(),
        }
        return self.decoder(z), kl, diagnostics

    @torch.inference_mode()
    def reconstruct(self, x: Tensor) -> Tensor:
        mu, _ = self.encode(x)
        return self.decoder(mu)


class VGGPerceptualLoss(nn.Module):
    """L1 distance at three frozen VGG16 feature depths."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        features = models.vgg16(weights=weights).features
        self.blocks = nn.ModuleList(
            [features[:4], features[4:9], features[9:16]]
        )
        self.blocks.eval().requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction = (prediction.add(1.0).mul(0.5) - self.mean) / self.std
        target = (target.add(1.0).mul(0.5) - self.mean) / self.std
        loss = prediction.new_zeros(())
        for block in self.blocks:
            prediction = block(prediction)
            target = block(target)
            loss = loss + F.l1_loss(prediction, target)
        return loss


class PatchDiscriminator(nn.Module):
    """Spectrally normalized PatchGAN returning a grid of logits."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()

        def block(
            in_channels: int, out_channels: int, normalize: bool
        ) -> list[nn.Module]:
            layers: list[nn.Module] = [
                spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(3, base_channels, False),
            *block(base_channels, base_channels * 2, True),
            *block(base_channels * 2, base_channels * 4, True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, 1, 3, 1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def discriminator_hinge_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return 0.5 * (
        F.relu(1.0 - real_logits).mean()
        + F.relu(1.0 + fake_logits).mean()
    )


def adaptive_adversarial_weight(
    reconstruction_loss: Tensor,
    adversarial_loss: Tensor,
    last_layer: nn.Parameter,
    maximum: float = 1e4,
) -> Tensor:
    reconstruction_gradient = torch.autograd.grad(
        reconstruction_loss, last_layer, retain_graph=True
    )[0]
    adversarial_gradient = torch.autograd.grad(
        adversarial_loss, last_layer, retain_graph=True
    )[0]
    weight = reconstruction_gradient.norm() / (
        adversarial_gradient.norm() + 1e-4
    )
    return weight.clamp(0.0, maximum).detach()


def autoencoder_step(
    autoencoder: nn.Module,
    discriminator: PatchDiscriminator,
    perceptual: VGGPerceptualLoss,
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    step: int,
    discriminator_start: int,
    perceptual_weight: float,
    latent_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    reconstruction, latent_loss, diagnostics = autoencoder(x)
    pixel_loss = F.l1_loss(reconstruction, x)
    feature_loss = perceptual(reconstruction, x)
    base_loss = pixel_loss + perceptual_weight * feature_loss + latent_weight * latent_loss

    discriminator.requires_grad_(False)
    try:
        adversarial_loss = -discriminator(reconstruction).mean()
        if step >= discriminator_start:
            decoder = autoencoder.decoder
            adaptive_weight = adaptive_adversarial_weight(
                base_loss, adversarial_loss, decoder.last_layer
            )
        else:
            adaptive_weight = x.new_zeros(())
        loss = base_loss + adaptive_weight * adversarial_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    finally:
        discriminator.requires_grad_(True)

    metrics = {
        "ae_loss": loss.detach(),
        "pixel": pixel_loss.detach(),
        "perceptual": feature_loss.detach(),
        "latent": latent_loss.detach(),
        "adversarial": adversarial_loss.detach(),
        "adaptive_weight": adaptive_weight.detach(),
        **diagnostics,
    }
    return reconstruction.detach(), metrics


def discriminator_step(
    discriminator: PatchDiscriminator,
    x: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    step: int,
    discriminator_start: int,
) -> Tensor:
    if step < discriminator_start:
        return x.new_zeros(())
    real_logits = discriminator(x)
    fake_logits = discriminator(reconstruction.detach())
    loss = discriminator_hinge_loss(real_logits, fake_logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def build_autoencoder(
    bottleneck: str,
    latent_channels: int,
    codebook_size: int,
    hidden_channels: int = 128,
) -> nn.Module:
    if bottleneck == "vq":
        return VQAutoencoder(
            latent_channels, codebook_size, hidden_channels
        )
    if bottleneck == "kl":
        return KLAutoencoder(latent_channels, hidden_channels)
    raise ValueError(f"unknown bottleneck: {bottleneck}")


@torch.inference_mode()
def estimate_kl_latent_scale(
    autoencoder: KLAutoencoder,
    loader: DataLoader,
    device: torch.device,
    maximum_batches: int = 100,
) -> float:
    """Estimate one dataset-level scale so encoded mu has global std near 1."""
    value_sum = torch.zeros((), device=device, dtype=torch.float64)
    square_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch_index, (x, _) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        mu, _ = autoencoder.encode(x.to(device, non_blocking=True))
        mu = mu.double()
        value_sum += mu.sum()
        square_sum += mu.square().sum()
        count += mu.numel()
    mean = value_sum / count
    variance = square_sum / count - mean.square()
    return float(torch.rsqrt(variance.clamp_min(1e-8)).item())


def smoke_test() -> None:
    torch.manual_seed(7)
    perceptual = VGGPerceptualLoss(pretrained=False)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    for bottleneck, latent_channels, latent_weight in (
        ("vq", 8, 1.0),
        ("kl", 4, 1e-6),
    ):
        autoencoder = build_autoencoder(
            bottleneck, latent_channels, codebook_size=32, hidden_channels=32
        )
        discriminator = PatchDiscriminator(base_channels=16)
        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        reconstruction, metrics = autoencoder_step(
            autoencoder,
            discriminator,
            perceptual,
            x,
            ae_optimizer,
            step=1,
            discriminator_start=0,
            perceptual_weight=0.1,
            latent_weight=latent_weight,
        )
        d_loss = discriminator_step(
            discriminator, x, reconstruction, d_optimizer, 1, 0
        )
        assert reconstruction.shape == x.shape
        assert autoencoder.decoder.last_layer.grad is not None
        print(
            f"{bottleneck} smoke test passed: ae={metrics['ae_loss'].item():.3f}, "
            f"D={d_loss.item():.3f}, adaptive={metrics['adaptive_weight'].item():.3f}"
        )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_channels = args.latent_channels
    if latent_channels is None:
        latent_channels = 64 if args.bottleneck == "vq" else 4
    out_dir = PROJECT_ROOT / "output" / f"{args.bottleneck}_perceptual_autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        PROJECT_ROOT / "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    autoencoder = build_autoencoder(
        args.bottleneck, latent_channels, args.codebook_size
    ).to(device)
    discriminator = PatchDiscriminator().to(device)
    perceptual = VGGPerceptualLoss(
        pretrained=not args.no_pretrained_perceptual
    ).to(device)
    ae_optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=args.lr, betas=(0.5, 0.9)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9)
    )
    latent_weight = (
        args.vq_weight if args.bottleneck == "vq" else args.kl_weight
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        autoencoder.train()
        discriminator.train()
        sums = torch.zeros(7, device=device)
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, metrics = autoencoder_step(
                autoencoder,
                discriminator,
                perceptual,
                x,
                ae_optimizer,
                global_step,
                args.discriminator_start,
                args.perceptual_weight,
                latent_weight,
            )
            d_loss = discriminator_step(
                discriminator,
                x,
                reconstruction,
                d_optimizer,
                global_step,
                args.discriminator_start,
            )
            batch_size = x.shape[0]
            sums += torch.stack(
                [
                    metrics["ae_loss"],
                    metrics["pixel"],
                    metrics["perceptual"],
                    metrics["latent"],
                    metrics["adversarial"],
                    metrics["adaptive_weight"],
                    d_loss,
                ]
            ) * batch_size
            examples += batch_size
            global_step += 1

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: AE={means[0]:.4f}, pixel={means[1]:.4f}, "
            f"perceptual={means[2]:.4f}, latent={means[3]:.4f}, "
            f"adv={means[4]:.4f}, lambda={means[5]:.3f}, D={means[6]:.4f}"
        )
        autoencoder.eval()
        with torch.inference_mode():
            evaluation = autoencoder.reconstruct(x[:8])
        save_image(
            torch.cat([x[:8], evaluation]).mul(0.5).add(0.5),
            out_dir / f"epoch_{epoch:03d}.png",
            nrow=8,
        )

    latent_scale: float | None = None
    if isinstance(autoencoder, KLAutoencoder):
        autoencoder.eval()
        latent_scale = estimate_kl_latent_scale(autoencoder, loader, device)
        print(f"estimated dataset-level latent scale: {latent_scale:.6f}")

    torch.save(
        {
            "model": autoencoder.state_dict(),
            "bottleneck": args.bottleneck,
            "latent_channels": latent_channels,
            "codebook_size": args.codebook_size,
            # Downstream training multiplies encoded z by this value and
            # divides sampled z by it immediately before decoding.
            "latent_scale": latent_scale,
        },
        out_dir / "autoencoder.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--bottleneck", choices=("vq", "kl"), default="vq")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=None,
        help="default: 64 for vq, 4 for kl",
    )
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--discriminator-start", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4.5e-6)
    parser.add_argument("--discriminator-lr", type=float, default=4.5e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained-perceptual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
