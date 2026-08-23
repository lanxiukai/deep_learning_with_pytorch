"""VAE-GAN: learn reconstruction similarity in discriminator feature space.

The decoder is also the GAN generator.  Three optimizer phases make the
original conceptual objective concrete:

1. encoder: KL prior cost + discriminator-feature reconstruction;
2. decoder: feature reconstruction + make reconstructions/prior samples real;
3. discriminator: classify real images against both detached fake sources.

This is a short mechanism experiment, not a VAE-GAN reproduction project.
It uses a spatial Gaussian latent so the same encoder/decoder blocks can be
reused in the later KL-AE lesson, saves only final weights, and evaluates
paired fidelity separately from prior samples.  Adversarial sharpness can
invent plausible details.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.cifar10 import (
    make_cifar10_loader,
    normalized_cifar10_transform,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.perceptual_autoencoder import (
    KLPerceptualAutoencoder32,
    PatchDiscriminator32,
)
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)


PROJECT_ROOT = infer_project_root()


def set_trainable(module: torch.nn.Module, value: bool) -> None:
    module.requires_grad_(value)


def encoder_step(
    model: KLPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    kl_weight: float,
) -> dict[str, Tensor]:
    """Update E through the frozen G and frozen discriminator feature map."""
    set_trainable(model.encoder, True)
    set_trainable(model.decoder, False)
    set_trainable(discriminator, False)
    mu, logvar = model.encode(x)
    reconstruction = model.decoder(reparameterize_logvar(mu, logvar))
    with torch.no_grad():
        real_features = discriminator.extract_features(x)
    fake_features = discriminator.extract_features(reconstruction)
    feature_loss = F.mse_loss(fake_features, real_features)
    kl = diagonal_gaussian_kl_from_logvar(mu, logvar).flatten(1).sum(dim=1).mean()
    loss = feature_loss + float(kl_weight) * kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "encoder": loss.detach(),
        "feature": feature_loss.detach(),
        "kl": kl.detach(),
    }


def decoder_step(
    model: KLPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    adversarial_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Update G from learned similarity and two generator fake sources."""
    set_trainable(model.encoder, False)
    set_trainable(model.decoder, True)
    set_trainable(discriminator, False)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        posterior_z = reparameterize_logvar(mu, logvar)
    reconstruction = model.decoder(posterior_z.detach())
    prior_sample = model.decoder(torch.randn_like(posterior_z))
    with torch.no_grad():
        real_features = discriminator.extract_features(x)
    feature_loss = F.mse_loss(
        discriminator.extract_features(reconstruction), real_features
    )
    reconstruction_logits = discriminator(reconstruction)
    prior_logits = discriminator(prior_sample)
    adversarial = 0.5 * (
        F.binary_cross_entropy_with_logits(
            reconstruction_logits, torch.ones_like(reconstruction_logits)
        )
        + F.binary_cross_entropy_with_logits(prior_logits, torch.ones_like(prior_logits))
    )
    loss = feature_loss + float(adversarial_weight) * adversarial
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return reconstruction.detach(), prior_sample.detach(), {
        "decoder": loss.detach(),
        "generator_adversarial": adversarial.detach(),
    }


def discriminator_step(
    discriminator: PatchDiscriminator32,
    x: Tensor,
    reconstruction: Tensor,
    prior_sample: Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tensor:
    """Update D; both generator outputs are explicitly detached."""
    set_trainable(discriminator, True)
    real_logits = discriminator(x)
    reconstruction_logits = discriminator(reconstruction.detach())
    prior_logits = discriminator(prior_sample.detach())
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )
    fake_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(
            reconstruction_logits, torch.zeros_like(reconstruction_logits)
        )
        + F.binary_cross_entropy_with_logits(
            prior_logits, torch.zeros_like(prior_logits)
        )
    )
    loss = 0.5 * (real_loss + fake_loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def vae_gan_iteration(
    model: KLPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    x: Tensor,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    *,
    kl_weight: float,
    adversarial_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    encoder_metrics = encoder_step(
        model, discriminator, x, encoder_optimizer, kl_weight=kl_weight
    )
    reconstruction, prior_sample, decoder_metrics = decoder_step(
        model,
        discriminator,
        x,
        decoder_optimizer,
        adversarial_weight=adversarial_weight,
    )
    d_loss = discriminator_step(
        discriminator,
        x,
        reconstruction,
        prior_sample,
        discriminator_optimizer,
    )
    set_trainable(model, True)
    return reconstruction, prior_sample, {
        **encoder_metrics,
        **decoder_metrics,
        "discriminator": d_loss,
        "pixel_l1": F.l1_loss(reconstruction, x).detach(),
    }


def smoke_test() -> None:
    torch.manual_seed(7)
    model = KLPerceptualAutoencoder32(latent_channels=4, hidden_channels=32)
    discriminator = PatchDiscriminator32(base_channels=16)
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=1e-4)
    decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    reconstruction, prior, metrics = vae_gan_iteration(
        model,
        discriminator,
        x,
        encoder_optimizer,
        decoder_optimizer,
        discriminator_optimizer,
        kl_weight=1e-4,
        adversarial_weight=1.0,
    )
    assert reconstruction.shape == prior.shape == x.shape
    assert model.encoder.net[0].weight.grad is not None
    assert model.decoder.last_layer.grad is not None
    assert discriminator.head[-1].weight.grad is not None
    print(
        "smoke test passed: "
        + ", ".join(f"{name}={value.item():.3f}" for name, value in metrics.items())
    )


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    transform = normalized_cifar10_transform(horizontal_flip=False)
    root = PROJECT_ROOT / "data" / "cifar10"
    return (
        make_cifar10_loader(
            root,
            args.batch_size,
            device,
            train=True,
            transform=transform,
            num_workers=args.workers,
        ),
        make_cifar10_loader(
            root,
            args.batch_size,
            device,
            train=False,
            transform=transform,
            num_workers=args.workers,
        ),
    )


@torch.inference_mode()
def evaluate(
    model: KLPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[dict[str, float], Tensor]:
    model.eval()
    discriminator.eval()
    totals = torch.zeros(6, device=device)
    examples = 0
    comparison = None
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        reconstruction = model.decoder(mu)
        prior = model.decoder(torch.randn_like(mu))
        real_features = discriminator.extract_features(x)
        reconstruction_features = discriminator.extract_features(
            reconstruction
        )
        values = torch.stack(
            [
                F.l1_loss(reconstruction, x),
                F.mse_loss(reconstruction_features, real_features),
                diagonal_gaussian_kl_from_logvar(
                    mu, logvar
                ).flatten(1).sum(dim=1).mean(),
                discriminator(x).mean(),
                discriminator(reconstruction).mean(),
                discriminator(prior).mean(),
            ]
        )
        totals += values * x.shape[0]
        examples += x.shape[0]
        if comparison is None:
            comparison = torch.cat(
                (x[:8], reconstruction[:8], prior[:8])
            ).cpu()
    values = (totals / examples).tolist()
    assert comparison is not None
    return {
        "pixel_l1": values[0],
        "learned_feature_mse": values[1],
        "kl": values[2],
        "real_logit": values[3],
        "reconstruction_logit": values[4],
        "prior_logit": values[5],
    }, comparison


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vae_gan"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders(args, device)
    model = KLPerceptualAutoencoder32(
        latent_channels=args.latent_channels, hidden_channels=args.hidden_channels
    ).to(device)
    discriminator = PatchDiscriminator32(args.discriminator_channels).to(device)
    encoder_optimizer = torch.optim.Adam(
        model.encoder.parameters(), lr=args.lr, betas=(0.5, 0.9)
    )
    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=args.lr, betas=(0.5, 0.9)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9)
    )

    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(7, device=device)
        examples = 0
        model.train()
        discriminator.train()
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            reconstruction, prior, metrics = vae_gan_iteration(
                model,
                discriminator,
                x,
                encoder_optimizer,
                decoder_optimizer,
                discriminator_optimizer,
                kl_weight=args.kl_weight,
                adversarial_weight=args.adversarial_weight,
            )
            sums += torch.stack(list(metrics.values())) * x.shape[0]
            examples += x.shape[0]
        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: E={means[0]:.4f}, feature={means[1]:.4f}, "
            f"KL={means[2]:.3f}, G={means[3]:.4f}, G_adv={means[4]:.4f}, "
            f"D={means[5]:.4f}, pixel_L1={means[6]:.4f}"
        )
    validation, comparison = evaluate(
        model, discriminator, validation_loader, device=device
    )
    save_image(
        comparison.mul(0.5).add(0.5),
        out_dir / "real_reconstruction_prior.png",
        nrow=8,
    )

    torch.save(
        {
            "format_version": 2,
            "model_name": "vae_gan",
            "state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "model_config": {
                "latent_channels": args.latent_channels,
                "hidden_channels": args.hidden_channels,
            },
            "kl_weight": args.kl_weight,
            "adversarial_weight": args.adversarial_weight,
            "dataset": "CIFAR-10",
            "image_size": 32,
            "value_range": [-1.0, 1.0],
            "validation_metrics": validation,
        },
        out_dir / "vae_gan.pth",
    )
    print(
        f"validation: L1={validation['pixel_l1']:.4f}, "
        f"feature={validation['learned_feature_mse']:.4f}, "
        f"logits(real,reconstruction,prior)="
        f"({validation['real_logit']:.3f},"
        f"{validation['reconstruction_logit']:.3f},"
        f"{validation['prior_logit']:.3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
