"""VAE-GAN: learn reconstruction similarity in discriminator feature space.

The decoder is also the GAN generator.  Three optimizer phases make the
original conceptual objective concrete:

1. encoder: KL prior cost + discriminator-feature reconstruction;
2. decoder: feature reconstruction + make reconstructions/prior samples real;
3. discriminator: classify real images against both detached fake sources.

Training first freezes an independent feature network and checks that feature
reconstruction reaches the encoder/decoder.  It then replaces that fixed
metric with the moving discriminator features and executes the three optimizer
boundaries above.  The scaffold is not a separate model or checkpoint.  This
remains a compact mechanism experiment rather than a paper-scale reproduction;
paired fidelity and prior generation stay separate because adversarial
sharpness can invent plausible details.
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
from dl_utils.training.checkpoints import reproducibility_metadata
from dl_utils.vae.perceptual_autoencoder import (
    KLPerceptualAutoencoder32,
    PatchDiscriminator32,
    RandomFeaturePerceptualLoss,
    build_perceptual_loss,
)
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)


PROJECT_ROOT = infer_project_root()


def set_trainable(module: torch.nn.Module, value: bool) -> None:
    module.requires_grad_(value)


def fixed_feature_step(
    model: KLPerceptualAutoencoder32,
    feature_network: torch.nn.Module,
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    kl_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Validate the feature-reconstruction path before learning the metric."""
    set_trainable(model, True)
    set_trainable(feature_network, False)
    reconstruction, mu, logvar, _ = model(x)
    feature_loss = feature_network(reconstruction, x)
    kl = diagonal_gaussian_kl_from_logvar(
        mu, logvar
    ).flatten(1).sum(dim=1).mean()
    loss = feature_loss + float(kl_weight) * kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return reconstruction.detach(), {
        "loss": loss.detach(),
        "feature": feature_loss.detach(),
        "kl": kl.detach(),
        "pixel_l1": F.l1_loss(reconstruction, x).detach(),
    }


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
        + F.binary_cross_entropy_with_logits(
            prior_logits, torch.ones_like(prior_logits)
        )
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
    fixed_features = RandomFeaturePerceptualLoss(channels=8)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    fixed_before = [
        parameter.detach().clone() for parameter in fixed_features.parameters()
    ]
    scaffold_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    _, scaffold_metrics = fixed_feature_step(
        model, fixed_features, x, scaffold_optimizer, kl_weight=1e-4
    )
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(
            fixed_before, fixed_features.parameters()
        )
    )

    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=1e-4)
    decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4
    )
    decoder_before = [
        parameter.detach().clone() for parameter in model.decoder.parameters()
    ]
    discriminator_before = [
        parameter.detach().clone() for parameter in discriminator.parameters()
    ]
    encoder_metrics = encoder_step(
        model, discriminator, x, encoder_optimizer, kl_weight=1e-4
    )
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(decoder_before, model.decoder.parameters())
    )
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(
            discriminator_before, discriminator.parameters()
        )
    )

    encoder_before = [
        parameter.detach().clone() for parameter in model.encoder.parameters()
    ]
    discriminator_before = [
        parameter.detach().clone() for parameter in discriminator.parameters()
    ]
    reconstruction, prior, decoder_metrics = decoder_step(
        model,
        discriminator,
        x,
        decoder_optimizer,
        adversarial_weight=1.0,
    )
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(encoder_before, model.encoder.parameters())
    )
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(
            discriminator_before, discriminator.parameters()
        )
    )

    model_before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    d_loss = discriminator_step(
        discriminator,
        x,
        reconstruction,
        prior,
        discriminator_optimizer,
    )
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(model_before, model.parameters())
    )
    assert reconstruction.shape == prior.shape == x.shape
    assert model.encoder.net[0].weight.grad is not None
    assert model.decoder.last_layer.grad is not None
    assert discriminator.head[-1].weight.grad is not None
    metrics = {
        "scaffold": scaffold_metrics["loss"],
        "encoder": encoder_metrics["encoder"],
        "decoder": decoder_metrics["decoder"],
        "discriminator": d_loss,
    }
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


@torch.inference_mode()
def evaluate_fixed_features(
    model: KLPerceptualAutoencoder32,
    feature_network: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    feature_network.eval()
    totals = torch.zeros(3, device=device)
    examples = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        reconstruction = model.decoder(mu)
        totals += torch.stack(
            (
                F.l1_loss(reconstruction, x),
                feature_network(reconstruction, x),
                diagonal_gaussian_kl_from_logvar(
                    mu, logvar
                ).flatten(1).sum(dim=1).mean(),
            )
        ) * x.shape[0]
        examples += x.shape[0]
    values = (totals / examples).tolist()
    return {
        "pixel_l1": values[0],
        "fixed_feature_distance": values[1],
        "kl": values[2],
    }


def train(args: argparse.Namespace) -> None:
    if args.scaffold_epochs < 1:
        raise ValueError("scaffold_epochs must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vae" / "vae_gan"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders(args, device)
    model = KLPerceptualAutoencoder32(
        latent_channels=args.latent_channels, hidden_channels=args.hidden_channels
    ).to(device)
    feature_network = build_perceptual_loss(
        args.fixed_feature_network, device
    )
    scaffold_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.9)
    )
    for epoch in range(1, args.scaffold_epochs + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        examples = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            _, metrics = fixed_feature_step(
                model,
                feature_network,
                x,
                scaffold_optimizer,
                kl_weight=args.kl_weight,
            )
            sums += torch.stack(tuple(metrics.values())) * x.shape[0]
            examples += x.shape[0]
        values = (sums / examples).tolist()
        print(
            f"scaffold {epoch:03d}: loss={values[0]:.4f}, "
            f"feature={values[1]:.4f}, KL={values[2]:.3f}, "
            f"pixel_L1={values[3]:.4f}"
        )
    scaffold_validation = evaluate_fixed_features(
        model, feature_network, validation_loader, device=device
    )
    print("fixed-feature scaffold complete; switching to learned features")
    del feature_network, scaffold_optimizer

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
            "roadmap_role": "historical_anchor",
            "roadmap_step": 8,
            "direct_baselines": ["standard VAE", "non-saturating GAN"],
            "visible_increment": "learned discriminator-feature similarity",
            "state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "model_config": {
                "latent_channels": args.latent_channels,
                "hidden_channels": args.hidden_channels,
            },
            "kl_weight": args.kl_weight,
            "adversarial_weight": args.adversarial_weight,
            "internal_stages": [
                {
                    "name": "fixed_feature_scaffold",
                    "feature_network": args.fixed_feature_network,
                },
                {"name": "dynamic_discriminator_feature"},
            ],
            "dataset": "CIFAR-10",
            "image_size": 32,
            "value_range": [-1.0, 1.0],
            "validation_metrics": validation,
            "controlled_baseline_metrics": {
                "fixed_feature_scaffold": scaffold_validation,
            },
            **reproducibility_metadata(
                models={
                    "autoencoder": model,
                    "discriminator": discriminator,
                },
                seed=args.seed,
                training_budget={
                    "scaffold_epochs": args.scaffold_epochs,
                    "learned_feature_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learned_feature_updates": args.epochs
                    * len(train_loader),
                },
                data_preprocessing={
                    "spatial": "native 32x32",
                    "value_range": [-1.0, 1.0],
                    "augmentation": "none",
                },
            ),
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
    parser.add_argument("--scaffold-epochs", type=int, default=2)
    parser.add_argument(
        "--fixed-feature-network",
        choices=("vgg", "random"),
        default="vgg",
    )
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
