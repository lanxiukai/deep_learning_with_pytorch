"""KL-regularized perceptual autoencoder for a latent-diffusion interface.

This is the continuous sibling of VQGAN, not "VQGAN with one flag changed".
Its posterior is a spatial diagonal Gaussian; the first-stage objective uses
the LDM-style reduction

    NLL = mean_batch sum_elements((L1 + perceptual) * exp(-s) + s)
    loss = NLL + lambda_KL * KL + adaptive_weight * generator_hinge

``s`` is represented as a Parameter for fidelity to the released loss module,
but deliberately excluded from the optimizer, matching the standard released
wiring.  The checkpoint records a single dataset-level latent scale.  This
script trains a tokenizer: unconditional image generation requires a separate
latent model (the diffusion lessons), so random-normal decodes are saved only
as a prior-interface diagnostic.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.perceptual_autoencoder import (
    KLPerceptualAutoencoder32,
    PatchDiscriminator32,
    RandomFeaturePerceptualLoss,
    VGGPerceptualLoss,
    adaptive_adversarial_weight,
    discriminator_hinge_loss,
)
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar


PROJECT_ROOT = infer_project_root()


def kl_autoencoder_step(
    model: KLPerceptualAutoencoder32,
    discriminator: PatchDiscriminator32,
    perceptual: nn.Module,
    reconstruction_logvar: nn.Parameter,
    x: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
    perceptual_weight: float,
    kl_weight: float,
    discriminator_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    reconstruction, mu, logvar, _ = model(x)
    pixel_map = (reconstruction - x).abs()
    perceptual_per_sample = perceptual(reconstruction, x, reduction="none")
    feature_loss = perceptual_per_sample.mean()
    element_loss = pixel_map + float(perceptual_weight) * perceptual_per_sample.view(
        -1, 1, 1, 1
    )
    nll = (
        element_loss * torch.exp(-reconstruction_logvar)
        + reconstruction_logvar
    ).flatten(1).sum(dim=1).mean()
    kl = diagonal_gaussian_kl_from_logvar(mu, logvar).flatten(1).sum(dim=1).mean()

    discriminator.requires_grad_(False)
    try:
        adversarial = -discriminator(reconstruction).mean()
        if step >= discriminator_start:
            adaptive = adaptive_adversarial_weight(
                nll,
                adversarial,
                model.decoder.last_layer,
                scale=discriminator_weight,
            )
        else:
            adaptive = x.new_zeros(())
        loss = nll + float(kl_weight) * kl + adaptive * adversarial
        optimizer.zero_grad(set_to_none=True)
        # ``reconstruction_logvar`` is intentionally absent from optimizer.
        reconstruction_logvar.grad = None
        loss.backward()
        optimizer.step()
    finally:
        discriminator.requires_grad_(True)
    return reconstruction.detach(), mu.detach(), {
        "autoencoder": loss.detach(),
        "nll": nll.detach(),
        "pixel_l1": pixel_map.mean().detach(),
        "perceptual_vgg": feature_loss.detach(),
        "kl": kl.detach(),
        "generator_adversarial": adversarial.detach(),
        "adaptive_weight": adaptive.detach(),
    }


def discriminator_step(
    discriminator: PatchDiscriminator32,
    x: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
) -> Tensor:
    if step < discriminator_start:
        return x.new_zeros(())
    loss = discriminator_hinge_loss(
        discriminator(x), discriminator(reconstruction.detach())
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.inference_mode()
def estimate_mean_latent_scale(
    model: KLPerceptualAutoencoder32,
    loader: DataLoader,
    device: torch.device,
    *,
    maximum_batches: int,
) -> float:
    """Estimate one scale so posterior means have dataset-global std near one."""
    total = torch.zeros((), device=device, dtype=torch.float64)
    squared_total = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch_index, (x, _) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        mu, _ = model.encode(x.to(device, non_blocking=True))
        values = mu.double()
        total += values.sum()
        squared_total += values.square().sum()
        count += values.numel()
    if count == 0:
        raise ValueError("cannot estimate latent scale from an empty loader")
    mean = total / count
    variance = squared_total / count - mean.square()
    return float(torch.rsqrt(variance.clamp_min(1e-8)).item())


def smoke_test() -> None:
    torch.manual_seed(7)
    model = KLPerceptualAutoencoder32(latent_channels=4, hidden_channels=32)
    discriminator = PatchDiscriminator32(base_channels=16)
    perceptual = RandomFeaturePerceptualLoss(channels=8)
    reconstruction_logvar = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    before = reconstruction_logvar.detach().clone()
    reconstruction, mu, metrics = kl_autoencoder_step(
        model,
        discriminator,
        perceptual,
        reconstruction_logvar,
        x,
        optimizer,
        step=1,
        discriminator_start=0,
        perceptual_weight=0.1,
        kl_weight=1e-6,
        discriminator_weight=1.0,
    )
    d_loss = discriminator_step(
        discriminator,
        x,
        reconstruction,
        d_optimizer,
        step=1,
        discriminator_start=0,
    )
    assert reconstruction.shape == x.shape
    assert mu.shape == (2, 4, 8, 8)
    scaled = model.encode_latent(x, sample=False, latent_scale=0.5)
    assert model.decode_latent(scaled, latent_scale=0.5).shape == x.shape
    assert reconstruction_logvar.grad is not None
    assert torch.equal(before, reconstruction_logvar.detach())
    assert model.decoder.last_layer.grad is not None
    print(
        f"smoke test passed: AE={metrics['autoencoder'].item():.3f}, "
        f"D={d_loss.item():.3f}, KL={metrics['kl'].item():.3f}, "
        f"adaptive={metrics['adaptive_weight'].item():.3f}"
    )


def make_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = datasets.CIFAR10(
        PROJECT_ROOT / "data" / "cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def build_perceptual(name: str, device: torch.device) -> nn.Module:
    if name == "vgg":
        return VGGPerceptualLoss().to(device)
    if name == "random":
        print("warning: random features are a gradient-path debug mode, not a perceptual metric")
        return RandomFeaturePerceptualLoss().to(device)
    raise ValueError(f"unknown perceptual network: {name}")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "kl_autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args, device)
    model = KLPerceptualAutoencoder32(
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)
    discriminator = PatchDiscriminator32(args.discriminator_channels).to(device)
    perceptual = build_perceptual(args.perceptual, device)
    reconstruction_logvar = nn.Parameter(
        torch.tensor(float(args.reconstruction_logvar), device=device)
    )
    # Deliberately only model.parameters(): released LDM wiring leaves the
    # loss module's log-variance Parameter out of the Adam parameter group.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.5, 0.9)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9)
    )
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()
        sums = torch.zeros(8, device=device)
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, mu, metrics = kl_autoencoder_step(
                model,
                discriminator,
                perceptual,
                reconstruction_logvar,
                x,
                optimizer,
                step=global_step,
                discriminator_start=args.discriminator_start,
                perceptual_weight=args.perceptual_weight,
                kl_weight=args.kl_weight,
                discriminator_weight=args.discriminator_weight,
            )
            d_loss = discriminator_step(
                discriminator,
                x,
                reconstruction,
                d_optimizer,
                step=global_step,
                discriminator_start=args.discriminator_start,
            )
            sums += torch.stack([*metrics.values(), d_loss]) * x.shape[0]
            examples += x.shape[0]
            global_step += 1
        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: AE={means[0]:.3f}, NLL={means[1]:.3f}, "
            f"L1={means[2]:.4f}, VGG={means[3]:.4f}, KL={means[4]:.3f}, "
            f"G={means[5]:.4f}, lambda={means[6]:.3f}, D={means[7]:.4f}, "
            f"s={reconstruction_logvar.item():.3f} (fixed)"
        )
        model.eval()
        with torch.inference_mode():
            deterministic = model.reconstruct(x[:8])
            prior_diagnostic = model.decoder(torch.randn_like(mu[:8]))
        save_image(
            torch.cat((x[:8], deterministic)).mul(0.5).add(0.5),
            out_dir / f"reconstruction_{epoch:03d}.png",
            nrow=8,
        )
        save_image(
            prior_diagnostic.mul(0.5).add(0.5),
            out_dir / f"prior_decode_diagnostic_{epoch:03d}.png",
            nrow=8,
        )

    model.eval()
    latent_scale = estimate_mean_latent_scale(
        model,
        loader,
        device,
        maximum_batches=args.scale_batches,
    )
    print(f"dataset-level posterior-mean latent scale={latent_scale:.6f}")
    torch.save(
        {
            "format_version": 1,
            "model_name": "kl_perceptual_autoencoder",
            "state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "model_config": {
                "latent_channels": args.latent_channels,
                "hidden_channels": args.hidden_channels,
            },
            "reconstruction_logvar": reconstruction_logvar.detach().cpu(),
            "reconstruction_reduction": "sum_elements_then_mean_batch",
            "kl_weight": args.kl_weight,
            "latent_scale": latent_scale,
            "latent_scale_estimator": "global_std_of_posterior_mean",
            "global_step": global_step,
            "optimizer_states": {
                "autoencoder": optimizer.state_dict(),
                "discriminator": d_optimizer.state_dict(),
            },
        },
        out_dir / "kl_autoencoder.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--perceptual", choices=("vgg", "random"), default="vgg")
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--reconstruction-logvar", type=float, default=0.0)
    parser.add_argument("--discriminator-weight", type=float, default=1.0)
    parser.add_argument("--discriminator-start", type=int, default=1000)
    parser.add_argument("--scale-batches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.smoke_test:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
