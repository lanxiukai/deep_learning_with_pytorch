"""Beta-TCVAE: add targeted total-correlation pressure to a standard VAE.

The previous beta-VAE lesson multiplies every part of the sample KL by beta.
Beta-TCVAE keeps the ordinary VAE rate and adds only

    distortion + rate + (beta - 1) * TC_MWS

where ``TC_MWS`` is Chen et al.'s minibatch-weighted training surrogate for
``KL(q(z) || product_j q(z_j))``.  No new network is introduced: relative to
beta-VAE, the only algorithmic increment is the aggregated-posterior loss.

MNIST makes the density calculation quick enough to inspect on one 12 GB GPU,
but its digit label is not a complete set of independent generative factors.
Samples or traversals from this script are therefore not proof of semantic
disentanglement.

Examples:
    python genai/2.0_variational_autoencoder/1.4_beta_tc_vae.py --smoke-test
    python genai/2.0_variational_autoencoder/1.4_beta_tc_vae.py --beta 6
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae_common import (
    ConvGaussianVAE28,
    diagonal_gaussian_kl_from_logvar,
    diagonal_gaussian_log_density,
)


PROJECT_ROOT = infer_project_root()


def mws_kl_decomposition(
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    dataset_size: int,
) -> dict[str, Tensor]:
    """Estimate MI, TC, and dimension-wise KL with minibatch weighting.

    ``log q(z_i[d] | x_j)`` is materialized as ``[B, B, D]``.  Consequently
    memory grows quadratically with batch size; this visible implementation is
    intended for teaching rather than large-scale optimization.

    The ``1 / (N B)`` MWS expression is an unnormalized training score.  Its
    absolute TC value includes an estimator-dependent constant and must not be
    reported as a normalized held-out TC measurement.
    """
    batch_size, latent_dim = z.shape
    if batch_size < 2:
        raise ValueError("the minibatch TC estimator needs batch_size >= 2")
    if dataset_size < batch_size:
        raise ValueError("dataset_size must be at least the batch size")
    if mu.shape != z.shape or logvar.shape != z.shape:
        raise ValueError("z, mu, and logvar must have matching [B, D] shapes")

    log_q_z_given_x = diagonal_gaussian_log_density(z, mu, logvar).sum(dim=1)
    pair_log_q = diagonal_gaussian_log_density(
        z[:, None, :], mu[None, :, :], logvar[None, :, :]
    )
    log_mws_weight = math.log(dataset_size * batch_size)
    log_q_z = torch.logsumexp(pair_log_q.sum(dim=2), dim=1) - log_mws_weight
    log_prod_q_z = (
        torch.logsumexp(pair_log_q, dim=1) - log_mws_weight
    ).sum(dim=1)
    log_p_z = diagonal_gaussian_log_density(
        z, torch.zeros_like(z), torch.zeros_like(z)
    ).sum(dim=1)
    mutual_information_mws = (log_q_z_given_x - log_q_z).mean()
    total_correlation_mws = (log_q_z - log_prod_q_z).mean()
    dimension_kl_mws = (log_prod_q_z - log_p_z).mean()
    # MWS differs from a normalized batch-mixture score only by constants.
    # Removing them makes logs readable and leaves every parameter gradient
    # exactly unchanged.  These centered values remain minibatch estimates,
    # not normalized held-out measurements of the dataset-level quantities.
    log_dataset_size = math.log(dataset_size)
    return {
        "mutual_information_centered": mutual_information_mws - log_dataset_size,
        "total_correlation_centered": total_correlation_mws
        - (latent_dim - 1) * log_dataset_size,
        "dimension_kl_centered": dimension_kl_mws
        + latent_dim * log_dataset_size,
        "total_correlation_mws_raw": total_correlation_mws,
    }


def beta_tc_vae_loss(
    reconstruction: Tensor,
    x: Tensor,
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    beta: float,
    dataset_size: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the paper-aligned ``D + R + (beta - 1) TC`` objective."""
    if beta < 1:
        raise ValueError("this lesson uses beta >= 1 so TC is not rewarded")
    distortion = F.binary_cross_entropy(reconstruction, x, reduction="none")
    distortion = distortion.flatten(1).sum(dim=1).mean()
    rate = diagonal_gaussian_kl_from_logvar(mu, logvar).sum(dim=1).mean()
    decomposition = mws_kl_decomposition(
        z, mu, logvar, dataset_size=dataset_size
    )
    tc = decomposition["total_correlation_centered"]
    loss = distortion + rate + (float(beta) - 1.0) * tc
    return loss, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "mutual_information_centered": decomposition[
            "mutual_information_centered"
        ].detach(),
        "total_correlation_centered": tc.detach(),
        "dimension_kl_centered": decomposition["dimension_kl_centered"].detach(),
        "total_correlation_mws_raw": decomposition[
            "total_correlation_mws_raw"
        ].detach(),
    }


def smoke_test() -> None:
    torch.manual_seed(7)
    model = ConvGaussianVAE28(latent_dim=6)
    x = torch.rand(8, 1, 28, 28)
    reconstruction, mu, logvar, z = model(x)
    loss, terms = beta_tc_vae_loss(
        reconstruction,
        x,
        z,
        mu,
        logvar,
        beta=6.0,
        dataset_size=60_000,
    )
    loss.backward()
    assert reconstruction.shape == x.shape
    assert model.posterior.weight.grad is not None
    assert torch.isfinite(loss)
    print(
        "smoke test passed: "
        + ", ".join(f"{name}={value.item():.3f}" for name, value in terms.items())
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "beta_tc_vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = ConvGaussianVAE28(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fixed_z = torch.randn(64, args.latent_dim, generator=generator, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = torch.zeros(7, device=device)
        examples = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, mu, logvar, z = model(x)
            loss, terms = beta_tc_vae_loss(
                reconstruction,
                x,
                z,
                mu,
                logvar,
                beta=args.beta,
                dataset_size=len(dataset),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [loss.detach(), *terms.values()]
            sums += torch.stack(values) * x.shape[0]
            examples += x.shape[0]

        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: loss={means[0]:.3f}, D={means[1]:.3f}, "
            f"R={means[2]:.3f}, MI_centered={means[3]:.3f}, "
            f"TC_centered={means[4]:.3f}, dimKL_centered={means[5]:.3f}, "
            f"TC_MWS_raw={means[6]:.3f}"
        )
        model.eval()
        with torch.inference_mode():
            samples = model.decode(fixed_z)
        save_image(samples, out_dir / f"epoch_{epoch:03d}.png", nrow=8)

    torch.save(
        {
            "format_version": 1,
            "model_name": "beta_tc_vae",
            "state_dict": model.state_dict(),
            "model_config": {"latent_dim": args.latent_dim},
            "objective": "distortion + rate + (beta - 1) * centered_TC_MWS",
            "beta": args.beta,
            "dataset_size": len(dataset),
        },
        out_dir / "beta_tc_vae.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=2e-4)
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
