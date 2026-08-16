"""Ladder VAE: fuse top-down prior prediction with bottom-up evidence.

Relative to ``2.0_hierarchical_vae.py``, the generative model and multi-layer
ELBO are unchanged.  The single algorithmic increment is the lower posterior:

    precision_q = precision_prior + precision_evidence
    mean_q = variance_q * (
        precision_prior * mean_prior + precision_evidence * mean_evidence
    )

The top posterior is a boundary case: it comes from bottom-up evidence alone
and is compared with ``N(0, I)`` in the KL.  It must not be multiplied by that
prior as if an additional evidence source existed.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae_common import (
    fuse_diagonal_gaussians,
    reparameterize_logvar,
    split_gaussian_parameters,
)
from dl_utils.vae.vae_hierarchy import (
    ActiveUnitAccumulator,
    CompactHierarchicalVAE,
    active_units,
    hierarchical_vae_loss,
)


PROJECT_ROOT = infer_project_root()


class LadderVAE(CompactHierarchicalVAE):
    """Two-level Ladder posterior on the shared hierarchical generator."""

    def __init__(self, z1_dim: int = 32, z2_dim: int = 16) -> None:
        super().__init__(z1_dim, z2_dim)
        # Replace the baseline q(z1|z2,x) network by Gaussian evidence that is
        # deliberately independent of z2 before precision fusion.
        del self.lower_posterior
        self.lower_evidence = nn.Linear(512, 2 * z1_dim)

    def infer(self, x: Tensor, *, sample: bool = True) -> dict[str, Tensor]:
        h1 = self.bottom_up_1(x)
        h2 = self.bottom_up_2(h1)
        evidence1_mu, evidence1_logvar = split_gaussian_parameters(
            self.lower_evidence(h1)
        )
        # Original Ladder top boundary: evidence only, no prior fusion.
        q2_mu, q2_logvar = split_gaussian_parameters(self.top_posterior(h2))
        z2 = reparameterize_logvar(q2_mu, q2_logvar) if sample else q2_mu
        p1_mu, p1_logvar = split_gaussian_parameters(self.lower_prior(z2))
        q1_mu, q1_logvar = fuse_diagonal_gaussians(
            p1_mu, p1_logvar, evidence1_mu, evidence1_logvar
        )
        z1 = reparameterize_logvar(q1_mu, q1_logvar) if sample else q1_mu
        return {
            "z1": z1,
            "z2": z2,
            "q1_mu": q1_mu,
            "q1_logvar": q1_logvar,
            "p1_mu": p1_mu,
            "p1_logvar": p1_logvar,
            "q2_mu": q2_mu,
            "q2_logvar": q2_logvar,
            "p2_mu": torch.zeros_like(q2_mu),
            "p2_logvar": torch.zeros_like(q2_logvar),
            "evidence1_mu": evidence1_mu,
            "evidence1_logvar": evidence1_logvar,
            "bottom_up_h1": h1,
        }


def smoke_test() -> None:
    torch.manual_seed(7)
    model = LadderVAE(z1_dim=12, z2_dim=6)
    x = torch.rand(8, 1, 28, 28)
    reconstruction, latents = model(x)
    loss, terms = hierarchical_vae_loss(
        reconstruction, x, latents, kl_weight=0.5, free_bits=0.2
    )
    loss.backward()
    lower_active, top_active = active_units(latents)
    posterior_variance = latents["q1_logvar"].exp()
    assert reconstruction.shape == x.shape
    assert torch.all(posterior_variance <= latents["p1_logvar"].exp() + 1e-6)
    assert torch.all(
        posterior_variance <= latents["evidence1_logvar"].exp() + 1e-6
    )
    assert model.lower_prior[-1].weight.grad is not None
    print(
        "smoke test passed: "
        + ", ".join(f"{key}={value.item():.3f}" for key, value in terms.items())
        + f", active(z1,z2)=({lower_active.item()},{top_active.item()})"
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "ladder_vae"
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
    model = LadderVAE(args.z1_dim, args.z2_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        kl_weight = min(1.0, epoch / max(args.warmup_epochs, 1))
        sums = torch.zeros(4, device=device)
        examples = 0
        active_statistics = ActiveUnitAccumulator()
        model.train()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            reconstruction, latents = model(x)
            loss, terms = hierarchical_vae_loss(
                reconstruction,
                x,
                latents,
                kl_weight=kl_weight,
                free_bits=args.free_bits,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            active_statistics.update(latents)
            values = [
                loss.detach(),
                terms["distortion"],
                terms["kl_z1"],
                terms["kl_z2"],
            ]
            sums += torch.stack(values) * x.shape[0]
            examples += x.shape[0]

        means = (sums / examples).tolist()
        lower_active, top_active = active_statistics.counts()
        print(
            f"epoch {epoch:03d}: beta={kl_weight:.2f}, loss={means[0]:.3f}, "
            f"D={means[1]:.3f}, KL(z1|z2)={means[2]:.3f}, "
            f"KL(z2)={means[3]:.3f}, AU=({lower_active},{top_active})"
        )
        model.eval()
        with torch.inference_mode():
            samples = model.sample(64, device=device)
            latents = model.infer(x[:8], sample=False)
            reconstructions = model.decode(latents["q1_mu"])
            counterfactual = model.decode(latents["p1_mu"])
        save_image(samples, out_dir / f"samples_{epoch:03d}.png", nrow=8)
        save_image(
            torch.cat((x[:8], reconstructions, counterfactual)),
            out_dir / f"lower_layer_counterfactual_{epoch:03d}.png",
            nrow=8,
        )

    torch.save(
        {
            "format_version": 1,
            "model_name": "ladder_vae",
            "state_dict": model.state_dict(),
            "model_config": {"z1_dim": args.z1_dim, "z2_dim": args.z2_dim},
            "free_bits_per_layer_nats": args.free_bits,
            "warmup_epochs": args.warmup_epochs,
        },
        out_dir / "ladder_vae.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--free-bits", type=float, default=0.5)
    parser.add_argument("--z1-dim", type=int, default=32)
    parser.add_argument("--z2-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
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
