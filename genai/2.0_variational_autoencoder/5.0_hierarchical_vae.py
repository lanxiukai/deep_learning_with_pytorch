"""Two-level HVAE baseline with matched top-down prior/posterior conditions.

The generator is

    p(z2) p(z1 | z2) p(x | z1),

and the recognition model is

    q(z2 | x) q(z1 | z2, x).

The negative ELBO therefore contains a top KL against N(0, I) and a lower KL
between distributions conditioned on the same sampled z2.  KL warm-up and
group-wise free bits are visible optimization controls, not guarantees that a
layer is informative.  The next lesson changes only the lower posterior.
"""

from __future__ import annotations

import argparse

import torch

from dl_utils.data.factor_shapes import (
    FACTOR_NAMES,
    FACTOR_SIZES,
    render_factor_shapes,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.hierarchy_training import (
    make_factor_shape_loaders,
    train_hierarchy,
)
from dl_utils.vae.vae_hierarchy import (
    HierarchicalVAE32,
    hierarchical_vae_loss,
)


PROJECT_ROOT = infer_project_root()


def smoke_test() -> None:
    set_seed(7)
    factors = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 2, 3, 4],
            [2, 3, 5, 6, 6],
            [0, 2, 3, 1, 5],
        ]
    )
    x = render_factor_shapes(factors)
    model = HierarchicalVAE32(
        z1_dim=8,
        z2_dim=4,
        hidden_channels=32,
        context_dim=32,
    )
    reconstruction, latents = model(x)
    loss, terms = hierarchical_vae_loss(
        reconstruction,
        x,
        latents,
        kl_weight=0.5,
        free_bits=1.0,
    )
    loss.backward()
    assert reconstruction.shape == x.shape
    assert latents["q1_mu"].shape == (4, 8)
    assert latents["q2_mu"].shape == (4, 4)
    assert model.lower_prior[-1].weight.grad is not None
    assert model.lower_posterior[-1].weight.grad is not None
    assert all(torch.isfinite(value) for value in terms.values())
    print(
        "smoke test passed: "
        f"D={terms['distortion'].item():.3f}, "
        f"KL1={terms['kl_z1'].item():.3f}, "
        f"KL2={terms['kl_z2'].item():.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--z1-dim", type=int, default=24)
    parser.add_argument("--z2-dim", type=int, default=12)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=float, default=10.0)
    parser.add_argument("--free-bits", type=float, default=1.0)
    parser.add_argument(
        "--active-variance-threshold", type=float, default=1e-2
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_factor_shape_loaders(
        batch_size=args.batch_size,
        workers=args.workers,
        split_seed=args.split_seed,
        device=device,
    )
    model = HierarchicalVAE32(
        z1_dim=args.z1_dim,
        z2_dim=args.z2_dim,
        hidden_channels=args.hidden_channels,
        context_dim=args.context_dim,
    ).to(device)
    train_hierarchy(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        free_bits=args.free_bits,
        active_variance_threshold=args.active_variance_threshold,
        seed=args.seed,
        out_dir=(
            PROJECT_ROOT
            / "output"
            / "hierarchical_vae"
            / args.run_name
        ),
        checkpoint_metadata={
            "model_name": "hierarchical_vae",
            "roadmap_role": "teaching_transition",
            "roadmap_step": 4,
            "direct_baseline": "standard VAE",
            "visible_increment": "second latent and conditional prior",
            "seed": args.seed,
            "dataset": "FactorShapes32",
            "factor_names": FACTOR_NAMES,
            "factor_sizes": FACTOR_SIZES,
            "split_seed": args.split_seed,
            "image_size": 32,
            "observation": "independent Bernoulli mean",
            "loss_reduction": "sum pixels per sample, then mean batch",
        },
    )


if __name__ == "__main__":
    main()
