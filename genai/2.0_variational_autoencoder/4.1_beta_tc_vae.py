"""Beta-TCVAE: target aggregated-posterior dependence at the same data scale.

The network and FactorShapes32 protocol match `4.0_beta_vae.py`.  Only the
objective changes:

    distortion + rate + (beta - 1) * TC_MWS.

TC is estimated from a [batch, batch, latent] log-density tensor.  The raw
minibatch-weighted score and its parameter-independent centered form are both
logged: centering makes values readable but does not turn the surrogate into
an unbiased dataset-level TC estimate.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.factor_shapes import (
    FACTOR_NAMES,
    FACTOR_SIZES,
    FactorShapes32,
    render_factor_shapes,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import reproducibility_metadata
from dl_utils.vae.inference import GaussianVAE32, model_config
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    diagonal_gaussian_log_density,
    reparameterize_logvar,
)


PROJECT_ROOT = infer_project_root()


def parse_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(dict.fromkeys(float(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("beta values must be at least one")
    return values


def parse_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item) for item in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def number_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def mws_kl_decomposition(
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    dataset_size: int,
) -> dict[str, Tensor]:
    """Estimate MI, TC, and dimension-wise KL with minibatch weighting."""
    batch_size, latent_dim = z.shape
    if batch_size < 2:
        raise ValueError("the minibatch estimator needs batch_size >= 2")
    if dataset_size < batch_size:
        raise ValueError("dataset_size must be at least the batch size")
    if mu.shape != z.shape or logvar.shape != z.shape:
        raise ValueError("z, mu, and logvar must have matching [B, D] shapes")

    log_q_z_given_x = diagonal_gaussian_log_density(
        z, mu, logvar
    ).sum(dim=1)
    pair_log_q = diagonal_gaussian_log_density(
        z[:, None, :], mu[None, :, :], logvar[None, :, :]
    )
    log_mws_weight = math.log(dataset_size * batch_size)
    log_q_z = (
        torch.logsumexp(pair_log_q.sum(dim=2), dim=1)
        - log_mws_weight
    )
    log_product_q_z = (
        torch.logsumexp(pair_log_q, dim=1) - log_mws_weight
    ).sum(dim=1)
    log_p_z = diagonal_gaussian_log_density(
        z, torch.zeros_like(z), torch.zeros_like(z)
    ).sum(dim=1)

    mutual_information_raw = (log_q_z_given_x - log_q_z).mean()
    total_correlation_raw = (log_q_z - log_product_q_z).mean()
    dimension_kl_raw = (log_product_q_z - log_p_z).mean()
    log_dataset_size = math.log(dataset_size)
    return {
        "mutual_information_centered": (
            mutual_information_raw - log_dataset_size
        ),
        "total_correlation_centered": (
            total_correlation_raw
            - (latent_dim - 1) * log_dataset_size
        ),
        "dimension_kl_centered": (
            dimension_kl_raw + latent_dim * log_dataset_size
        ),
        "total_correlation_mws_raw": total_correlation_raw,
    }


def beta_tc_vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    beta: float,
    dataset_size: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return D + R + (beta - 1) TC with all coefficients explicit."""
    if beta < 1:
        raise ValueError("beta must be at least one")
    distortion = F.binary_cross_entropy(
        reconstruction, target, reduction="none"
    ).flatten(1).sum(dim=1).mean()
    rate = diagonal_gaussian_kl_from_logvar(
        mu, logvar
    ).sum(dim=1).mean()
    decomposition = mws_kl_decomposition(
        z, mu, logvar, dataset_size=dataset_size
    )
    total_correlation = decomposition["total_correlation_centered"]
    loss = (
        distortion
        + rate
        + (float(beta) - 1.0) * total_correlation
    )
    return loss, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "mutual_information_centered": decomposition[
            "mutual_information_centered"
        ].detach(),
        "total_correlation_centered": total_correlation.detach(),
        "dimension_kl_centered": decomposition[
            "dimension_kl_centered"
        ].detach(),
        "total_correlation_mws_raw": decomposition[
            "total_correlation_mws_raw"
        ].detach(),
    }


def make_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    train_set = FactorShapes32(
        split="train", split_seed=args.split_seed
    )
    test_set = FactorShapes32(
        split="test", split_seed=args.split_seed
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(test_set, shuffle=False, drop_last=False, **common),
    )


@torch.inference_mode()
def evaluate(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    beta: float,
    training_dataset_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(6, device=device)
    examples = 0
    means = []
    for x, _ in loader:
        if x.shape[0] < 2:
            continue
        x = x.to(device, non_blocking=True)
        mu, logvar, _ = model.encode(x)
        z = reparameterize_logvar(mu, logvar)
        reconstruction = model.decode(mu)
        distortion = F.binary_cross_entropy(
            reconstruction, x, reduction="none"
        ).flatten(1).sum(dim=1).mean()
        rate = diagonal_gaussian_kl_from_logvar(
            mu, logvar
        ).sum(dim=1).mean()
        decomposition = mws_kl_decomposition(
            z,
            mu,
            logvar,
            dataset_size=training_dataset_size,
        )
        totals += torch.stack(
            [
                distortion,
                rate,
                decomposition["mutual_information_centered"],
                decomposition["total_correlation_centered"],
                decomposition["dimension_kl_centered"],
                decomposition["total_correlation_mws_raw"],
            ]
        ) * x.shape[0]
        examples += x.shape[0]
        means.append(mu.cpu())
    values = (totals / examples).tolist()
    all_means = torch.cat(means)
    return {
        "distortion": values[0],
        "rate": values[1],
        "mutual_information_centered": values[2],
        "total_correlation_centered": values[3],
        "dimension_kl_centered": values[4],
        "total_correlation_mws_raw": values[5],
        "objective": values[0]
        + values[1]
        + (beta - 1.0) * values[3],
        "active_units": float(
            (all_means.var(dim=0, unbiased=False) > 1e-2).sum()
        ),
    }


def train_one(
    args: argparse.Namespace,
    *,
    beta: float,
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    set_seed(seed)
    model = GaussianVAE32(
        latent_dim=args.latent_dim,
        hidden_channels=args.hidden_channels,
        context_dim=args.context_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset_size = len(train_loader.dataset)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = torch.zeros(7, device=device)
        examples = 0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            mu, logvar, _ = model.encode(x)
            z = reparameterize_logvar(mu, logvar)
            reconstruction = model.decode(z)
            loss, terms = beta_tc_vae_loss(
                reconstruction,
                x,
                z,
                mu,
                logvar,
                beta=beta,
                dataset_size=dataset_size,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += torch.stack(
                [loss.detach(), *terms.values()]
            ) * x.shape[0]
            examples += x.shape[0]
        values = (totals / examples).tolist()
        print(
            f"beta={beta:g}, seed={seed}, epoch {epoch:03d}: "
            f"loss={values[0]:.3f}, D={values[1]:.3f}, "
            f"R={values[2]:.3f}, MIc={values[3]:.3f}, "
            f"TCc={values[4]:.3f}, dimKLc={values[5]:.3f}, "
            f"TCraw={values[6]:.3f}"
        )

    validation = evaluate(
        model,
        test_loader,
        beta=beta,
        training_dataset_size=dataset_size,
        device=device,
    )
    out_dir = (
        PROJECT_ROOT
        / "output"
        / "vae"
        / "beta_tc_vae"
        / f"beta_{number_slug(beta)}"
        / f"seed_{seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_name": "beta_tc_vae",
            "roadmap_role": "focused_branch",
            "roadmap_step": "3X",
            "direct_baseline": "beta-VAE at matched per-sample KL rate",
            "visible_increment": "explicit aggregate-posterior TC weighting",
            "state_dict": model.state_dict(),
            "model_config": model_config(model),
            "beta": beta,
            "seed": seed,
            "dataset": "FactorShapes32",
            "factor_names": FACTOR_NAMES,
            "factor_sizes": FACTOR_SIZES,
            "split_seed": args.split_seed,
            "image_size": 32,
            "observation": "independent Bernoulli mean",
            "loss_reduction": "sum pixels per sample, then mean batch",
            "tc_estimator": "minibatch-weighted sampling, centered for logging",
            "validation_metrics": validation,
            **reproducibility_metadata(
                models={"beta_tc_vae": model},
                seed=seed,
                training_budget={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "optimizer_updates": args.epochs * len(train_loader),
                },
                data_preprocessing={
                    "renderer": "FactorShapes32 deterministic renderer",
                    "value_range": [0.0, 1.0],
                    "augmentation": "none",
                },
            ),
        },
        out_dir / "model.pth",
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    print(
        f"beta={beta:g}, seed={seed}: validation "
        f"D={validation['distortion']:.3f}, R={validation['rate']:.3f}, "
        f"TCc={validation['total_correlation_centered']:.3f}; "
        f"saved {out_dir}"
    )


def smoke_test() -> None:
    set_seed(7)
    factors = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 2, 3, 4],
            [2, 3, 5, 6, 6],
            [0, 2, 3, 1, 5],
            [1, 0, 4, 2, 2],
            [2, 2, 1, 5, 1],
        ]
    )
    x = render_factor_shapes(factors)
    model = GaussianVAE32(
        latent_dim=6, hidden_channels=32, context_dim=16
    )
    mu, logvar, _ = model.encode(x)
    z = reparameterize_logvar(mu, logvar)
    reconstruction = model.decode(z)
    loss, terms = beta_tc_vae_loss(
        reconstruction,
        x,
        z,
        mu,
        logvar,
        beta=4.0,
        dataset_size=3_000,
    )
    loss.backward()
    assert model.base_posterior.weight.grad is not None
    assert all(torch.isfinite(value) for value in terms.values())
    print(
        "smoke test passed: "
        f"D={terms['distortion'].item():.3f}, "
        f"R={terms['rate'].item():.3f}, "
        f"TCc={terms['total_correlation_centered'].item():.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--betas", type=parse_floats, default=(2.0, 4.0, 6.0)
    )
    parser.add_argument(
        "--seeds", type=parse_ints, default=(11, 22, 33)
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args, device)
    for beta in args.betas:
        for seed in args.seeds:
            train_one(
                args,
                beta=beta,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
            )


if __name__ == "__main__":
    main()
