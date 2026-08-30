"""Test whether each stochastic VAE layer adds information beyond its prior.

For both the ordinary HVAE and Ladder posterior, this script reports per-layer
rate and active units, then performs three sampled reconstruction
counterfactuals:

* replace only q(z1 | z2, x) with p(z1 | z2);
* replace only q(z2 | x) with p(z2), retaining lower data evidence;
* replace both posterior layers with their matching priors.

It also saves two controlled resampling grids.  Any apparent global/local
division remains an empirical observation, not a property implied by the
hierarchical ELBO.

Data:
    FactorShapes32 test split, generated in memory with the split seed stored
    in the compared checkpoints. Factor annotations do not enter the models.

Checkpoints:
    output/vae/hierarchical_vae/baseline/model.pth: default HVAE
    output/vae/ladder_vae/baseline/model.pth: default Ladder VAE
    At least one checkpoint is required; missing models are reported.

Outputs:
    output/vae/hierarchical_vae_evaluation/metrics.json: comparison report
    output/vae/hierarchical_vae_evaluation/<model>/posterior_and_prior_replacements.png
    output/vae/hierarchical_vae_evaluation/<model>/fixed_top_resampled_lower_prior.png
    output/vae/hierarchical_vae_evaluation/<model>/fixed_evidence_changed_top.png

Evaluation data -- FactorShapes32 test:
Available and evaluated images:       706
Batch size:                            64
Intervention samples per image:         8
Visual variants per fixed input:        6
Active-variance threshold:           0.01

Default dimensions:
Evaluation input:                32x32 grayscale
Generated image:                 32x32 grayscale
Latent vectors:                  z1=24 / z2=12 values

Model size:
Hierarchical VAE:                 0.876 M parameters
Ladder VAE:                       0.874 M parameters
Loaded total:                     1.750 M parameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.factor_shapes import FactorShapes32
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)
from dl_utils.vae.vae_hierarchy import (
    ActiveUnitAccumulator,
    HierarchicalVAE32,
    LadderVAE32,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"


def load_model(
    path: Path, device: torch.device
) -> tuple[HierarchicalVAE32, dict[str, object]]:
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )
    model_name = checkpoint.get("model_name")
    if model_name == "hierarchical_vae":
        model: HierarchicalVAE32 = HierarchicalVAE32(
            **checkpoint["model_config"]
        )
    elif model_name == "ladder_vae":
        model = LadderVAE32(**checkpoint["model_config"])
    else:
        raise ValueError(f"{path} is not an HVAE/Ladder checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def make_test_loader(
    args: argparse.Namespace,
    *,
    split_seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = FactorShapes32(split="test", split_seed=split_seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )


def _sample_gaussian(mu: Tensor, logvar: Tensor, samples: int) -> Tensor:
    epsilon = torch.randn(
        mu.shape[0],
        samples,
        mu.shape[1],
        device=mu.device,
        dtype=mu.dtype,
    )
    return mu[:, None, :] + torch.exp(
        0.5 * logvar[:, None, :]
    ) * epsilon


def _decode_distortion(
    model: HierarchicalVAE32, z1: Tensor, target: Tensor
) -> Tensor:
    reconstruction = model.decode(z1)
    expanded_target = target[:, None, ...].expand_as(reconstruction)
    return F.binary_cross_entropy(
        reconstruction, expanded_target, reduction="none"
    ).flatten(2).sum(dim=2)


@torch.inference_mode()
def sampled_counterfactual_distortions(
    model: HierarchicalVAE32,
    x: Tensor,
    *,
    samples: int,
) -> dict[str, Tensor]:
    """Use paired top samples to isolate lower and upper posterior removal."""
    lower_evidence, q2_mu, q2_logvar = model.bottom_up(x)
    batch_size = x.shape[0]
    repeated_evidence = lower_evidence[:, None, :].expand(
        -1, samples, -1
    ).reshape(batch_size * samples, -1)

    q_z2 = _sample_gaussian(q2_mu, q2_logvar, samples)
    q_z2_flat = q_z2.reshape(batch_size * samples, model.z2_dim)
    p1_mu, p1_logvar, q1_mu, q1_logvar = model.lower_distributions(
        repeated_evidence, q_z2_flat
    )
    posterior_z1 = reparameterize_logvar(q1_mu, q1_logvar).reshape(
        batch_size, samples, model.z1_dim
    )
    lower_prior_z1 = reparameterize_logvar(
        p1_mu, p1_logvar
    ).reshape(batch_size, samples, model.z1_dim)

    p_z2 = torch.randn_like(q_z2)
    p_z2_flat = p_z2.reshape(batch_size * samples, model.z2_dim)
    p_top_p1_mu, p_top_p1_logvar, p_top_q1_mu, p_top_q1_logvar = (
        model.lower_distributions(repeated_evidence, p_z2_flat)
    )
    top_replaced_z1 = reparameterize_logvar(
        p_top_q1_mu, p_top_q1_logvar
    ).reshape(batch_size, samples, model.z1_dim)
    both_replaced_z1 = reparameterize_logvar(
        p_top_p1_mu, p_top_p1_logvar
    ).reshape(batch_size, samples, model.z1_dim)
    return {
        "posterior": _decode_distortion(model, posterior_z1, x),
        "lower_prior": _decode_distortion(model, lower_prior_z1, x),
        "top_prior": _decode_distortion(model, top_replaced_z1, x),
        "both_priors": _decode_distortion(model, both_replaced_z1, x),
    }


@torch.inference_mode()
def evaluate_model(
    model: HierarchicalVAE32,
    loader: DataLoader,
    *,
    intervention_samples: int,
    active_variance_threshold: float,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "posterior": 0.0,
        "lower_prior": 0.0,
        "top_prior": 0.0,
        "both_priors": 0.0,
    }
    deterministic_distortion = 0.0
    kl_z1 = 0.0
    kl_z2 = 0.0
    examples = 0
    active = ActiveUnitAccumulator()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        latents = model.infer(x, sample=False)
        reconstruction = model.decode(latents["z1"])
        deterministic_distortion += float(
            F.binary_cross_entropy(reconstruction, x, reduction="sum")
        )
        kl_z1 += float(
            diagonal_gaussian_kl_from_logvar(
                latents["q1_mu"],
                latents["q1_logvar"],
                latents["p1_mu"],
                latents["p1_logvar"],
            ).sum()
        )
        kl_z2 += float(
            diagonal_gaussian_kl_from_logvar(
                latents["q2_mu"], latents["q2_logvar"]
            ).sum()
        )
        active.update(latents)
        counterfactuals = sampled_counterfactual_distortions(
            model, x, samples=intervention_samples
        )
        for name, values in counterfactuals.items():
            totals[name] += float(values.sum()) / intervention_samples
        examples += x.shape[0]
    active_z1, active_z2 = active.counts(
        variance_threshold=active_variance_threshold
    )
    sampled_posterior = totals["posterior"] / examples
    return {
        "posterior_mean_distortion": (
            deterministic_distortion / examples
        ),
        "sampled_posterior_distortion": sampled_posterior,
        "lower_prior_replacement_distortion": (
            totals["lower_prior"] / examples
        ),
        "top_prior_replacement_distortion": (
            totals["top_prior"] / examples
        ),
        "both_priors_replacement_distortion": (
            totals["both_priors"] / examples
        ),
        "lower_replacement_delta": (
            totals["lower_prior"] / examples - sampled_posterior
        ),
        "top_replacement_delta": (
            totals["top_prior"] / examples - sampled_posterior
        ),
        "both_replacement_delta": (
            totals["both_priors"] / examples - sampled_posterior
        ),
        "kl_z1": kl_z1 / examples,
        "kl_z2": kl_z2 / examples,
        "total_rate": (kl_z1 + kl_z2) / examples,
        "active_z1_corrections": float(active_z1),
        "active_z2_units": float(active_z2),
    }


@torch.inference_mode()
def save_counterfactual_grids(
    model: HierarchicalVAE32,
    loader: DataLoader,
    out_dir: Path,
    *,
    variants: int,
    device: torch.device,
) -> dict[str, float]:
    x, _ = next(iter(loader))
    x = x[:8].to(device)
    lower_evidence, q2_mu, q2_logvar = model.bottom_up(x)
    posterior = model.infer_from_top(
        lower_evidence,
        q2_mu,
        q2_logvar,
        q2_mu,
        sample_lower=False,
    )
    zero_z2 = torch.zeros_like(q2_mu)
    top_replaced = model.infer_from_top(
        lower_evidence,
        q2_mu,
        q2_logvar,
        zero_z2,
        sample_lower=False,
    )
    zero_p1_mu, _, _, _ = model.lower_distributions(
        lower_evidence, zero_z2
    )
    summary = torch.cat(
        (
            x,
            model.decode(posterior["q1_mu"]),
            model.decode(posterior["p1_mu"]),
            model.decode(top_replaced["q1_mu"]),
            model.decode(zero_p1_mu),
        )
    )
    save_image(
        summary,
        out_dir / "posterior_and_prior_replacements.png",
        nrow=8,
    )

    fixed_top = q2_mu
    p1_mu, p1_logvar, _, _ = model.lower_distributions(
        lower_evidence, fixed_top
    )
    lower_samples = _sample_gaussian(
        p1_mu, p1_logvar, variants
    )
    lower_images = model.decode(lower_samples)
    save_image(
        lower_images.flatten(0, 1),
        out_dir / "fixed_top_resampled_lower_prior.png",
        nrow=variants,
    )

    top_samples = _sample_gaussian(
        q2_mu, q2_logvar, variants
    )
    repeated_evidence = lower_evidence[:, None, :].expand(
        -1, variants, -1
    ).reshape(-1, lower_evidence.shape[1])
    _, _, changed_q1_mu, _ = model.lower_distributions(
        repeated_evidence,
        top_samples.reshape(-1, model.z2_dim),
    )
    upper_images = model.decode(
        changed_q1_mu.reshape(8, variants, model.z1_dim)
    )
    save_image(
        upper_images.flatten(0, 1),
        out_dir / "fixed_evidence_changed_top.png",
        nrow=variants,
    )
    return {
        "fixed_top_lower_pixel_standard_deviation": float(
            lower_images.std(dim=1).mean()
        ),
        "fixed_evidence_top_pixel_standard_deviation": float(
            upper_images.std(dim=1).mean()
        ),
    }


def checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "hierarchical_vae": args.hvae_checkpoint,
        "ladder_vae": args.ladder_checkpoint,
    }


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = checkpoint_paths(args)
    available = {name: path for name, path in paths.items() if path.exists()}
    if not available:
        raise FileNotFoundError(
            "no hierarchy checkpoint exists; run 5.0 and/or 5.1 first"
        )
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in available.items()
    }
    split_seeds = {
        int(checkpoint["split_seed"])
        for checkpoint in checkpoints.values()
    }
    if len(split_seeds) != 1:
        raise ValueError("compared checkpoints must use one data split")
    loader = make_test_loader(
        args, split_seed=split_seeds.pop(), device=device
    )
    out_root = OUTPUT_ROOT / "hierarchical_vae_evaluation"
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "protocol": {
            "dataset": "FactorShapes32 test",
            "intervention_samples": args.intervention_samples,
            "active_variance_threshold": args.active_variance_threshold,
            "interpretation": (
                "Layer interventions test incremental information; they do "
                "not prove a semantic global/local hierarchy."
            ),
        },
        "missing_optional_checkpoints": [
            name for name, path in paths.items() if not path.exists()
        ],
        "models": {},
    }
    for name, path in available.items():
        model, checkpoint = load_model(path, device)
        metrics = evaluate_model(
            model,
            loader,
            intervention_samples=args.intervention_samples,
            active_variance_threshold=args.active_variance_threshold,
            device=device,
        )
        model_out_dir = out_root / name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        metrics.update(
            save_counterfactual_grids(
                model,
                loader,
                model_out_dir,
                variants=args.visual_variants,
                device=device,
            )
        )
        metrics["checkpoint"] = str(path.relative_to(PROJECT_ROOT))
        metrics["posterior_family"] = checkpoint["posterior_family"]
        metrics["warmup_epochs"] = checkpoint["warmup_epochs"]
        metrics["free_bits_per_group"] = checkpoint[
            "free_bits_per_group"
        ]
        results["models"][name] = metrics
        print(
            f"{name}: KL1={metrics['kl_z1']:.3f}, "
            f"KL2={metrics['kl_z2']:.3f}, "
            f"lower delta={metrics['lower_replacement_delta']:.3f}, "
            f"top delta={metrics['top_replacement_delta']:.3f}"
        )
    (out_root / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved evaluation to {out_root}")


def smoke_test() -> None:
    set_seed(7)
    dataset = FactorShapes32(split="test")
    loader = DataLoader(dataset, batch_size=4)
    for model in (
        HierarchicalVAE32(
            z1_dim=6,
            z2_dim=3,
            hidden_channels=32,
            context_dim=24,
        ),
        LadderVAE32(
            z1_dim=6,
            z2_dim=3,
            hidden_channels=32,
            context_dim=24,
        ),
    ):
        metrics = evaluate_model(
            model,
            loader,
            intervention_samples=2,
            active_variance_threshold=1e-3,
            device=torch.device("cpu"),
        )
        assert torch.isfinite(
            torch.tensor(metrics["sampled_posterior_distortion"])
        )
        assert metrics["total_rate"] >= 0
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--intervention-samples", type=int, default=8)
    parser.add_argument("--visual-variants", type=int, default=6)
    parser.add_argument(
        "--active-variance-threshold", type=float, default=1e-2
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--hvae-checkpoint",
        type=Path,
        default=(
            OUTPUT_ROOT
            / "hierarchical_vae"
            / "baseline"
            / "model.pth"
        ),
    )
    parser.add_argument(
        "--ladder-checkpoint",
        type=Path,
        default=OUTPUT_ROOT / "ladder_vae" / "baseline" / "model.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
