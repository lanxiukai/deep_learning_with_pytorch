"""Compare IWAE bounds and IAF posterior families under one held-out protocol.

Every available checkpoint is re-evaluated with the same large K, particle
chunk size, test examples, and repeated random estimates.  This separates the
training objective from the likelihood estimator.  Posterior-mean
reconstruction from a deterministic posterior representative, Monte Carlo
rate, active units, ESS/K, and prior samples are reported alongside the bound;
none is treated as a substitute for the rest. For IAF, transforming the base
Gaussian mean is not generally the exact transformed-posterior expectation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.inference import (
    FlowGaussianVAE32,
    GaussianVAE32,
    importance_log_weights,
    log_mean_exp,
)


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"


def make_test_loader(
    args: argparse.Namespace, device: torch.device
) -> DataLoader:
    dataset = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )


def load_model(
    path: Path, device: torch.device
) -> tuple[GaussianVAE32, dict[str, object]]:
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )
    model_name = checkpoint.get("model_name")
    if model_name == "iwae":
        model: GaussianVAE32 = GaussianVAE32(
            **checkpoint["model_config"]
        )
    elif model_name == "iaf_vae":
        model = FlowGaussianVAE32(**checkpoint["model_config"])
    else:
        raise ValueError(f"{path} is not an IWAE/IAF checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def evaluate_model(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    particles: int,
    particle_chunk_size: int,
    repeats: int,
    max_examples: int,
    active_variance_threshold: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    repeat_bounds = []
    all_ess_fractions = []
    all_weight_ranges = []
    all_rates = []
    all_log_determinants = []
    posterior_codes = []
    distortion_total = 0.0
    distortion_examples = 0

    for repeat in range(repeats):
        set_seed(seed + repeat)
        bound_total = 0.0
        examples = 0
        for x, _ in loader:
            remaining = max_examples - examples
            if remaining <= 0:
                break
            x = x[:remaining].to(device, non_blocking=True)
            log_weights, terms = importance_log_weights(
                model,
                x,
                particles=particles,
                particle_chunk_size=particle_chunk_size,
            )
            per_example_bound = log_mean_exp(log_weights)
            normalized = torch.softmax(log_weights, dim=1)
            ess_fraction = (
                normalized.square().sum(dim=1).reciprocal() / particles
            )
            weight_range = (
                log_weights.max(dim=1).values
                - log_weights.min(dim=1).values
            )
            rate_samples = terms["log_q"] - terms["log_pz"]
            bound_total += float(per_example_bound.sum())
            all_ess_fractions.append(ess_fraction.cpu())
            all_weight_ranges.append(weight_range.cpu())
            all_rates.append(rate_samples.mean(dim=1).cpu())
            all_log_determinants.append(
                terms["log_determinant"].mean(dim=1).cpu()
            )
            if repeat == 0:
                representative = model.posterior_representative(x)
                reconstruction = model.decode(representative)
                distortion_total += float(
                    F.binary_cross_entropy(
                        reconstruction, x, reduction="sum"
                    )
                )
                distortion_examples += x.shape[0]
                posterior_codes.append(representative.cpu())
            examples += x.shape[0]
        repeat_bounds.append(bound_total / examples)

    ess = torch.cat(all_ess_fractions)
    ranges = torch.cat(all_weight_ranges)
    rates = torch.cat(all_rates)
    log_determinants = torch.cat(all_log_determinants)
    codes = torch.cat(posterior_codes)
    code_variance = codes.var(dim=0, unbiased=False)
    bounds = torch.tensor(repeat_bounds, dtype=torch.float64)
    return {
        "evaluation_particles": particles,
        "repeats": repeats,
        "examples_per_repeat": min(max_examples, len(loader.dataset)),
        "bound_mean": float(bounds.mean()),
        "bound_repeat_standard_deviation": float(
            bounds.std(unbiased=False)
        ),
        "repeat_bounds": repeat_bounds,
        "deterministic_posterior_representative_distortion": (
            distortion_total / distortion_examples
        ),
        "monte_carlo_rate": float(rates.mean()),
        "active_units": int(
            (code_variance > active_variance_threshold).sum()
        ),
        "ess_fraction_mean": float(ess.mean()),
        "ess_fraction_quantiles": [
            float(value)
            for value in torch.quantile(
                ess, torch.tensor([0.1, 0.5, 0.9])
            )
        ],
        "log_weight_range_mean": float(ranges.mean()),
        "mean_log_determinant": float(log_determinants.mean()),
    }


@torch.inference_mode()
def save_model_comparison(
    model: GaussianVAE32,
    loader: DataLoader,
    path: Path,
    *,
    device: torch.device,
) -> None:
    real, _ = next(iter(loader))
    real = real[:16].to(device)
    reconstructions = model.reconstruct(real)
    samples = model.sample(16, device=device)
    save_image(
        torch.cat((real, reconstructions, samples)),
        path,
        nrow=16,
    )


def checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "vae_elbo_k1": args.iwae_k1_checkpoint,
        "iwae_k5": args.iwae_k5_checkpoint,
        "iaf_elbo": args.iaf_checkpoint,
    }


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_test_loader(args, device)
    paths = checkpoint_paths(args)
    available = {name: path for name, path in paths.items() if path.exists()}
    if not available:
        raise FileNotFoundError(
            "no default checkpoint exists; run 3.0_iwae.py and/or "
            "3.1_iaf_vae.py first"
        )
    out_dir = OUTPUT_ROOT / "inference_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "protocol": {
            "dataset": "MNIST test",
            "image_size": 32,
            "observation": "independent Bernoulli mean",
            "evaluation_particles": args.evaluation_particles,
            "particle_chunk_size": args.particle_chunk_size,
            "repeats": args.repeats,
            "max_examples": args.max_examples,
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
            particles=args.evaluation_particles,
            particle_chunk_size=args.particle_chunk_size,
            repeats=args.repeats,
            max_examples=args.max_examples,
            active_variance_threshold=args.active_variance_threshold,
            seed=args.seed,
            device=device,
        )
        metrics["training_particles"] = checkpoint.get(
            "training_particles"
        )
        metrics["training_updates"] = checkpoint.get("training_updates")
        metrics["particle_evaluations"] = checkpoint.get(
            "particle_evaluations"
        )
        metrics["budget_protocol"] = checkpoint.get("budget_protocol")
        metrics["gradient_diagnostics"] = checkpoint.get(
            "gradient_diagnostics"
        )
        results["models"][name] = metrics
        save_model_comparison(
            model,
            loader,
            out_dir / f"{name}_real_reconstruction_prior.png",
            device=device,
        )
        print(
            f"{name}: bound={metrics['bound_mean']:.3f} +/- "
            f"{metrics['bound_repeat_standard_deviation']:.3f}, "
            f"ESS/K={metrics['ess_fraction_mean']:.3f}, "
            f"rate={metrics['monte_carlo_rate']:.3f}"
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved comparison to {out_dir}")


def smoke_test() -> None:
    set_seed(7)
    loader = DataLoader(
        list(
            zip(
                torch.rand(12, 1, 32, 32),
                torch.arange(12) % 10,
            )
        ),
        batch_size=4,
    )
    for model in (
        GaussianVAE32(
            latent_dim=4, hidden_channels=32, context_dim=16
        ),
        FlowGaussianVAE32(
            latent_dim=4,
            hidden_channels=32,
            context_dim=16,
            flow_layers=2,
            flow_hidden_dim=16,
        ),
    ):
        metrics = evaluate_model(
            model,
            loader,
            particles=5,
            particle_chunk_size=2,
            repeats=2,
            max_examples=8,
            active_variance_threshold=1e-3,
            seed=7,
            device=torch.device("cpu"),
        )
        assert metrics["examples_per_repeat"] == 8
        assert 0.0 < metrics["ess_fraction_mean"] <= 1.0
        assert torch.isfinite(torch.tensor(metrics["bound_mean"]))
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluation-particles", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--active-variance-threshold", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--iwae-k1-checkpoint",
        type=Path,
        default=OUTPUT_ROOT / "iwae" / "k1" / "model.pth",
    )
    parser.add_argument(
        "--iwae-k5-checkpoint",
        type=Path,
        default=OUTPUT_ROOT / "iwae" / "k5" / "model.pth",
    )
    parser.add_argument(
        "--iaf-checkpoint",
        type=Path,
        default=OUTPUT_ROOT / "iaf_vae" / "model.pth",
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
