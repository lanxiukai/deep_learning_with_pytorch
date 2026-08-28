"""Evaluate KL-control and TC representation branches.

Beta-VAE and beta-TCVAE use actual per-sample KL and nearest-rate pairs. Both
branches share active units, a fixed-capacity factor probe, MIG, and modularity
on the same procedural split.

The metrics use procedural ground-truth factors.  They describe this data and
inductive bias, not a theorem that unsupervised semantic factors are
identifiable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from dl_utils.data.factor_shapes import (
    FACTOR_NAMES,
    FACTOR_SIZES,
    FactorShapes32,
    all_factor_combinations,
    render_factor_shapes,
)
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.disentanglement import (
    discretize_codes,
    mig_and_modularity,
    mutual_information_matrix,
    quantile_bin_edges,
    train_factor_probe,
)
from dl_utils.vae.inference import GaussianVAE32
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar


PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.beta_vae_root.rglob("model.pth"))
    paths.extend(sorted(args.beta_tc_vae_root.rglob("model.pth")))
    return paths


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[GaussianVAE32, dict[str, object]]:
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )
    if checkpoint.get("model_name") not in {"beta_vae", "beta_tc_vae"}:
        raise ValueError(f"{path} is not a beta-VAE family checkpoint")
    model = GaussianVAE32(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def make_loaders(
    args: argparse.Namespace,
    *,
    split_seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_set = FactorShapes32(split="train", split_seed=split_seed)
    test_set = FactorShapes32(split="test", split_seed=split_seed)
    common = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    return DataLoader(train_set, **common), DataLoader(test_set, **common)


@torch.inference_mode()
def encode_dataset(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    calculate_reconstruction: bool,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    codes = []
    factors = []
    rates = []
    distortion = 0.0
    examples = 0
    for x, batch_factors in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar, _ = model.encode(x)
        per_dimension_rate = diagonal_gaussian_kl_from_logvar(mu, logvar)
        if calculate_reconstruction:
            reconstruction = model.decode(mu)
            distortion += float(
                F.binary_cross_entropy(
                    reconstruction, x, reduction="sum"
                )
            )
        codes.append(mu.cpu())
        rates.append(per_dimension_rate.cpu())
        factors.append(batch_factors)
        examples += x.shape[0]
    all_codes = torch.cat(codes)
    all_rates = torch.cat(rates)
    return all_codes, torch.cat(factors), {
        "distortion": (
            distortion / examples if calculate_reconstruction else None
        ),
        "gaussian_reference_kl": float(all_rates.sum(dim=1).mean()),
        "mean_rate_per_dimension": all_rates.mean(dim=0).tolist(),
        "active_units": int(
            (
                all_codes.var(dim=0, unbiased=False)
                > 1e-2
            ).sum()
        ),
        "active_kl_dimensions": int(
            (all_rates.mean(dim=0) > 0.05).sum()
        ),
    }


def evaluate_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
) -> dict[str, object]:
    model, checkpoint = load_checkpoint(path, device)
    train_loader, test_loader = loaders
    train_codes, train_factors, _ = encode_dataset(
        model,
        train_loader,
        calculate_reconstruction=False,
        device=device,
    )
    test_codes, test_factors, metrics = encode_dataset(
        model,
        test_loader,
        calculate_reconstruction=True,
        device=device,
    )
    edges = quantile_bin_edges(train_codes, args.code_bins)
    discrete_test_codes = discretize_codes(test_codes, edges)
    mutual_information = mutual_information_matrix(
        discrete_test_codes,
        test_factors,
        code_bins=args.code_bins,
        factor_sizes=FACTOR_SIZES,
    )
    representation = mig_and_modularity(
        mutual_information,
        test_factors,
        factor_sizes=FACTOR_SIZES,
    )
    probe = train_factor_probe(
        train_codes,
        train_factors,
        test_codes,
        test_factors,
        factor_sizes=FACTOR_SIZES,
        hidden_dim=args.probe_hidden_dim,
        steps=args.probe_steps,
        learning_rate=args.probe_lr,
        seed=args.probe_seed,
        device=device,
    )
    model_name = str(checkpoint["model_name"])
    training_budget = checkpoint.get("training_budget")
    data_preprocessing = checkpoint.get("data_preprocessing")
    if not isinstance(training_budget, dict) or not isinstance(
        data_preprocessing, dict
    ):
        raise ValueError(
            "checkpoint predates the reproducibility contract; retrain it"
        )
    rate = metrics.pop("gaussian_reference_kl")
    metrics.update(
        {
            "model_name": model_name,
            "control_name": "beta",
            "control_value": float(checkpoint["beta"]),
            "rate": rate,
            "seed": int(checkpoint["seed"]),
            "model_config": checkpoint["model_config"],
            "training_budget": training_budget,
            "data_preprocessing": data_preprocessing,
            "checkpoint": str(path.relative_to(PROJECT_ROOT)),
            "mig": float(representation["mig"]),
            "mig_per_factor": {
                name: float(value)
                for name, value in zip(
                    FACTOR_NAMES, representation["mig_per_factor"]
                )
            },
            "modularity": float(representation["modularity"]),
            "probe_mean_accuracy": float(probe["mean_accuracy"]),
            "probe_accuracy_per_factor": {
                name: float(value)
                for name, value in zip(
                    FACTOR_NAMES, probe["per_factor_accuracy"]
                )
            },
            "mutual_information_matrix": mutual_information.tolist(),
        }
    )
    return metrics


def aggregate_records(
    records: list[dict[str, object]]
) -> list[dict[str, object]]:
    groups: dict[
        tuple[str, str, float], list[dict[str, object]]
    ] = defaultdict(list)
    for record in records:
        groups[
            (
                str(record["model_name"]),
                str(record["control_name"]),
                float(record["control_value"]),
            )
        ].append(record)
    aggregate = []
    numeric_keys = (
        "distortion",
        "active_units",
        "mig",
        "modularity",
        "probe_mean_accuracy",
    )
    for (model_name, control_name, control_value), group in sorted(
        groups.items()
    ):
        summary: dict[str, object] = {
            "model_name": model_name,
            "control_name": control_name,
            "control_value": control_value,
            "seeds": [int(record["seed"]) for record in group],
        }
        for key in numeric_keys:
            values = torch.tensor(
                [float(record[key]) for record in group],
                dtype=torch.float64,
            )
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_standard_deviation"] = float(
                values.std(unbiased=False)
            )
        values = torch.tensor(
            [float(record["rate"]) for record in group],
            dtype=torch.float64,
        )
        summary["rate_mean"] = float(values.mean())
        summary["rate_standard_deviation"] = float(
            values.std(unbiased=False)
        )
        aggregate.append(summary)
    return aggregate


def match_rates(
    records: list[dict[str, object]]
) -> list[dict[str, object]]:
    beta_records = [
        record for record in records if record["model_name"] == "beta_vae"
    ]
    tc_records = [
        record
        for record in records
        if record["model_name"] == "beta_tc_vae"
    ]
    matches = []
    for tc_record in tc_records:
        candidates = [
            record
            for record in beta_records
            if record["seed"] == tc_record["seed"]
            and record["model_config"] == tc_record["model_config"]
            and record["training_budget"] == tc_record["training_budget"]
            and record["data_preprocessing"]
            == tc_record["data_preprocessing"]
        ]
        if not candidates:
            continue
        baseline = min(
            candidates,
            key=lambda record: abs(
                float(record["rate"]) - float(tc_record["rate"])
            ),
        )
        matches.append(
            {
                "seed": tc_record["seed"],
                "beta_tc_beta": tc_record["control_value"],
                "beta_vae_beta": baseline["control_value"],
                "beta_tc_rate": tc_record["rate"],
                "beta_vae_rate": baseline["rate"],
                "absolute_rate_gap": abs(
                    float(tc_record["rate"]) - float(baseline["rate"])
                ),
                "distortion_delta_tc_minus_beta": (
                    float(tc_record["distortion"])
                    - float(baseline["distortion"])
                ),
                "mig_delta_tc_minus_beta": (
                    float(tc_record["mig"]) - float(baseline["mig"])
                ),
                "modularity_delta_tc_minus_beta": (
                    float(tc_record["modularity"])
                    - float(baseline["modularity"])
                ),
                "probe_accuracy_delta_tc_minus_beta": (
                    float(tc_record["probe_mean_accuracy"])
                    - float(baseline["probe_mean_accuracy"])
                ),
            }
        )
    return matches


def save_summary_plot(
    records: list[dict[str, object]], path: Path
) -> None:
    markers = {"beta_vae": "o", "beta_tc_vae": "^"}
    colors = {
        "beta_vae": "tab:blue",
        "beta_tc_vae": "tab:orange",
    }
    with plt.ioff():
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for model_name in ("beta_vae", "beta_tc_vae"):
            group = [
                record
                for record in records
                if record["model_name"] == model_name
            ]
            if not group:
                continue
            label = "beta-VAE" if model_name == "beta_vae" else "beta-TCVAE"
            axes[0].scatter(
                [record["rate"] for record in group],
                [record["distortion"] for record in group],
                marker=markers[model_name],
                color=colors[model_name],
                label=label,
                alpha=0.8,
            )
            axes[1].scatter(
                [record["rate"] for record in group],
                [record["mig"] for record in group],
                marker=markers[model_name],
                color=colors[model_name],
                label=label,
                alpha=0.8,
            )
        axes[0].set(
            xlabel="Rate (nats / image)",
            ylabel="Bernoulli distortion",
            title="Rate-distortion working points",
        )
        axes[1].set(
            xlabel="Rate (nats / image)",
            ylabel="MIG",
            title="Compactness must be compared at matched rate",
        )
        for axis in axes:
            axis.grid(alpha=0.25)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.probe_seed)
    paths = discover_checkpoints(args)
    if not paths:
        raise FileNotFoundError(
            "no beta-VAE checkpoints found; run 4.0_beta_vae.py and "
            "4.1_beta_tc_vae.py first"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_seeds = set()
    metadata = []
    for path in paths:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
        if checkpoint.get("model_name") in {"beta_vae", "beta_tc_vae"}:
            split_seeds.add(int(checkpoint["split_seed"]))
            metadata.append(path)
    if len(split_seeds) != 1:
        raise ValueError(
            "all compared checkpoints must use one FactorShapes split seed"
        )
    split_seed = split_seeds.pop()
    loaders = make_loaders(args, split_seed=split_seed, device=device)
    records = []
    for path in metadata:
        record = evaluate_checkpoint(
            path, args=args, loaders=loaders, device=device
        )
        records.append(record)
        print(
            f"{record['model_name']} "
            f"{record['control_name']}={record['control_value']:g} "
            f"seed={record['seed']}: D={record['distortion']:.3f}, "
            f"R={record['rate']:.3f}, "
            f"MIG={record['mig']:.3f}, "
            f"probe={record['probe_mean_accuracy']:.3f}"
        )

    out_dir = OUTPUT_ROOT / "disentanglement_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": {
            "dataset": "FactorShapes32",
            "factor_names": FACTOR_NAMES,
            "factor_sizes": FACTOR_SIZES,
            "split_seed": split_seed,
            "code_bins": args.code_bins,
            "probe_steps": args.probe_steps,
            "comparison_contracts": {
                "beta_tc_vae": "nearest per-sample KL rate",
            },
            "identifiability_warning": (
                "Controlled-factor scores do not establish unsupervised "
                "semantic identifiability."
            ),
        },
        "runs": records,
        "aggregate_by_model_and_control": aggregate_records(records),
        "nearest_rate_pairs": match_rates(records),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    save_summary_plot(records, out_dir / "rate_distortion_and_mig.png")
    print(f"saved evaluation to {out_dir}")


def smoke_test() -> None:
    set_seed(7)
    factors = all_factor_combinations()[:320]
    codes = torch.randn(len(factors), 10) * 0.05
    codes[:, :5] += factors.to(torch.float32)
    train_codes, test_codes = codes[:240], codes[240:]
    train_factors, test_factors = factors[:240], factors[240:]
    edges = quantile_bin_edges(train_codes, bins=8)
    discrete = discretize_codes(test_codes, edges)
    mutual_information = mutual_information_matrix(
        discrete,
        test_factors,
        code_bins=8,
        factor_sizes=FACTOR_SIZES,
    )
    scores = mig_and_modularity(
        mutual_information,
        test_factors,
        factor_sizes=FACTOR_SIZES,
    )
    probe = train_factor_probe(
        train_codes,
        train_factors,
        test_codes,
        test_factors,
        factor_sizes=FACTOR_SIZES,
        hidden_dim=16,
        steps=5,
        learning_rate=1e-2,
        seed=7,
        device=torch.device("cpu"),
    )
    images = render_factor_shapes(factors[:4])
    assert images.shape == (4, 1, 32, 32)
    assert mutual_information.shape == (10, 5)
    assert torch.isfinite(scores["mig"])
    assert torch.isfinite(probe["mean_accuracy"])
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--code-bins", type=int, default=10)
    parser.add_argument("--probe-hidden-dim", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--beta-vae-root",
        type=Path,
        default=OUTPUT_ROOT / "beta_vae",
    )
    parser.add_argument(
        "--beta-tc-vae-root",
        type=Path,
        default=OUTPUT_ROOT / "beta_tc_vae",
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
