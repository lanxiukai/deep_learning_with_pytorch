"""Evaluation helpers for controlled-factor VAE representation experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def quantile_bin_edges(reference_codes: Tensor, bins: int) -> Tensor:
    """Fit per-coordinate quantile edges without using factor labels."""
    if reference_codes.ndim != 2:
        raise ValueError("reference_codes must have shape [samples, latent]")
    if bins < 2:
        raise ValueError("bins must be at least two")
    quantiles = torch.linspace(
        0.0, 1.0, bins + 1, device=reference_codes.device
    )[1:-1]
    return torch.quantile(reference_codes, quantiles, dim=0).transpose(0, 1)


def discretize_codes(codes: Tensor, edges: Tensor) -> Tensor:
    """Apply independently fitted bin edges to each latent coordinate."""
    if codes.ndim != 2 or edges.ndim != 2:
        raise ValueError("codes and edges must both be matrices")
    if codes.shape[1] != edges.shape[0]:
        raise ValueError("edges need one row per latent coordinate")
    columns = [
        torch.bucketize(
            codes[:, index].contiguous(),
            edges[index].contiguous(),
        )
        for index in range(codes.shape[1])
    ]
    return torch.stack(columns, dim=1)


def discrete_mutual_information(
    left: Tensor,
    right: Tensor,
    *,
    left_size: int,
    right_size: int,
) -> Tensor:
    """Empirical mutual information between two integer-valued vectors."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left and right must have matching vector shapes")
    joint_index = left.to(torch.long) * right_size + right.to(torch.long)
    joint = torch.bincount(
        joint_index, minlength=left_size * right_size
    ).reshape(left_size, right_size).to(dtype=torch.float64)
    joint /= joint.sum()
    left_probability = joint.sum(dim=1, keepdim=True)
    right_probability = joint.sum(dim=0, keepdim=True)
    positive = joint > 0
    ratio = joint / (left_probability * right_probability).clamp_min(1e-12)
    return (joint[positive] * ratio[positive].log()).sum()


def factor_entropy(factor: Tensor, size: int) -> Tensor:
    probability = torch.bincount(
        factor.to(torch.long), minlength=size
    ).to(dtype=torch.float64)
    probability /= probability.sum()
    positive = probability > 0
    return -(probability[positive] * probability[positive].log()).sum()


def mutual_information_matrix(
    discrete_codes: Tensor,
    factors: Tensor,
    *,
    code_bins: int,
    factor_sizes: tuple[int, ...],
) -> Tensor:
    """Return [latent, factor] empirical mutual information."""
    if discrete_codes.ndim != 2 or factors.ndim != 2:
        raise ValueError("codes and factors must both be matrices")
    if discrete_codes.shape[0] != factors.shape[0]:
        raise ValueError("codes and factors must have the same sample count")
    if factors.shape[1] != len(factor_sizes):
        raise ValueError("factor_sizes must describe every factor column")
    matrix = torch.empty(
        discrete_codes.shape[1],
        factors.shape[1],
        dtype=torch.float64,
    )
    for latent_index in range(discrete_codes.shape[1]):
        for factor_index, factor_size in enumerate(factor_sizes):
            matrix[latent_index, factor_index] = discrete_mutual_information(
                discrete_codes[:, latent_index],
                factors[:, factor_index],
                left_size=code_bins,
                right_size=factor_size,
            )
    return matrix


def mig_and_modularity(
    mutual_information: Tensor,
    factors: Tensor,
    *,
    factor_sizes: tuple[int, ...],
) -> dict[str, Tensor]:
    """Return MIG compactness and a complementary per-latent modularity score."""
    if mutual_information.ndim != 2:
        raise ValueError("mutual_information must be [latent, factor]")
    if mutual_information.shape[1] != len(factor_sizes):
        raise ValueError("factor_sizes must match the MI factor dimension")
    if mutual_information.shape[0] < 2:
        raise ValueError("MIG needs at least two latent coordinates")
    entropies = torch.stack(
        [
            factor_entropy(factors[:, index], size)
            for index, size in enumerate(factor_sizes)
        ]
    )
    top_two = mutual_information.topk(2, dim=0).values
    per_factor_mig = (top_two[0] - top_two[1]) / entropies.clamp_min(1e-12)

    maximum, assignment = mutual_information.max(dim=1)
    ideal = torch.zeros_like(mutual_information)
    ideal[
        torch.arange(mutual_information.shape[0]), assignment
    ] = maximum
    denominator = maximum.square() * max(
        1, mutual_information.shape[1] - 1
    )
    deviation = (
        (mutual_information - ideal).square().sum(dim=1)
        / denominator.clamp_min(1e-12)
    )
    modularity_per_latent = torch.where(
        maximum > 0,
        1.0 - deviation,
        torch.zeros_like(deviation),
    )
    return {
        "mig": per_factor_mig.mean(),
        "mig_per_factor": per_factor_mig,
        "modularity": modularity_per_latent.mean(),
        "modularity_per_latent": modularity_per_latent,
    }


class FactorProbe(nn.Module):
    """Small supervised probe for representation informativeness only."""

    def __init__(
        self,
        latent_dim: int,
        factor_sizes: tuple[int, ...],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, size) for size in factor_sizes]
        )

    def forward(self, codes: Tensor) -> list[Tensor]:
        features = self.features(codes)
        return [head(features) for head in self.heads]


def train_factor_probe(
    train_codes: Tensor,
    train_factors: Tensor,
    test_codes: Tensor,
    test_factors: Tensor,
    *,
    factor_sizes: tuple[int, ...],
    hidden_dim: int = 64,
    steps: int = 300,
    learning_rate: float = 1e-2,
    seed: int = 0,
    device: torch.device,
) -> dict[str, Tensor]:
    """Fit one fixed-capacity probe and return held-out factor accuracies."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    mean = train_codes.mean(dim=0, keepdim=True)
    scale = train_codes.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_codes = ((train_codes - mean) / scale).to(device)
    test_codes = ((test_codes - mean) / scale).to(device)
    train_factors = train_factors.to(device)
    test_factors = test_factors.to(device)
    probe = FactorProbe(
        train_codes.shape[1], factor_sizes, hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    for _ in range(steps):
        logits = probe(train_codes)
        loss = torch.stack(
            [
                F.cross_entropy(output, train_factors[:, index])
                for index, output in enumerate(logits)
            ]
        ).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.inference_mode():
        accuracies = torch.stack(
            [
                (
                    output.argmax(dim=1) == test_factors[:, index]
                ).float().mean()
                for index, output in enumerate(probe(test_codes))
            ]
        ).cpu()
    return {
        "mean_accuracy": accuracies.mean(),
        "per_factor_accuracy": accuracies,
    }


__all__ = [
    "FactorProbe",
    "discrete_mutual_information",
    "discretize_codes",
    "factor_entropy",
    "mig_and_modularity",
    "mutual_information_matrix",
    "quantile_bin_edges",
    "train_factor_probe",
]
