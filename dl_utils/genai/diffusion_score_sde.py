"""Continuous VP-SDE marginals and explicit reverse-time score samplers.

The module contains the continuous probability path and its numerical reverse
updates, independent of the U-Net used to estimate the score.  Euler--Maruyama
handles the stochastic reverse SDE; Euler or Heun handles the deterministic
probability-flow ODE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


ScoreSampler = Literal["reverse_sde", "probability_flow"]

__all__ = ["ScoreSampler", "VPSDE", "sample_vp_sde"]


def _randn_like(sample: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        sample.shape,
        device=sample.device,
        dtype=sample.dtype,
        generator=generator,
    )


@dataclass(frozen=True)
class VPSDE:
    """Continuous linear-beta variance-preserving SDE.

    ``dx = -0.5 beta(t) x dt + sqrt(beta(t)) dw`` with
    ``beta(t) = beta_min + t (beta_max - beta_min)``.
    """

    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 < self.beta_min < self.beta_max:
            raise ValueError("expected 0 < beta_min < beta_max")

    def beta(self, time: Tensor) -> Tensor:
        return self.beta_min + time * (self.beta_max - self.beta_min)

    def marginal_coefficients(
        self,
        time: Tensor,
        sample: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if time.ndim != 1 or time.shape[0] != sample.shape[0]:
            raise ValueError("time must have shape [batch]")
        log_alpha = (
            -0.25 * (self.beta_max - self.beta_min) * time.square()
            - 0.5 * self.beta_min * time
        )
        alpha = log_alpha.exp()
        sigma = (1.0 - (2.0 * log_alpha).exp()).clamp(min=1e-12).sqrt()
        shape = (sample.shape[0], *((1,) * (sample.ndim - 1)))
        return alpha.reshape(shape), sigma.reshape(shape)

    def marginal_sample(
        self,
        x0: Tensor,
        time: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha, sigma = self.marginal_coefficients(time, x0)
        return alpha * x0 + sigma * noise, alpha, sigma

    def score_target(self, noise: Tensor, sigma: Tensor) -> Tensor:
        return -noise / sigma


def _guided_score(
    model: nn.Module,
    sample: Tensor,
    time: Tensor,
    labels: Tensor | None,
    guidance_scale: float,
    time_embedding_scale: float,
) -> Tensor:
    embedded_time = time * time_embedding_scale
    if labels is None:
        return model(sample, embedded_time, None)
    conditional = model(sample, embedded_time, labels)
    if guidance_scale == 1.0:
        return conditional
    unconditional = model(sample, embedded_time, None)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.no_grad()
def sample_vp_sde(
    model: nn.Module,
    sde: VPSDE,
    shape: Sequence[int],
    *,
    sampler: ScoreSampler = "probability_flow",
    num_steps: int = 250,
    time_epsilon: float = 1e-3,
    ode_solver: Literal["euler", "heun"] = "heun",
    time_embedding_scale: float = 1000.0,
    labels: Tensor | None = None,
    guidance_scale: float = 1.0,
    initial_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    trajectory_frames: int = 8,
) -> tuple[Tensor, list[Tensor]]:
    """Integrate the reverse VP-SDE or its probability-flow ODE.

    Euler--Maruyama is used for the stochastic reverse SDE.  The deterministic
    probability-flow ODE supports Euler and Heun, making the numerical solver a
    visible choice rather than hiding it behind a general ODE package.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 0.0 < time_epsilon < 1.0:
        raise ValueError("time_epsilon must lie in (0, 1)")
    if sampler not in ("reverse_sde", "probability_flow"):
        raise ValueError(f"unknown score sampler: {sampler}")
    if ode_solver not in ("euler", "heun"):
        raise ValueError(f"unknown ODE solver: {ode_solver}")

    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("model must contain parameters") from error
    device, dtype = parameter.device, parameter.dtype
    shape = tuple(shape)
    if len(shape) != 4 or shape[0] < 1:
        raise ValueError("shape must be [batch, channels, height, width]")

    if initial_noise is None:
        sample = torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError("initial_noise does not match requested shape")
        sample = initial_noise.to(device=device, dtype=dtype).clone()
    if labels is not None:
        labels = labels.to(device=device, dtype=torch.long)
        if labels.shape != (shape[0],):
            raise ValueError("labels must have shape [batch]")

    times = torch.linspace(1.0, time_epsilon, num_steps + 1, device=device)
    capture_positions: set[int] = set()
    trajectory: list[Tensor] = []
    if return_trajectory:
        if trajectory_frames < 2:
            raise ValueError("trajectory_frames must be at least 2")
        trajectory.append(sample.detach().cpu())
        capture_count = min(trajectory_frames - 1, num_steps)
        capture_positions = set(
            torch.linspace(0, num_steps - 1, capture_count)
            .round()
            .long()
            .tolist()
        )

    def vector_field(
        state: Tensor,
        time_batch: Tensor,
        probability_flow: bool,
    ) -> Tensor:
        beta = sde.beta(time_batch).reshape(
            state.shape[0], *((1,) * (state.ndim - 1))
        )
        score = _guided_score(
            model,
            state,
            time_batch,
            labels,
            guidance_scale,
            time_embedding_scale,
        )
        score_coefficient = 0.5 if probability_flow else 1.0
        return -0.5 * beta * state - score_coefficient * beta * score

    was_training = model.training
    model.eval()
    try:
        for index in range(num_steps):
            current_time = times[index]
            next_time = times[index + 1]
            dt = next_time - current_time
            time_batch = torch.full(
                (shape[0],), current_time, device=device, dtype=dtype
            )

            if sampler == "reverse_sde":
                drift = vector_field(sample, time_batch, probability_flow=False)
                mean = sample + drift * dt
                if index + 1 < num_steps:
                    beta = sde.beta(time_batch).reshape(
                        shape[0], *((1,) * (sample.ndim - 1))
                    )
                    sample = mean + (beta * (-dt)).sqrt() * _randn_like(
                        sample, generator
                    )
                else:
                    sample = mean
            else:
                field = vector_field(sample, time_batch, probability_flow=True)
                euler = sample + field * dt
                if ode_solver == "euler":
                    sample = euler
                else:
                    next_batch = torch.full(
                        (shape[0],), next_time, device=device, dtype=dtype
                    )
                    next_field = vector_field(
                        euler, next_batch, probability_flow=True
                    )
                    sample = sample + 0.5 * (field + next_field) * dt

            if return_trajectory and index in capture_positions:
                trajectory.append(sample.detach().cpu())

        final_time = torch.full(
            (shape[0],), time_epsilon, device=device, dtype=dtype
        )
        final_score = _guided_score(
            model,
            sample,
            final_time,
            labels,
            guidance_scale,
            time_embedding_scale,
        )
        alpha, sigma = sde.marginal_coefficients(final_time, sample)
        x0 = (sample + sigma.square() * final_score) / alpha
        if return_trajectory and trajectory:
            trajectory[-1] = x0.detach().cpu()
    finally:
        model.train(was_training)

    return x0.clamp(-1.0, 1.0), trajectory
