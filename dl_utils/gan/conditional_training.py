"""Shared optimization loop for the class-conditional hinge-GAN lessons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from dl_utils.gan.sn_gan import (
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.training.accelerator import BF16Precision, FP32Precision
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.optimization import UpdateRatioSchedule


@dataclass(frozen=True)
class ConditionalHingeEpochResult:
    """Mean losses and global discriminator phase after one epoch."""

    discriminator: float
    generator_total: float
    generator_adversarial: float
    generator_regularization: float
    discriminator_steps: int


def train_conditional_hinge_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader,
    optimizer_g,
    optimizer_d,
    device: torch.device,
    precision: BF16Precision | FP32Precision,
    *,
    z_dim: int,
    num_classes: int,
    generator_batch_size: int,
    discriminator_steps: int,
    update_schedule: UpdateRatioSchedule,
    generator_regularizer: Callable[[nn.Module], torch.Tensor] | None = None,
    after_generator_step: Callable[[], None] | None = None,
    progress_bar=None,
) -> ConditionalHingeEpochResult:
    """Train one epoch while keeping each model's update ratio explicit."""
    if discriminator_steps < 0:
        raise ValueError("discriminator_steps must be non-negative.")
    if generator_batch_size < 1:
        raise ValueError("generator_batch_size must be positive.")
    discriminator_metrics = MetricAccumulator(("loss",), device=device)
    generator_metrics = MetricAccumulator(
        ("total", "adversarial", "regularization"),
        device=device,
    )

    for real, labels in loader:
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = real.shape[0]

        optimizer_d.zero_grad(set_to_none=True)
        with precision.autocast():
            noise = torch.randn(batch_size, z_dim, device=device)
            fake_labels = torch.randint(
                num_classes,
                (batch_size,),
                device=device,
            )
            with torch.no_grad():
                fake = generator(noise, fake_labels)
            real_scores = discriminator(real, labels)
            fake_scores = discriminator(fake, fake_labels)
            loss_d = discriminator_hinge_loss(real_scores, fake_scores)
        precision.backward_step(loss_d, optimizer_d)

        discriminator_metrics.update((loss_d,), num_examples=batch_size)
        discriminator_steps += 1
        if update_schedule.generator_due(discriminator_steps):
            sampled_labels = torch.randint(
                num_classes,
                (generator_batch_size,),
                device=device,
            )
            noise = torch.randn(generator_batch_size, z_dim, device=device)
            discriminator.requires_grad_(False)
            try:
                optimizer_g.zero_grad(set_to_none=True)
                with precision.autocast():
                    fake = generator(noise, sampled_labels)
                    loss_g_adversarial = generator_hinge_loss(
                        discriminator(fake, sampled_labels)
                    )
                loss_g_regularization = (
                    loss_g_adversarial.new_zeros(())
                    if generator_regularizer is None
                    else generator_regularizer(generator)
                )
                loss_g_total = loss_g_adversarial.float() + loss_g_regularization
                precision.backward_step(loss_g_total, optimizer_g)
                if after_generator_step is not None:
                    after_generator_step()
            finally:
                discriminator.requires_grad_(True)

            generator_metrics.update(
                (
                    loss_g_total,
                    loss_g_adversarial,
                    loss_g_regularization,
                ),
                num_examples=generator_batch_size,
            )

        if progress_bar is not None:
            progress_bar.update(1)

    generator_losses = generator_metrics.compute()
    return ConditionalHingeEpochResult(
        discriminator=discriminator_metrics.compute()["loss"],
        generator_total=generator_losses["total"],
        generator_adversarial=generator_losses["adversarial"],
        generator_regularization=generator_losses["regularization"],
        discriminator_steps=discriminator_steps,
    )


__all__ = [
    "ConditionalHingeEpochResult",
    "train_conditional_hinge_epoch",
]
