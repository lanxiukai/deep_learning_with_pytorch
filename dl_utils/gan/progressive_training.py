"""Phase scheduling primitives for progressively growing GANs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressivePhase:
    """One fade-in or stabilization phase at a fixed resolution."""

    resolution: int
    name: str
    batch_size: int
    num_batches: int

    @property
    def num_images(self):
        return self.num_batches * self.batch_size


def build_progressive_schedule(
    *,
    resolutions: Sequence[int],
    batch_sizes: Mapping[int, int],
    phase_kimg: int,
) -> tuple[ProgressivePhase, ...]:
    """Build the fade-in and stabilization phases used by ProGAN."""
    if phase_kimg < 1:
        raise ValueError("phase_kimg must be positive.")

    resolutions = tuple(int(value) for value in resolutions)
    if not resolutions:
        raise ValueError("resolutions must not be empty.")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must not contain duplicates.")

    phases = []
    for resolution in resolutions:
        if resolution not in batch_sizes:
            raise ValueError(
                f"batch size is missing for resolution {resolution}."
            )
        batch_size = int(batch_sizes[resolution])
        target_images = phase_kimg * 1_000
        if batch_size < 1 or target_images % batch_size:
            raise ValueError(
                "phase image budget must be divisible by positive batch sizes"
            )
        names = (
            ("stabilization",)
            if resolution == resolutions[0]
            else ("fade-in", "stabilization")
        )
        phases.extend(
            ProgressivePhase(
                resolution=resolution,
                name=name,
                batch_size=batch_size,
                num_batches=target_images // batch_size,
            )
            for name in names
        )
    return tuple(phases)


def phase_alpha(phase: ProgressivePhase, batch_index: int) -> float:
    """Return the linear fade-in coefficient for one phase batch."""
    if not 0 <= batch_index < phase.num_batches:
        raise ValueError("batch_index must identify a batch in the phase.")
    if phase.name != "fade-in" or phase.num_batches == 1:
        return 1.0
    return batch_index / (phase.num_batches - 1)


__all__ = [
    "ProgressivePhase",
    "build_progressive_schedule",
    "phase_alpha",
]
