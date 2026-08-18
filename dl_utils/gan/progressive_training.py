"""Image-budget scheduling primitives for progressively growing GANs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTick:
    """One recoverable slice of a fixed-resolution progressive phase."""

    resolution: int
    name: str
    tick_index: int
    num_ticks: int
    batch_size: int
    num_batches: int
    phase_batch_start: int
    phase_num_batches: int

    @property
    def num_images(self):
        return self.num_batches * self.batch_size


def build_progressive_schedule(
    num_examples: int,
    *,
    resolutions: Sequence[int],
    batch_sizes: Mapping[int, int],
    phase_kimg: int,
    tick_kimg: int,
) -> tuple[TrainingTick, ...]:
    """Split every progressive phase into exact, recoverable image ticks."""
    if num_examples < 1:
        raise ValueError("num_examples must be positive.")
    if phase_kimg < 1 or tick_kimg < 1:
        raise ValueError("phase_kimg and tick_kimg must be positive.")
    if phase_kimg % tick_kimg:
        raise ValueError("phase_kimg must be divisible by tick_kimg.")

    normalized_resolutions = tuple(int(value) for value in resolutions)
    if not normalized_resolutions:
        raise ValueError("resolutions must not be empty.")
    if len(set(normalized_resolutions)) != len(normalized_resolutions):
        raise ValueError("resolutions must not contain duplicates.")

    ticks_per_phase = phase_kimg // tick_kimg
    ticks = []
    for resolution in normalized_resolutions:
        if resolution not in batch_sizes:
            raise ValueError(
                f"batch size is missing for resolution {resolution}."
            )
        batch_size = int(batch_sizes[resolution])
        if batch_size < 1:
            raise ValueError("batch sizes must be positive.")
        dataset_batches = num_examples // batch_size
        if dataset_batches == 0:
            raise ValueError(
                f"batch size {batch_size} exceeds dataset size {num_examples}"
            )

        target_images = phase_kimg * 1_000
        if target_images % batch_size:
            raise ValueError(
                "phase image budget must be divisible by every batch size"
            )
        phase_num_batches = target_images // batch_size
        base_batches, extra_batches = divmod(
            phase_num_batches,
            ticks_per_phase,
        )
        phase_names = (
            ("stabilization",)
            if resolution == normalized_resolutions[0]
            else ("fade-in", "stabilization")
        )
        for name in phase_names:
            phase_batch_start = 0
            for tick_index in range(ticks_per_phase):
                num_batches = base_batches + (
                    tick_index < extra_batches
                )
                if num_batches > dataset_batches:
                    raise ValueError(
                        "one training tick exceeds a complete dataset pass"
                    )
                ticks.append(
                    TrainingTick(
                        resolution=resolution,
                        name=name,
                        tick_index=tick_index,
                        num_ticks=ticks_per_phase,
                        batch_size=batch_size,
                        num_batches=num_batches,
                        phase_batch_start=phase_batch_start,
                        phase_num_batches=phase_num_batches,
                    )
                )
                phase_batch_start += num_batches
    return tuple(ticks)


def phase_alpha(tick: TrainingTick, batch_index: int) -> float:
    """Return the linear fade-in coefficient for one scheduled batch."""
    if not 0 <= batch_index < tick.num_batches:
        raise ValueError("batch_index must identify a batch in the tick.")
    if tick.name != "fade-in":
        return 1.0
    if tick.phase_num_batches == 1:
        return 1.0
    current_step = tick.phase_batch_start + batch_index
    return current_step / (tick.phase_num_batches - 1)


__all__ = [
    "TrainingTick",
    "build_progressive_schedule",
    "phase_alpha",
]
