# `dl_utils`

`dl_utils` is the internal utility package shared by this repository's
PyTorch lessons. It provides reusable data, model, runtime, checkpoint, and
plotting primitives; lesson scripts remain the authoritative place for
training loops, experiment budgets, and model-specific objectives.

## Install

Run these commands from the repository root:

```bash
uv sync --locked
```

Use all optional lesson dependencies only when needed:

```bash
uv sync --all-extras --locked
```

For a lean CelebA-download profile, use:

```bash
uv sync --no-dev --extra celeba --locked
```

The `examples` extra contains larger lesson-only dependencies, and `celeba`
adds the Kaggle downloader.

## Find the right module

| Need | Start with |
|---|---|
| D2L-style textbook helpers | [d2l/](d2l/) |
| Dataset loading, downloads, and image preparation | [data/](data/) |
| Diffusion, EBM, GAN, or VAE building blocks | [diffusion/](diffusion/), [ebm/](ebm/), [gan/](gan/), or [vae/](vae/) |
| Device, precision, checkpoints, metrics, and optimization | [runtime/](runtime/) and [training/](training/) |
| Project paths, output directories, and figures | [filesystem/](filesystem/) and [plot/](plot/) |

Common focused entry points include:

- [data/celeba.py](data/celeba.py) and [data/vision.py](data/vision.py) for
  dataset loaders;
- [training/checkpoints.py](training/checkpoints.py),
  [training/accelerator.py](training/accelerator.py), and
  [training/metrics.py](training/metrics.py) for training infrastructure;
- [gan/training.py](gan/training.py) for shared GAN runtime and artifacts;
- [vae/](vae/) for the focused VAE learning route; and
- [plot/images.py](plot/images.py) and [plot/figures.py](plot/figures.py) for
  artifacts and figures.

## Scope and side effects

- Names listed in a module's `__all__` are public-API candidates; unprefixed
  names without it may still be lesson-facing helpers.
- Source files are the canonical behavior. Read their docstrings and call
  sites before changing an interface.
- Downloads, directory resets, artifact writes, and random-seed operations
  have external effects. Keep them explicit in a lesson or script.
- GAN artifacts default to `output/gan/` unless `DL_OUTPUT_ROOT` is set.

## Detailed reference

[MODULE_REFERENCE.md](MODULE_REFERENCE.md) contains the complete alphabetical
module inventory, import dependencies, and public-entry list. Keep it separate
from this overview so an agent or focused maintenance task can read only the
relevant source file or reference section instead of loading the entire
catalog.

When a module's imports or public entries change, update the corresponding
reference entry. Update this README only when the package map, setup path, or
common entry points change.
