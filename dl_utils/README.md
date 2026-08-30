# `dl_utils`

`dl_utils` is the internal utility package shared by this repository's
PyTorch lessons. It provides reusable data, model, runtime, checkpoint, and
plotting primitives; lesson scripts remain the authoritative place for
training loops, experiment budgets, and model-specific objectives.

## Environment

This package is installed in editable mode by the repository's
[locked project setup](../README.md#quick-start). It does not own a separate
environment or dependency workflow.

## Package map

| Area | Responsibility | Start with |
|---|---|---|
| [d2l/](d2l/) | D2L-style textbook helpers | The relevant lesson call site |
| [data/](data/) | Downloads, datasets, image preparation, and loaders | [celeba.py](data/celeba.py) and [vision.py](data/vision.py) |
| [diffusion/](diffusion/), [ebm/](ebm/), [gan/](gan/), [vae/](vae/) | Model-family building blocks | The importing lesson and focused source module |
| [runtime/](runtime/), [training/](training/) | Devices, precision, checkpoints, metrics, and optimization | [accelerator.py](training/accelerator.py), [checkpoints.py](training/checkpoints.py), and [metrics.py](training/metrics.py) |
| [filesystem/](filesystem/), [plot/](plot/) | Project paths, output directories, and figures | [figures.py](plot/figures.py) and [images.py](plot/images.py) |

## Design boundaries

- [diffusion/ddpm.py](diffusion/ddpm.py) is a compatibility facade. New
  lessons import the focused discrete diffusion, score-SDE, and U-Net modules
  directly.
- [gan/training.py](gan/training.py) owns shared BF16 runtime selection, data
  access, output paths, EMA setup, checkpoints, and sample artifacts. Lesson
  scripts retain model schedules, objectives, regularization, and update order.
- [vae/vae.py](vae/vae.py) preserves the 256x256 introductory VAE and the
  comparable standard/beta-VAE training path. The focused 32x32 modules cover
  hierarchical VAEs, discrete tokenizers and priors, and perceptual
  autoencoders.
- [training/checkpoints.py](training/checkpoints.py) owns serialization and
  state restoration; [training/session.py](training/session.py) manages output
  lifecycles without owning optimization loops.

## API and side effects

- Source files are canonical. Module docstrings explain purpose, and `__all__`
  identifies explicitly exported names; unprefixed names without it may still
  be lesson-facing helpers.
- Downloads, directory resets, artifact writes, and random-seed operations
  have external effects. Keep them explicit in a lesson or script.
- GAN artifacts default to `output/gan/` unless `DL_OUTPUT_ROOT` is set.
- Update this README only when the package map or cross-module design
  boundaries change; imports and symbol inventories belong to source and type
  checking.
