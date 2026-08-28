# Deep Learning with PyTorch

An educational PyTorch repository for introductory deep learning and generative
AI. Lessons are organized in a suggested reading order; shared utilities and
operational workflows live outside the lesson directories so their behavior
remains explicit and reusable.

## Quick start

The project uses a locked, project-local uv environment:

```bash
uv sync --locked
uv run --locked --no-sync python tool_scripts/pytorch_test.py
```

The second command is a read-only check of the installed PyTorch/CUDA runtime.
Python is pinned by [`.python-version`](.python-version), direct dependencies
by [`pyproject.toml`](pyproject.toml), and exact resolution by
[`uv.lock`](uv.lock).

## Explore the repository

| Goal | Start here |
|---|---|
| Core deep-learning lessons | [`deep_learning/`](deep_learning/) |
| Generative-AI lessons | [`genai/`](genai/) |
| Shared data, models, training, and plotting utilities | [`dl_utils/README.md`](dl_utils/README.md) |
| Dataset preparation, local utilities, cloud workflows, and benchmarks | [`tool_scripts/README.md`](tool_scripts/README.md) |
| Project dependencies and tooling configuration | [`pyproject.toml`](pyproject.toml) |

The lesson directory names are numbered in suggested reading order; they do
not correspond directly to book chapter numbers.

## Documentation map

Read the short overview first, then open a detailed reference only for the
task at hand:

| Area | Overview | Detailed reference |
|---|---|---|
| Shared utility package | [`dl_utils/README.md`](dl_utils/README.md) | [`dl_utils/MODULE_REFERENCE.md`](dl_utils/MODULE_REFERENCE.md) |
| Repository scripts | [`tool_scripts/README.md`](tool_scripts/README.md) | [`tool_scripts/SCRIPT_REFERENCE.md`](tool_scripts/SCRIPT_REFERENCE.md) |
| Generative-model lesson routes | [GAN roadmap](genai/1.0_generative_adversarial_network/0.0-ROADMAP.md), [VAE roadmap](genai/2.0_variational_autoencoder/0.0-ROADMAP.md), [diffusion roadmap](genai/3.0_diffusion_model/0.0-ROADMAP.md) | Relevant lesson source file |

## Datasets

For a minimal CelebA downloader profile:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset celeba
```

For all lesson datasets, synchronize the `examples` extra and use
`--dataset all`. Dataset choices, external-provider requirements, and derived
caches are documented in the
[script reference](tool_scripts/SCRIPT_REFERENCE.md#dataset-preparation).

## Cloud GAN workflows and benchmarks

`cloud_gan.sh` provisions and operates on an existing cloud GPU host; it does
not create, stop, or destroy provider resources. GPU and storage billing remain
the operator's responsibility. The full lifecycle, host requirements, MPS/tmux
operation, result download, and troubleshooting live in the
[script reference](tool_scripts/SCRIPT_REFERENCE.md#cloud-gpu-workflow).

GAN benchmarks require prepared CelebA data and a compatible CUDA GPU. See the
[benchmark guide](tool_scripts/SCRIPT_REFERENCE.md#gan-benchmarks) before
running `test_gan_concurrency.sh`, `benchmark_h100_precision.sh`, or
`benchmark_stylegan_gpu.sh`.

## Quality checks

Run checks from the locked project environment:

```bash
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv run --locked --no-sync pyright
```

## References

- [Dive into Deep Learning](https://d2l.ai) — Aston Zhang et al.
- [Deep Learning](https://www.deeplearningbook.org) — Ian Goodfellow, Yoshua
  Bengio, and Aaron Courville.
- [Deep Generative Modeling](https://link.springer.com/book/10.1007/978-3-031-64087-2) —
  Jakub M. Tomczak.
- [Learn Generative AI with PyTorch](https://www.manning.com/books/learn-generative-ai-with-pytorch) —
  Mark Liu.
