# Tool Scripts

This README is the operating index for repository-level tools: dataset
preparation, cloud GPU entry points, GAN benchmarks, environment inspection,
and standalone visualizations. Exact CLI options remain in each script's
`--help` output.

## Running tools

Run commands from the repository root. Prepare the environment through the
[project quick start](../README.md#quick-start), then use these invocation
patterns:

```bash
uv run --locked --no-sync python tool_scripts/SCRIPT.py
bash tool_scripts/SCRIPT.sh --help
```

`pytorch_test.py` is the read-only runtime check used by the quick start.

## Choose a task

| Goal | Start with | Main effect |
|---|---|---|
| Inspect the local PyTorch/CUDA runtime | `pytorch_test.py` | Read-only |
| Download or prepare a lesson dataset | `download_dataset.py` | Downloads data and may build derived caches |
| Create a local visualization | `plot_fashion_mnist.py`, `sgd_animation.py`, or `word_frequency.py` | Downloads data when needed and writes under `output/` |
| Prepare an existing cloud GPU host | `setup_cloud_gpu.sh` | Installs host prerequisites and synchronizes the project environment |
| Provision, smoke-test, train, monitor, or download a cloud GAN run | `cloud_gan.sh` | Operates SN-GAN, SAGAN, or BigGAN on an existing RTX 5090 host |
| Check Imagenette-128 GAN throughput and finite losses | `benchmark_rtx5090_gans.sh` | Sequentially exercises the three full-width models and writes JSON results |
| Implement a benchmark wrapper | `benchmark_gan_training.py` | Internal worker; prefer a shell wrapper |

## Safety boundaries

- Dataset and visualization commands can write under `data/` and `output/`.
- Cloud commands use an instance that already exists. They do not create, stop,
  or destroy provider resources, so billing remains your responsibility.
- `cloud_gan.sh download` previews an rsync merge by default; pass `--apply`
  only after reviewing the plan.
- Benchmark output directories must be new, protecting prior measurements from
  accidental reuse.

## Dataset profiles

Base dependencies cover torchvision datasets and the project's small download
registry. Run the downloader without arguments to prepare every dataset in its
documented order. This includes the required CelebA and glasses derived data,
archive extraction, and filtered Imagenette-128 cache preparation:

```bash
uv sync --extra examples --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py
```

The default sequence is `mnist`, `fashion-mnist`, `cifar10`, `house-prices`,
`time-machine`, `celeba`, `anime-face`, `glasses`, `airfoil`, `fra-eng`,
`pokemon`, then `imagenette`. Imagenette downloads directly from the official
fast.ai-hosted archive without authentication. Its archive, extracted
Imagenette-320 source, and derived cache fit comfortably in a typical teaching
workspace.

Select one or several datasets by listing them after `--dataset`. Selections
run in the order given, and duplicates are ignored after their first
appearance:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset mnist cifar10 celeba glasses

uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset imagenette
```

Imagenette preparation is resumable. It keeps the official 320-pixel source,
center-crops and bilinearly resizes each retained image from the official train
and validation splits once, then writes quality-95 JPEGs under
`data/imagenette-128/{train,val}/<WNID>`. By default it excludes sources with a
short edge below 128 pixels or more than 50 million total pixels and records
them in
`data/imagenette-128/.imagenette-128-excluded.jsonl`. Use
`--imagenette-min-source-edge`, `--imagenette-max-source-pixels`, and
`--imagenette-workers` to change those preparation settings. Training and
evaluation read only a cache with a compatible completion marker.

Explicit `--dataset all` is equivalent to omitting the option. The downloader
continues through the selected sequence after individual provider failures and
reports all failed datasets at the end. Selecting CelebA always prepares the
black/blond CycleGAN splits; selecting glasses always classifies, corrects, and
builds the 256-pixel cache.

## Cloud GAN runbook

[CLOUD_GAN_RUNBOOK.md](CLOUD_GAN_RUNBOOK.md) contains the host requirements,
provisioning and training lifecycle, benchmark interpretation, cleanup
checklist, and troubleshooting guidance.

For focused work, start with this README, then open only the relevant script or
runbook section. Exact actions, flags, choices, and defaults belong to each
script's `--help` output.
