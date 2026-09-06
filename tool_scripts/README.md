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
| Check CelebA-64 GAN throughput and finite losses | `benchmark_rtx5090_gans.sh` | Sequentially exercises the three lesson models and writes JSON results |
| Implement a benchmark wrapper | `benchmark_gan_training.py` | Internal worker; prefer a shell wrapper |

## Safety boundaries

- Dataset and visualization commands can write under `data/` and `output/`.
- Cloud commands use an instance that already exists. They do not create, stop,
  or destroy provider resources, so billing remains your responsibility.
- `cloud_gan.sh download` previews an rsync merge by default; pass `--apply`
  only after reviewing the plan.
- Fresh benchmark runs reset their selected output directory so each result
  belongs to the current run.

## Dataset profiles

Base dependencies cover torchvision datasets and the project's small download
registry. Run the downloader without arguments to prepare every dataset in its
documented order. This includes archive extraction and the required CelebA
and glasses derived data:

```bash
uv sync --extra examples --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py
```

The default sequence is `mnist`, `fashion-mnist`, `cifar10`, `house-prices`,
`time-machine`, `celeba`, `anime-face`, `glasses`, `airfoil`, `fra-eng`,
`pokemon`.

Select one or several datasets by listing them after `--dataset`. Selections
run in the order given, and duplicates are ignored after their first
appearance:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset mnist cifar10 celeba glasses
```

SN-GAN, SAGAN, BigGAN, VQ-VAE, FSQ, and VQGAN default to aligned CelebA under
`data/celeba`, using its official train and validation partitions. GANs use
64x64 faces; discrete tokenizers retain their 128x128 setup. Smiling labels
condition GANs and the tokenizers' second-stage priors.

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
