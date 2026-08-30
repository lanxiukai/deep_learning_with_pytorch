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
| Provision, validate, train, monitor, or download a cloud GAN run | `cloud_gan.sh` | Can mutate the remote host and write/download artifacts |
| Compare GAN workloads | `test_gan_concurrency.sh`, `benchmark_h100_precision.sh`, or `benchmark_stylegan_gpu.sh` | Consumes GPU time and writes benchmark results |
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
registry. Kaggle-backed datasets require the `celeba` or `examples` extra; the
complete lesson collection also needs `examples`:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset celeba

uv sync --extra examples --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset all
```

The downloader's `--help` output lists the current dataset names and optional
CelebA CycleGAN split. Kaggle-backed downloads may require authentication.
The `all` operation reports individual provider failures after attempting the
complete collection.

## Cloud GAN runbook

[CLOUD_GAN_RUNBOOK.md](CLOUD_GAN_RUNBOOK.md) contains the host requirements,
provisioning and training lifecycle, benchmark interpretation, cleanup
checklist, and troubleshooting guidance.

For focused work, start with this README, then open only the relevant script or
runbook section. Exact actions, flags, choices, and defaults belong to each
script's `--help` output.
