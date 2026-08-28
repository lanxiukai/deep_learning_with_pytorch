# Tool Scripts

This directory contains repository-level utilities for dataset preparation,
cloud GPU workflows, GAN benchmarks, environment inspection, and standalone
visualizations. Run commands from the repository root unless a script says
otherwise.

## Before running

Use the locked project environment for Python tools:

```bash
uv sync --locked
uv run --locked --no-sync python tool_scripts/pytorch_test.py
```

`pytorch_test.py` is a read-only runtime check. Shell scripts expose their
current options with `--help`.

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

## Detailed reference

[SCRIPT_REFERENCE.md](SCRIPT_REFERENCE.md) contains the full cloud runbook,
host requirements, environment profiles, benchmark interpretation, dataset
catalog, cleanup checklist, and troubleshooting guidance.

For focused work, start with this README, then open only the relevant script or
reference section. Do not load the full cloud workflow for a local inspection
or a small script change.
