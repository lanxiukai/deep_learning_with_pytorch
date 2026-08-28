# Tool Scripts

This directory contains repository-level utilities for dataset preparation,
cloud GPU workflows, GAN benchmarks, environment inspection, and standalone
visualizations. Run commands from the repository root unless a section says
otherwise.

The shell scripts print their current command-line options with `--help`. This
document focuses on how the scripts fit together, where to run them, and which
operations have persistent or billable effects.

## Script index

| Script | Role | Typical environment |
|---|---|---|
| `cloud_gan.sh` | Main entry point for provisioning, validation, detached GAN training, monitoring, and result download | WSL/local host and cloud GPU host |
| `setup_cloud_gpu.sh` | Prepare the locked project environment on an existing Ubuntu GPU host | Cloud GPU host |
| `test_gan_concurrency.sh` | Compare sequential and parallel BF16 GAN workloads | Supported cloud GPU |
| `benchmark_h100_precision.sh` | Compare strict FP32, TF32, and BF16 with one fixed workload | NVIDIA H100 |
| `benchmark_stylegan_gpu.sh` | Run or compare fixed-workload single-GPU StyleGAN measurements | CUDA GPU |
| `benchmark_gan_training.py` | Internal fixed-burst benchmark worker used by the shell benchmark scripts | CUDA GPU |
| `download_dataset.py` | Download and prepare one lesson dataset or the complete collection | Any project host |
| `plot_fashion_mnist.py` | Save grayscale and binarized Fashion-MNIST sample grids | Any project host |
| `pytorch_test.py` | Print the installed PyTorch and visible CUDA device information | Any project host |
| `sgd_animation.py` | Render the standalone SGD landscape animation | Any project host |
| `word_frequency.py` | Plot word counts for the prepared *Time Machine* dataset | Any project host |

`benchmark_gan_training.py` is primarily an implementation detail. Prefer the
shell benchmark wrappers because they add GPU monitoring, logs, metadata, and
comparisons.

## Cloud GPU workflow

### Scope and safety

The cloud scripts operate on an instance that you have already rented or
created. They do not create, stop, destroy, or otherwise control a provider
instance, so provider billing continues until you stop or destroy that
instance yourself.

The workflow uses two machines:

- **WSL/local host:** owns the SSH configuration, starts remote provisioning,
  and downloads results.
- **Cloud GPU host:** owns the project environment, CelebA data, training
  processes, checkpoints, and benchmark output.

The default remote project path is
`/workspace/deep-learning-with-pytorch`. Cloud artifacts are written under
`output-vast-dl/`, which is intentionally separate from the local lesson
default `output/`. Both `data/` and `output-vast-dl/` are ignored by Git.

`cloud_gan.sh provision` has persistent effects on the remote host. It:

1. runs `apt-get update` and installs `ca-certificates`, `curl`, and `git`;
2. creates `/workspace` when needed;
3. shallow-clones this repository, or runs `git pull --ff-only` in the
   existing checkout;
4. installs the pinned Python with uv and synchronizes the locked CelebA
   profile; and
5. downloads and validates CelebA directly on the remote host.

Provisioning requires root access or passwordless/non-interactive `sudo` for
system package installation. It does not install or update the NVIDIA driver
or CUDA Toolkit.

### Requirements

On the WSL/local host:

- `ssh` and `rsync`;
- an SSH config host, `vast-dl` by default; and
- enough local disk space for downloaded checkpoints, images, logs, and
  benchmark measurements.

On the cloud host:

- Ubuntu Linux on x86-64;
- a visible NVIDIA H100, B200, B300, RTX 5080, or RTX 5090 for the orchestrated
  GAN workflow;
- NVIDIA driver 580 or newer;
- BF16 support in the selected GPU and PyTorch runtime;
- a writable `/workspace` directory; and
- enough remote disk space for the uv/Python cache, project environment,
  CelebA, checkpoints, and generated samples.

The orchestration is designed for one visible GPU. On a multi-GPU instance,
isolate the intended GPU at the environment or container level before using
the training orchestration. The focused benchmark scripts additionally accept
`--gpu-index`.

An SSH entry can use any provider-specific host, user, port, and key. Keep
those details in `~/.ssh/config`, not in tracked repository files:

```sshconfig
Host vast-dl
    HostName <instance-address>
    User <remote-user>
    Port <ssh-port>
    IdentityFile ~/.ssh/<private-key>
```

Verify the connection before provisioning:

```bash
ssh vast-dl 'uname -srm && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'
```

### End-to-end runbook

#### 1. Provision the existing instance

Run from WSL or the local host:

```bash
bash tool_scripts/cloud_gan.sh provision
```

Use a different SSH config alias when needed:

```bash
bash tool_scripts/cloud_gan.sh provision --host vast-dl-direct
```

If the remote checkout already exists, provisioning requires it to accept a
fast-forward-only pull. Resolve remote uncommitted changes or branch divergence
deliberately instead of deleting the checkout.

Provisioning clones from the public GitHub remote; it does not upload the
WSL/local worktree. Make sure the required revision is available from that
remote, or prepare the intended checkout on the cloud host manually.

For a manually cloned checkout, run only the environment setup on the cloud
host, then prepare CelebA:

```bash
cd /workspace/deep-learning-with-pytorch
bash tool_scripts/setup_cloud_gpu.sh --profile celeba
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset celeba
```

#### 2. Validate before a full run

Connect to the cloud host and enter the remote checkout:

```bash
ssh vast-dl
cd /workspace/deep-learning-with-pytorch
bash tool_scripts/cloud_gan.sh validate --mps
```

`validate` first runs short isolated ProGAN, StyleGAN, and StyleGAN2 smoke
checks. It then compares sequential and concurrent fixed 128x128 BF16 bursts.
The `--mps` option applies to the concurrency benchmark; the smoke checks
remain isolated lesson runs.

For a smaller diagnostic scope:

```bash
# All three smoke checks, without the concurrency benchmark
bash tool_scripts/cloud_gan.sh smoke

# Only the sequential-versus-parallel benchmark
bash tool_scripts/cloud_gan.sh benchmark --mps
```

Do not start a full training run until the smoke checks complete with finite
metrics and the output looks plausible.

#### 3. Start detached training

Start one to three full lesson runs in dedicated tmux sessions on the cloud
host:

```bash
bash tool_scripts/cloud_gan.sh train --models progan,stylegan --mps
```

Supported model names are `progan`, `stylegan`, and `stylegan2`. Without
`--mps`, the processes share the GPU through the normal CUDA scheduler. With
`--mps`, the script starts or reuses a persistent NVIDIA MPS daemon under
`/tmp/dl-gan-mps`.

Training uses a private tmux server named `gan-cloud` and sessions named
`gan-progan`, `gan-stylegan`, and `gan-stylegan2`. A second `train` command is
rejected while sessions remain active, preventing accidental duplicate runs.

To resume completed or interrupted models from their latest checkpoints:

```bash
bash tool_scripts/cloud_gan.sh train \
  --models progan,stylegan \
  --mps \
  --resume
```

Every selected model must already have
`output-vast-dl/<model>/checkpoints/latest.pth`. `--resume` applies to all
models in the command; it is not a best-effort partial resume.

#### 4. Monitor and control training

Run on the cloud host:

```bash
bash tool_scripts/cloud_gan.sh status
bash tool_scripts/cloud_gan.sh attach stylegan
```

`status` shows GPU utilization, active private tmux sessions, and recent log
lines. After attaching, detach without stopping training by pressing
`Ctrl-b`, then `d`.

To interrupt one model deliberately, attach to it and press `Ctrl-c`, or use
the private tmux socket explicitly:

```bash
tmux -L gan-cloud kill-session -t gan-stylegan
```

When all MPS-backed training sessions have ended, stop the persistent daemon:

```bash
bash tool_scripts/cloud_gan.sh stop-mps
```

The command refuses to stop MPS while any session remains on the private tmux
server. The `--mps` mode in `test_gan_concurrency.sh` is different: it creates
a temporary private daemon and cleans it up automatically when the benchmark
exits.

#### 5. Preview and download results

Return to WSL or the local host. Preview is the default and does not copy
files:

```bash
bash tool_scripts/cloud_gan.sh download
```

Review the rsync plan, then apply it:

```bash
bash tool_scripts/cloud_gan.sh download --apply
```

The download merges the remote `output-vast-dl/` tree into the local tree. It
does not pass `--delete`, so local-only files are retained. A differing remote
file at the same relative path can replace the local copy; keep unrelated runs
in distinct timestamped output directories.

Confirm that required checkpoints, samples, summaries, and logs are present
locally before stopping or destroying the provider instance. Neither
`download` nor `stop-mps` changes provider billing state.

### Action reference

| Action | Run from | Main effect |
|---|---|---|
| `provision [--host HOST]` | WSL/local | Mutates the remote system, checkout, Python environment, and dataset tree |
| `smoke` | Cloud host | Runs short isolated checks for all three GAN lessons |
| `validate [BENCHMARK OPTIONS]` | Cloud host | Runs `smoke`, then the concurrency benchmark |
| `benchmark [OPTIONS]` | Cloud host | Delegates to `test_gan_concurrency.sh` |
| `precision [OPTIONS]` | H100 cloud host | Delegates to `benchmark_h100_precision.sh` |
| `train --models LIST [OPTIONS]` | Cloud host | Starts detached full lesson runs in private tmux sessions |
| `status` | Cloud host | Reports GPU state, sessions, and recent logs |
| `attach MODEL` | Interactive cloud shell | Attaches to one model session |
| `stop-mps` | Cloud host | Stops the persistent training MPS daemon after sessions end |
| `download [--host HOST] [--apply]` | WSL/local | Previews or applies an rsync merge from the remote output tree |

Use `bash tool_scripts/cloud_gan.sh --help` and the delegated script's
`--help` output for the complete, current option list.

## Cloud environment profiles

`setup_cloud_gpu.sh` synchronizes directly from `.python-version`,
`pyproject.toml`, and `uv.lock`. Its profiles are:

| Profile | Installed project dependencies |
|---|---|
| `core` | Base dependencies, including PyTorch and torchvision |
| `celeba` | Core plus the CelebA/Kaggle dependency, without development tools |
| `examples` | Core plus optional example dependencies |
| `full` | All optional dependencies and development/test tools |

Useful checks include:

```bash
# Print the planned system and uv operations without applying them
bash tool_scripts/setup_cloud_gpu.sh --profile celeba --dry-run

# Do not let setup install missing curl, git, or tmux packages
bash tool_scripts/setup_cloud_gpu.sh --profile celeba --skip-system-packages
```

The default persistent download/cache directory is `.cache/cloud-gpu/` inside
the remote project. `--state-dir` can move it to another persistent volume.
`--skip-gpu-check` is intended for image preparation or diagnostics; it does
not make GPU training work without a compatible visible device.

## GAN benchmarks

All GAN benchmark scripts require prepared CelebA data and a compatible CUDA
runtime. Output directories must not already exist, which protects previous
measurements from accidental reuse.

### Concurrency

```bash
bash tool_scripts/test_gan_concurrency.sh \
  --models progan,stylegan,stylegan2 \
  --mps
```

The result directory defaults to
`output-vast-dl/concurrency-benchmark/<UTC timestamp>/` and contains:

- sequential and parallel logs;
- per-model timing TSV files;
- per-mode `nvidia-smi` samples in `gpu.csv`; and
- `summary.txt` with elapsed times, aggregate speedup, and per-model results.

Use the aggregate speedup together with per-model throughput, peak memory, GPU
samples, and finite-metric checks. A faster aggregate time alone is not enough
to establish that every model benefits.

### Precision on H100

```bash
bash tool_scripts/cloud_gan.sh precision
```

This runs the same fixed StyleGAN load in strict FP32, TF32, and BF16. Inspect
`bf16_vs_tf32_speedup` in the generated `summary.txt` for the practical
training precision comparison; `bf16_vs_fp32_speedup` measures the gain over
strict FP32 CUDA math.

### Single-GPU comparison

Run identical commands on the baseline and candidate GPUs:

```bash
bash tool_scripts/benchmark_stylegan_gpu.sh \
  --batches 512 \
  --output-dir output-vast-dl/single-gpu-benchmark/<run-name>
```

Then compare the generated `training.tsv` files, passing the baseline first:

```bash
bash tool_scripts/benchmark_stylegan_gpu.sh --compare \
  <baseline>/training.tsv \
  <candidate>/training.tsv
```

Use `--compare-precision` instead when the hardware and workload are identical
but precision modes differ. Comparisons reject mismatched workload metadata.
The default 22,528 timed batches are a sustained benchmark and may run for
hours; use a smaller `--batches` value for an initial check.

## Dataset preparation

Use the locked project environment. Install the optional dependencies needed
by the requested dataset before running the downloader. For CelebA:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset celeba
```

For the complete lesson collection:

```bash
uv sync --extra examples --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset all
```

`--dataset all` continues after individual provider failures and reports the
failed dataset names at the end. Kaggle-backed downloads may require Kaggle
authentication. Use `--prepare-celeba-cyclegan` with a CelebA-only run to also
build the black-hair/blond-hair CycleGAN split.

## Standalone utilities

Run these utilities through the locked project environment:

```bash
# Verify the PyTorch/CUDA runtime
uv run --locked --no-sync python tool_scripts/pytorch_test.py

# Save Fashion-MNIST sample grids under output/
uv run --locked --no-sync python tool_scripts/plot_fashion_mnist.py

# Render output/sgd_animation.gif
uv run --locked --no-sync python tool_scripts/sgd_animation.py

# Requires data/time_machine/timemachine.txt
uv run --locked --no-sync python tool_scripts/word_frequency.py
```

Use `plot_fashion_mnist.py --help` for its data directory, resize, and output
path options. The SGD animation and word-frequency scripts currently use fixed
output paths under `output/`.

## Cost and cleanup checklist

Cloud work can continue consuming billable GPU time after an SSH disconnect.
Before leaving an instance:

1. check `cloud_gan.sh status` for active sessions;
2. verify whether any non-tmux CUDA processes remain with `nvidia-smi`;
3. stop the persistent MPS daemon after training sessions end;
4. preview and apply the result download from WSL/local;
5. inspect the downloaded files; and
6. stop or destroy the instance in the provider console or API.

CelebA, uv caches, environments, checkpoints, and generated images also use
provider storage. Account for persistent-volume charges separately from GPU
runtime charges.

## Troubleshooting

- **`nvidia-smi` is unavailable or the driver is too old:** select a suitable
  provider image or update the host outside these scripts. They never install
  an NVIDIA driver.
- **CUDA or BF16 preflight fails:** verify the visible GPU, driver, locked
  PyTorch build, and any container GPU passthrough.
- **CelebA is reported missing:** run the CelebA dataset command from the
  remote project root and wait for its validation to finish.
- **Provisioning cannot pull the checkout:** inspect the remote worktree and
  branch. Provisioning only permits a fast-forward pull and does not discard
  changes.
- **GAN tmux sessions are already active:** inspect them with `status` and
  attach or stop them deliberately before starting another training set.
- **A resume checkpoint is missing:** confirm
  `output-vast-dl/<model>/checkpoints/latest.pth` for every selected model.
- **MPS is unavailable:** omit `--mps`; the instance image may not expose
  `nvidia-cuda-mps-control`.
- **A benchmark output directory already exists:** choose a new directory or
  allow the script to create its UTC-timestamped default. Existing results are
  never reused automatically.
- **Download cannot find remote output:** confirm the SSH alias, remote
  checkout path, and that at least one cloud action created
  `output-vast-dl/`.
