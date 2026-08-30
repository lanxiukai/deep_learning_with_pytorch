# Cloud GAN runbook

Use this runbook for the operational sequence, safety boundaries, and result
interpretation of the repository's cloud GAN workflow. For the current action
and option list, run:

```bash
bash tool_scripts/cloud_gan.sh --help
```

The delegated setup and benchmark scripts also expose focused `--help` output.

## Safety model

The cloud scripts operate on an instance that you already created or rented.
They do not create, stop, destroy, or otherwise control provider resources, so
GPU and storage billing continues until you stop or destroy the instance with
the provider.

The workflow spans two machines:

| Machine | Responsibility |
|---|---|
| WSL or local host | Owns SSH configuration, starts provisioning, and downloads results |
| Cloud GPU host | Owns the project environment, CelebA data, training processes, checkpoints, and benchmark output |

The default remote checkout is `/workspace/deep-learning-with-pytorch`.
Cloud artifacts are written below `output-vast-dl/`, separately from the local
lesson default `output/`. Both `data/` and `output-vast-dl/` are ignored by Git.

Provisioning has persistent remote effects: it installs missing host
prerequisites, creates `/workspace` when needed, clones or fast-forward-updates
the repository, synchronizes a locked uv environment, and prepares CelebA.
It requires root access or passwordless, non-interactive `sudo` for missing
system packages. It never installs or updates the NVIDIA driver or CUDA Toolkit.

Result download is a preview by default. Applying a download merges the remote
output tree into the local tree without deleting local-only files, but a remote
file can replace a differing local file at the same relative path.

## Prerequisites

The WSL or local host needs:

- `ssh` and `rsync`;
- an SSH config host, `vast-dl` by default; and
- enough disk space for checkpoints, samples, logs, and benchmark output.

The cloud host needs:

- Ubuntu Linux on x86-64;
- a GPU and driver accepted by `cloud_gan.sh` preflight;
- CUDA BF16 support in the selected GPU and locked PyTorch runtime;
- a writable `/workspace`; and
- enough space for the uv cache, environment, CelebA, and generated artifacts.

The orchestration expects one visible GPU. Isolate the intended GPU at the
environment or container level on a multi-GPU host. Focused benchmark scripts
also accept `--gpu-index` where applicable.

Keep provider addresses, users, ports, and private-key paths in
`~/.ssh/config`, never in tracked repository files:

```sshconfig
Host vast-dl
    HostName <instance-address>
    User <remote-user>
    Port <ssh-port>
    IdentityFile ~/.ssh/<private-key>
```

Verify the connection and GPU before provisioning:

```bash
ssh vast-dl 'uname -srm && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'
```

## Standard workflow

### 1. Provision the existing host

Run from WSL or the local host:

```bash
bash tool_scripts/cloud_gan.sh provision
```

Pass `--host` for a different SSH config alias. Provisioning clones from the
public GitHub remote; it does not upload the local worktree. An existing remote
checkout must accept a fast-forward-only pull, so publish the required revision
or prepare the intended checkout on the host before continuing.

For a manually prepared remote checkout, run the setup and dataset commands
from that checkout instead:

```bash
bash tool_scripts/setup_cloud_gpu.sh --profile celeba
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset celeba
```

### 2. Validate before full training

Connect to the remote checkout and run the smoke checks plus concurrency test:

```bash
ssh vast-dl
cd /workspace/deep-learning-with-pytorch
bash tool_scripts/cloud_gan.sh validate --mps
```

Use `smoke` when only the isolated ProGAN, StyleGAN, and StyleGAN2 checks are
needed. Use `benchmark` when only the sequential-versus-concurrent comparison
is needed. Do not start a full run until smoke metrics are finite and the
generated output is plausible.

### 3. Start or resume detached training

Start selected lessons in dedicated tmux sessions:

```bash
bash tool_scripts/cloud_gan.sh train --models progan,stylegan --mps
```

Without `--mps`, processes share the GPU through the normal CUDA scheduler.
With it, the script starts or reuses a persistent NVIDIA MPS daemon below
`/tmp/dl-gan-mps`.

Training uses a private tmux server named `gan-cloud` and one session per
model. A second training command is rejected while those sessions remain
active, preventing accidental duplicate runs.

Resume every selected model from its latest checkpoint with:

```bash
bash tool_scripts/cloud_gan.sh train \
  --models progan,stylegan \
  --mps \
  --resume
```

Every selected model must already have
`output-vast-dl/<model>/checkpoints/latest.pth`; resume is not a best-effort
partial operation.

### 4. Monitor and control training

```bash
bash tool_scripts/cloud_gan.sh status
bash tool_scripts/cloud_gan.sh attach stylegan
```

Detach from tmux without stopping training by pressing `Ctrl-b`, then `d`.
To interrupt one model deliberately, attach and press `Ctrl-c`, or target its
private session explicitly:

```bash
tmux -L gan-cloud kill-session -t gan-stylegan
```

After all MPS-backed sessions finish, stop the persistent daemon:

```bash
bash tool_scripts/cloud_gan.sh stop-mps
```

The command refuses to stop MPS while managed sessions remain active. The
concurrency benchmark uses a separate temporary daemon and cleans it up when
the benchmark exits.

### 5. Download results and release the instance

Return to WSL or the local host and preview the merge:

```bash
bash tool_scripts/cloud_gan.sh download
```

After reviewing the rsync plan, apply it:

```bash
bash tool_scripts/cloud_gan.sh download --apply
```

Keep unrelated runs in distinct output directories to avoid same-path
replacement. Verify all required checkpoints, samples, summaries, and logs
locally before stopping or destroying the provider instance. Neither download
nor `stop-mps` changes provider billing state.

## Environment profiles

`setup_cloud_gpu.sh` synchronizes from `.python-version`, `pyproject.toml`, and
`uv.lock`:

| Profile | Installed project dependencies |
|---|---|
| `core` | Base dependencies, including PyTorch and torchvision |
| `celeba` | Core plus the CelebA/Kaggle dependency, without development tools |
| `examples` | Core plus optional lesson dependencies |
| `full` | All optional dependencies and development/test tools |

Use the setup script's `--help` for dry-run, system-package, GPU-check, and
persistent-cache controls. Dataset profiles and preparation commands are in
[README.md](README.md#dataset-profiles).

## Benchmarks

All GAN benchmarks require prepared CelebA data and a compatible CUDA runtime.
Their output directory must be new, protecting previous measurements from
accidental reuse.

| Question | Entry point | Interpret with |
|---|---|---|
| Do concurrent lesson runs improve aggregate completion time? | `cloud_gan.sh benchmark` or `test_gan_concurrency.sh` | Aggregate elapsed time, per-model throughput, peak memory, GPU samples, and finite metrics |
| How do strict FP32, TF32, and BF16 compare on one H100? | `cloud_gan.sh precision` | `bf16_vs_tf32_speedup` for the practical precision comparison |
| How do two single-GPU runs compare? | `benchmark_stylegan_gpu.sh` | Matching workload metadata and the generated `training.tsv` files |

For a single-GPU comparison, run the same command on each GPU and pass the
baseline result first:

```bash
bash tool_scripts/benchmark_stylegan_gpu.sh --compare \
  <baseline>/training.tsv \
  <candidate>/training.tsv
```

Use `--compare-precision` only when hardware and workload are identical but
precision modes differ. Start with a short run before committing to a sustained
benchmark. Exact workloads, defaults, and output options belong to each
benchmark script's `--help` output.

## Cleanup checklist

Before leaving a cloud host:

1. Check `cloud_gan.sh status` for managed training sessions.
2. Inspect `nvidia-smi` for any other CUDA processes.
3. Stop the persistent MPS daemon after managed sessions finish.
4. Preview, apply, and inspect the result download locally.
5. Stop or destroy the instance with the provider.

Account for persistent-volume charges from datasets, caches, environments,
checkpoints, and samples separately from GPU runtime charges.

## Troubleshooting

- **GPU or driver preflight fails:** select a compatible provider image and
  verify container GPU passthrough. These scripts do not repair the driver.
- **CelebA is missing:** run the CelebA dataset command from the remote project
  root and wait for validation to complete.
- **Provisioning cannot update the checkout:** inspect remote changes and
  branch divergence; provisioning never discards them.
- **Managed tmux sessions already exist:** inspect them with `status`, then
  attach or stop them deliberately before starting another run.
- **Resume reports a missing checkpoint:** verify `latest.pth` for every model
  selected by the command.
- **MPS is unavailable:** omit `--mps`; the provider image may not expose
  `nvidia-cuda-mps-control`.
- **A benchmark directory already exists:** choose a new path or use the
  timestamped default; existing results are never reused automatically.
- **Download cannot find output:** confirm the SSH alias, remote checkout path,
  and that a cloud action created `output-vast-dl/`.
