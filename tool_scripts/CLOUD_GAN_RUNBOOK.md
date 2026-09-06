# RTX 5090 CelebA GAN runbook

This workflow runs the repository's 64x64 SN-GAN, SAGAN, and BigGAN lessons
on an existing cloud RTX 5090. See the exact current interface with:

```bash
bash tool_scripts/cloud_gan.sh --help
```

## Safety and storage

The scripts never create, stop, destroy, or resize provider resources. Billing
continues until you act in the provider console. Provisioning can install
Ubuntu packages, clone or fast-forward the checkout, and synchronize the
locked uv environment. It never installs an NVIDIA driver or CUDA Toolkit.

Cloud provisioning only prepares data when `--download-celeba` is supplied.
That option installs the locked project-local `celeba` extra and invokes the
existing dataset downloader. The downloader's separate CelebA option is
still available for other experiments. Reserve space for aligned CelebA,
project caches, and optimizer checkpoints.

| Location | Default |
|---|---|
| Remote checkout | `/workspace/deep-learning-with-pytorch` |
| Prepared data | `data/celeba/img_align_celeba/` |
| Cloud artifacts | `output-vast-dl/<model>/` |
| Managed tmux server | `celeba-gan-cloud` |

Cloud output and data are ignored by Git. Result download merges files without
deleting local-only content, but a differing same-path local file can be
replaced when `--apply` is used.

## Prerequisites

The local host needs `ssh`, `rsync`, and an SSH config alias such as
`vast-dl`. Keep addresses and private-key paths out of tracked files:

```sshconfig
Host vast-dl
    HostName <instance-address>
    User <remote-user>
    Port <ssh-port>
    IdentityFile ~/.ssh/<private-key>
```

The cloud host needs Ubuntu x86-64, one visible RTX 5090, a compatible NVIDIA
driver, writable `/workspace`, and enough persistent storage. Verify it first:

```bash
ssh vast-dl 'uname -srm && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
```

## 1. Prepare the host

Provision the environment without downloading CelebA:

```bash
bash tool_scripts/cloud_gan.sh provision
```

This clones the public repository revision, so publish or otherwise place the
required revision on the remote host first. An existing checkout is updated
with `git pull --ff-only`; local remote-host changes are never discarded.

Opt into downloading aligned CelebA when needed:

```bash
bash tool_scripts/cloud_gan.sh provision --download-celeba
```

For an already provisioned checkout, prepare the data separately with:

```bash
uv sync --locked --no-dev --extra celeba
uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset celeba
```

The loader uses the official training split and the Smiling attribute. It
center-crops aligned faces to 178x178 and resizes them to 64x64 during loading.
Validation references come from the separate official validation split.

## 2. Validate the full models

The smoke action performs short, sequential forward/backward checks with
synthetic 64x64 batches. It uses the real 64-channel model definitions but a
batch of one, and it exercises each model's update ratio, BigGAN orthogonal
regularization, and EMA:

```bash
ssh vast-dl
cd /workspace/deep-learning-with-pytorch
bash tool_scripts/cloud_gan.sh smoke
```

For a longer throughput check with each lesson's RTX 5090 batch default:

```bash
bash tool_scripts/cloud_gan.sh benchmark --steps 20 --warmup-steps 5
```

To include the real data pipeline:

```bash
bash tool_scripts/cloud_gan.sh benchmark \
  --data-dir data/celeba \
  --steps 20
```

Results are written below a new timestamped
`output-vast-dl/benchmarks/rtx5090-*` directory. Each model gets a JSON summary
and a complete log with finite losses, elapsed time, throughput, and peak CUDA
memory. Models run sequentially because concurrent full-width training on one
GPU would confound both memory and throughput.

## 3. Start detached training

Start one model at a time:

```bash
bash tool_scripts/cloud_gan.sh train --model sn_gan
bash tool_scripts/cloud_gan.sh train --model sagan
bash tool_scripts/cloud_gan.sh train --model biggan
```

The command rejects a second managed run while one session is active. All
three lessons default to batch 64, two Smiling labels, and EMA sampling.
SN-GAN and BigGAN use eight epochs; SAGAN uses four. SN-GAN and SAGAN use
one D update per G update; BigGAN uses two.
All use BF16 on the 5090. Lower the batch size if memory is constrained:

```bash
bash tool_scripts/cloud_gan.sh train \
  --model sn_gan \
  --batch-size 8 \
  --generator-batch-size 16
```

Resume from the model's `checkpoints/latest.pth`:

```bash
bash tool_scripts/cloud_gan.sh train --model biggan --resume
```

Repeat the original batch, precision, and data-path overrides when resuming.
Checkpoint compatibility validates the model and optimization settings;
continue using the same dataset. A resumed run may extend the epoch budget. Fresh runs
reset their model output directory and per-model log directory; resuming
preserves both.

## 4. Monitor or stop the managed run

```bash
bash tool_scripts/cloud_gan.sh status
bash tool_scripts/cloud_gan.sh attach biggan
```

Detach without stopping training with `Ctrl-b`, then `d`. Stop a managed run
only when intended:

```bash
bash tool_scripts/cloud_gan.sh stop biggan
```

Stopping the tmux session interrupts the process but does not remove its latest
completed epoch checkpoint and does not change provider billing.

## 5. Download results

From the local host, preview the merge:

```bash
bash tool_scripts/cloud_gan.sh download
```

After inspecting the rsync plan, copy the files:

```bash
bash tool_scripts/cloud_gan.sh download --apply
```

Use `--host <ssh-alias>` for a non-default SSH config host. Confirm that all
required weights, checkpoints, sample grids, plots, JSON benchmark records,
and logs arrived before releasing the cloud instance.

If CelebA is also prepared locally, render the shared reference comparison
against the downloaded output tree with:

```bash
uv run --locked python \
  genai/1.0_generative_adversarial_network/6.3_sn_sagan_biggan_evaluation.py \
  --data-dir data/celeba \
  --output-root output-vast-dl
```

## Troubleshooting

- **RTX 5090 preflight fails:** check the selected instance, container GPU
  passthrough, driver, and locked PyTorch CUDA runtime. The scripts do not
  repair drivers.
- **CelebA readiness fails:** check the aligned JPEG directory,
  `list_eval_partition.csv`, and `list_attr_celeba.csv` below `data/celeba`,
  then rerun the explicit preparation command.
- **Dataset download fails:** check the locked `celeba` extra and the error
  reported by the downloader, then rerun the explicit command.
- **CUDA out of memory:** lower `--batch-size` (and SN-GAN's
  `--generator-batch-size`) while retaining the 64-channel architecture.
- **A managed session already exists:** use `status`, then attach or explicitly
  stop it before starting another model.
- **Resume metadata differs:** repeat the original CLI overrides and use the
  checkpoint from the matching model directory.
- **Download cannot find output:** verify the SSH alias, remote checkout path,
  and that a smoke, benchmark, or training action created `output-vast-dl/`.
