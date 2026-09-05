# RTX 5090 Imagenette GAN runbook

This workflow runs the repository's 128x128 SN-GAN, SAGAN, and BigGAN lessons
on an existing cloud RTX 5090. See the exact current interface with:

```bash
bash tool_scripts/cloud_gan.sh --help
```

## Safety and storage

The scripts never create, stop, destroy, or resize provider resources. Billing
continues until you act in the provider console. Provisioning can install
Ubuntu packages, clone or fast-forward the checkout, and synchronize the
locked uv environment. It never installs an NVIDIA driver or CUDA Toolkit.

The general dataset downloader includes Imagenette in its no-argument and
`--dataset all` sequence. Cloud provisioning does not invoke that default:
`cloud_gan.sh provision` requires `--download-imagenette` before it downloads
and prepares the data. No account or dataset license prompt is involved.
Reserve space for the archive, extracted Imagenette-320 source, derived
Imagenette-128 cache, uv caches, and optimizer checkpoints.

| Location | Default |
|---|---|
| Remote checkout | `/workspace/deep-learning-with-pytorch` |
| Prepared data | `data/imagenette-128/train/<WNID>/*.JPEG` |
| Cloud artifacts | `output-vast-dl/<model>/` |
| Managed tmux server | `imagenette-gan-cloud` |

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

Provision the environment without downloading Imagenette:

```bash
bash tool_scripts/cloud_gan.sh provision
```

This clones the public repository revision, so publish or otherwise place the
required revision on the remote host first. An existing checkout is updated
with `git pull --ff-only`; local remote-host changes are never discarded.

Opt into the direct Imagenette download and preprocessing when needed:

```bash
bash tool_scripts/cloud_gan.sh provision --download-imagenette
```

For an already provisioned checkout, prepare the data separately with:

```bash
uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset imagenette
```

Archive preparation builds the 10-class source tree, then creates the filtered
Imagenette-128 cache. Download, extraction, and resizing reuse completed files
after interruption. The cache records images excluded by the configured
source-size limits.

## 2. Validate the full models

The smoke action performs short, sequential forward/backward checks with
synthetic 128x128 batches. It uses the real 64-channel model definitions but a
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
  --data-dir data/imagenette-128 \
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

The command rejects a second managed run while one session is active. Defaults
are paper-oriented where practical: SN-GAN uses D/G batches 16/32, SAGAN uses
batch 16, and BigGAN uses batch 8. All use BF16 automatically on the 5090.
Reduce only the runtime batch if memory is constrained:

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

If the original run used epoch, batch, precision, or data-path overrides,
repeat them on resume because checkpoint compatibility validates the complete
run configuration. SN-GAN also validates that the epoch plan ends on a full
five-to-one update cycle.

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

If Imagenette is also prepared locally, render the shared reference comparison
against the downloaded output tree with:

```bash
uv run --locked python \
  genai/1.0_generative_adversarial_network/6.3_sn_sagan_biggan_evaluation.py \
  --data-dir data/imagenette-128 \
  --output-root output-vast-dl
```

## Troubleshooting

- **RTX 5090 preflight fails:** check the selected instance, container GPU
  passthrough, driver, and locked PyTorch CUDA runtime. The scripts do not
  repair drivers.
- **Imagenette readiness fails:** inspect the completion metadata and 10 WNID
  directories below `data/imagenette-128/train`, then rerun the explicit
  preparation command.
- **Dataset download fails:** rerun the same command; partial archive downloads
  and completed preprocessing files are reused.
- **CUDA out of memory:** lower `--batch-size` (and SN-GAN's
  `--generator-batch-size`) while retaining the 64-channel architecture.
- **A managed session already exists:** use `status`, then attach or explicitly
  stop it before starting another model.
- **Resume metadata differs:** repeat the original CLI overrides and use the
  checkpoint from the matching model directory.
- **Download cannot find output:** verify the SSH alias, remote checkout path,
  and that a smoke, benchmark, or training action created `output-vast-dl/`.
