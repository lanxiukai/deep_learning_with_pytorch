#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly TMUX_SOCKET="celeba-gan-cloud"
readonly REMOTE_PROJECT_ROOT="/workspace/deep-learning-with-pytorch"
readonly OUTPUT_ROOT="$PROJECT_ROOT/output-vast-dl"
readonly REMOTE_OUTPUT_ROOT="$REMOTE_PROJECT_ROOT/output-vast-dl"
readonly DEFAULT_DATA_DIR="$PROJECT_ROOT/data/celeba"

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/cloud_gan.sh ACTION [options]

Operate the CelebA-64 SN-GAN, SAGAN, and BigGAN lessons on one cloud RTX
5090. The script never creates, stops, or destroys provider instances.

Actions:
  provision [OPTIONS]          Prepare an existing remote RTX 5090 host
  smoke                       Run short synthetic optimization checks for all models
  benchmark [OPTIONS]         Run benchmark_rtx5090_gans.sh with OPTIONS
  train --model MODEL [...]   Start one detached full training run
  status                      Show the managed session, GPU, and recent logs
  attach MODEL                Attach to a model's tmux session
  stop MODEL                  Stop a model's managed tmux session
  download [OPTIONS]          Preview or apply a WSL-side result merge
  -h, --help                  Show this help

Provision options:
  --host HOST                 SSH config host (default: vast-dl)
  --download-celeba           Download aligned CelebA remotely

Train options:
  --model MODEL               sn_gan, sagan, or biggan
  --resume                    Resume output-vast-dl/MODEL/checkpoints/latest.pth
  --epochs N                  Override the lesson's epoch budget
  --batch-size N              Override the lesson's discriminator/shared batch
  --generator-batch-size N    Override SN-GAN's generator batch only
  --num-workers N             DataLoader workers (default: 8)
  --precision MODE            bf16 (default) or fp32
  --data-dir PATH             Aligned CelebA root (default: data/celeba)

Download options:
  --host HOST                 SSH config host (default: vast-dl)
  --apply                     Copy files; otherwise show an rsync preview
EOF
}

log() {
    printf '[cloud-celeba-gan] %s\n' "$*"
}

die() {
    printf '[cloud-celeba-gan] ERROR: %s\n' "$*" >&2
    exit 1
}

validate_host() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
        die "--host must be one SSH config host name"
}

validate_model() {
    case "$1" in
        sn_gan|sagan|biggan) ;;
        *) die "model must be sn_gan, sagan, or biggan" ;;
    esac
}

lesson_path() {
    case "$1" in
        sn_gan)
            printf '%s\n' \
                'genai/1.0_generative_adversarial_network/6.0_sn_gan.py'
            ;;
        sagan)
            printf '%s\n' \
                'genai/1.0_generative_adversarial_network/6.1_sagan.py'
            ;;
        biggan)
            printf '%s\n' \
                'genai/1.0_generative_adversarial_network/6.2_biggan.py'
            ;;
    esac
}

require_no_args() {
    local action="$1"
    shift
    (($# == 0)) || die "$action does not accept arguments"
}

cloud_preflight() {
    local require_data="$1"
    local data_dir="${2:-}"
    local gpu_name
    command -v uv >/dev/null 2>&1 || \
        die "uv is unavailable; run setup_cloud_gpu.sh"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    [[ "${gpu_name^^}" == *"RTX 5090"* ]] || \
        die "expected RTX 5090, found: $gpu_name"
    cd "$PROJECT_ROOT"
    uv run --locked --no-sync python -c \
        'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"'
    if [[ "$require_data" == true ]]; then
        CELEBA_ROOT="$data_dir" uv run --locked --no-sync python -c \
            'import os; from dl_utils.data.celeba import CelebAAlignedDataset; dataset=CelebAAlignedDataset(os.environ["CELEBA_ROOT"], attribute="Smiling"); print(f"CelebA training images: {len(dataset):,}")'
    fi
    log "preflight passed: $gpu_name"
}

provision_cloud() {
    local host="vast-dl"
    local download_celeba=false
    while (($# > 0)); do
        case "$1" in
            --host)
                (($# >= 2)) || die "--host requires a value"
                host="$2"
                shift 2
                ;;
            --download-celeba)
                download_celeba=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *) die "unknown provision option: $1" ;;
        esac
    done
    validate_host "$host"
    command -v ssh >/dev/null 2>&1 || die "ssh is unavailable"
    log "preparing $host:$REMOTE_PROJECT_ROOT"
    ssh "$host" bash -se -- "$download_celeba" <<'REMOTE'
set -Eeuo pipefail
download_celeba="$1"
project_root=/workspace/deep-learning-with-pytorch
if ((EUID == 0)); then
    privilege=()
else
    command -v sudo >/dev/null 2>&1 || {
        printf 'sudo is required for cloud provisioning\n' >&2
        exit 1
    }
    privilege=(sudo)
fi
"${privilege[@]}" apt-get update
"${privilege[@]}" env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y ca-certificates curl git rsync tmux
if [[ ! -d /workspace ]]; then
    "${privilege[@]}" mkdir -p /workspace
    if ((EUID != 0)); then
        "${privilege[@]}" chown "$(id -un):$(id -gn)" /workspace
    fi
fi
[[ -w /workspace ]] || {
    printf '/workspace is not writable by %s\n' "$(id -un)" >&2
    exit 1
}
if [[ ! -d "$project_root/.git" ]]; then
    [[ ! -e "$project_root" ]] || {
        printf '%s exists but is not a Git worktree\n' "$project_root" >&2
        exit 1
    }
    git clone --depth 1 \
        https://github.com/lanxiukai/deep-learning-with-pytorch.git \
        "$project_root"
else
    git -C "$project_root" pull --ff-only
fi
cd "$project_root"
bash tool_scripts/setup_cloud_gpu.sh --profile core
if [[ "$download_celeba" == true ]]; then
    uv sync --locked --no-dev --extra celeba
    uv run --locked --no-sync python tool_scripts/download_dataset.py \
        --dataset celeba
fi
REMOTE
    if [[ "$download_celeba" == true ]]; then
        log "host setup and CelebA-64 preparation completed"
    else
        log "host setup completed; CelebA was not downloaded"
    fi
}

run_smoke() {
    require_no_args smoke "$@"
    cloud_preflight false
    bash "$SCRIPT_DIR/benchmark_rtx5090_gans.sh" \
        --steps 5 \
        --warmup-steps 0 \
        --batch-size 1
}

run_benchmark() {
    cloud_preflight false
    exec bash "$SCRIPT_DIR/benchmark_rtx5090_gans.sh" "$@"
}

tmux_server_exists() {
    tmux -L "$TMUX_SOCKET" list-sessions >/dev/null 2>&1
}

train_worker() {
    local model="$1"
    local resume="$2"
    local epochs="$3"
    local batch_size="$4"
    local generator_batch_size="$5"
    local num_workers="$6"
    local precision="$7"
    local data_dir="$8"
    local script checkpoint log_file
    local -a command
    validate_model "$model"
    script="$(lesson_path "$model")"
    checkpoint="$OUTPUT_ROOT/$model/checkpoints/latest.pth"
    log_file="$OUTPUT_ROOT/logs/$model/train.log"
    command=(
        uv run --locked --no-sync python -u "$script"
        --data-dir "$data_dir"
        --output-root "$OUTPUT_ROOT"
        --num-workers "$num_workers"
        --precision "$precision"
    )
    [[ -z "$epochs" ]] || command+=(--epochs "$epochs")
    [[ -z "$batch_size" ]] || command+=(--batch-size "$batch_size")
    [[ -z "$generator_batch_size" ]] || \
        command+=(--generator-batch-size "$generator_batch_size")
    [[ "$resume" == false ]] || command+=(--resume-from "$checkpoint")
    cd "$PROJECT_ROOT"
    GAN_LOG_DIR="$OUTPUT_ROOT/logs/$model" GAN_RESUME="$resume" \
        uv run --locked --no-sync python -c \
        'import os; from pathlib import Path; from dl_utils.filesystem.directories import reset_dir; path=Path(os.environ["GAN_LOG_DIR"]); reset_dir(str(path)) if os.environ["GAN_RESUME"] == "false" or not path.exists() else None'
    {
        printf '\n=== %s model=%s resume=%s ===\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" "$resume"
        "${command[@]}"
    } 2>&1 | tee -a "$log_file"
}

run_train() {
    local model=""
    local resume=false
    local epochs=""
    local batch_size=""
    local generator_batch_size=""
    local num_workers=8
    local precision=bf16
    local data_dir="$DEFAULT_DATA_DIR"
    local session worker_command checkpoint
    while (($# > 0)); do
        case "$1" in
            --model)
                (($# >= 2)) || die "--model requires a value"
                model="$2"
                shift 2
                ;;
            --resume)
                resume=true
                shift
                ;;
            --epochs)
                (($# >= 2)) || die "--epochs requires a value"
                epochs="$2"
                shift 2
                ;;
            --batch-size)
                (($# >= 2)) || die "--batch-size requires a value"
                batch_size="$2"
                shift 2
                ;;
            --generator-batch-size)
                (($# >= 2)) || die "--generator-batch-size requires a value"
                generator_batch_size="$2"
                shift 2
                ;;
            --num-workers)
                (($# >= 2)) || die "--num-workers requires a value"
                num_workers="$2"
                shift 2
                ;;
            --precision)
                (($# >= 2)) || die "--precision requires a value"
                precision="$2"
                shift 2
                ;;
            --data-dir)
                (($# >= 2)) || die "--data-dir requires a value"
                data_dir="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *) die "unknown train option: $1" ;;
        esac
    done
    [[ -n "$model" ]] || die "train requires --model"
    validate_model "$model"
    [[ "$num_workers" =~ ^[0-9]+$ ]] || \
        die "--num-workers must be non-negative"
    for value in "$epochs" "$batch_size" "$generator_batch_size"; do
        [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || \
            die "epoch and batch overrides must be positive"
    done
    case "$precision" in
        bf16|fp32) ;;
        *) die "--precision must be bf16 or fp32" ;;
    esac
    if [[ -n "$generator_batch_size" && "$model" != sn_gan ]]; then
        die "--generator-batch-size applies only to sn_gan"
    fi

    cloud_preflight true "$data_dir"
    command -v tmux >/dev/null 2>&1 || die "tmux is unavailable"
    tmux_server_exists && \
        die "a managed GAN session is active; inspect it with status"
    checkpoint="$OUTPUT_ROOT/$model/checkpoints/latest.pth"
    if [[ "$resume" == true && ! -f "$checkpoint" ]]; then
        die "resume checkpoint is missing: $checkpoint"
    fi

    session="gan-$model"
    printf -v worker_command \
        'bash %q __train_worker %q %q %q %q %q %q %q %q' \
        "$SCRIPT_PATH" "$model" "$resume" "$epochs" "$batch_size" \
        "$generator_batch_size" "$num_workers" "$precision" "$data_dir"
    tmux -L "$TMUX_SOCKET" new-session -d \
        -s "$session" -c "$PROJECT_ROOT" "$worker_command"
    log "started $model in tmux session $session"
    log "monitor with: bash tool_scripts/cloud_gan.sh status"
}

run_status() {
    require_no_args status "$@"
    if tmux_server_exists; then
        tmux -L "$TMUX_SOCKET" list-sessions
    else
        log "no managed tmux session is active"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
    fi
    if [[ -d "$OUTPUT_ROOT/logs" ]]; then
        for log_file in "$OUTPUT_ROOT"/logs/*/*.log; do
            [[ -f "$log_file" ]] || continue
            printf '\n==> %s <==\n' "$log_file"
            tail -n 12 "$log_file"
        done
    fi
}

run_attach() {
    (($# == 1)) || die "attach requires one model"
    validate_model "$1"
    exec tmux -L "$TMUX_SOCKET" attach-session -t "gan-$1"
}

run_stop() {
    (($# == 1)) || die "stop requires one model"
    validate_model "$1"
    tmux -L "$TMUX_SOCKET" has-session -t "gan-$1" 2>/dev/null || \
        die "managed session not found: gan-$1"
    tmux -L "$TMUX_SOCKET" kill-session -t "gan-$1"
    log "stopped managed session gan-$1"
}

run_download() {
    local host="vast-dl"
    local apply=false
    local -a rsync_args
    while (($# > 0)); do
        case "$1" in
            --host)
                (($# >= 2)) || die "--host requires a value"
                host="$2"
                shift 2
                ;;
            --apply)
                apply=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *) die "unknown download option: $1" ;;
        esac
    done
    validate_host "$host"
    command -v ssh >/dev/null 2>&1 || die "ssh is unavailable"
    command -v rsync >/dev/null 2>&1 || die "rsync is unavailable"
    ssh "$host" test -d "$REMOTE_OUTPUT_ROOT" || \
        die "remote output directory is missing: $REMOTE_OUTPUT_ROOT"
    mkdir -p "$OUTPUT_ROOT"
    rsync_args=(-avh --itemize-changes)
    if [[ "$apply" == false ]]; then
        rsync_args+=(--dry-run)
        log "previewing result merge; pass --apply to copy files"
    fi
    rsync "${rsync_args[@]}" \
        "$host:$REMOTE_OUTPUT_ROOT/" "$OUTPUT_ROOT/"
}

action="${1:--help}"
if (($# > 0)); then
    shift
fi
case "$action" in
    provision) provision_cloud "$@" ;;
    smoke) run_smoke "$@" ;;
    benchmark) run_benchmark "$@" ;;
    train) run_train "$@" ;;
    status) run_status "$@" ;;
    attach) run_attach "$@" ;;
    stop) run_stop "$@" ;;
    download) run_download "$@" ;;
    __train_worker) train_worker "$@" ;;
    -h|--help|help) usage ;;
    *) die "unknown action: $action" ;;
esac
