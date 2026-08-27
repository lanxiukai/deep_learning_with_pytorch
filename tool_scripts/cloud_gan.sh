#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
readonly TMUX_SOCKET="gan-cloud"
readonly MPS_ROOT="/tmp/dl-gan-mps"
readonly REMOTE_PROJECT_ROOT="/workspace/deep-learning-with-pytorch"
readonly OUTPUT_ROOT="$PROJECT_ROOT/output-vast-dl"
readonly REMOTE_OUTPUT_ROOT="$REMOTE_PROJECT_ROOT/output-vast-dl"

selected_models=()
CLOUD_GPU_NAME=""

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/cloud_gan.sh ACTION [options]

Run the short validation, concurrency benchmark, detached training, monitoring,
or result download steps on B200, B300, RTX 5080, or RTX 5090.

Actions:
  provision [OPTIONS]            Prepare the host and download CelebA there
  validate [BENCHMARK OPTIONS]   Run smoke checks, then the concurrency test
  smoke                         Run all three isolated short training checks
  benchmark [OPTIONS]           Run test_gan_concurrency.sh with OPTIONS
  train --models LIST [OPTIONS] Start selected full runs in dedicated tmux sessions
  status                        Show training sessions, GPU state, and recent logs
  attach MODEL                  Attach to one model's tmux session
  stop-mps                      Stop the persistent MPS daemon after training
  download [OPTIONS]            Preview or apply a WSL-side output merge
  -h, --help                    Show this help

Train options:
  --models LIST                 Comma-separated progan, stylegan, stylegan2
  --mps                         Start and use a persistent MPS daemon
  --resume                      Resume every selected model from latest.pth
  --num-workers N               DataLoader workers per process (default: 0)
  --data-pipeline PIPELINE      cuda (default), auto, or cpu

Download options:
  --host HOST                   SSH config host (default: vast-dl)
  --apply                       Copy files; without this flag, only preview

Provision options:
  --host HOST                   SSH config host (default: vast-dl)

Examples:
  bash tool_scripts/cloud_gan.sh provision
  bash tool_scripts/cloud_gan.sh validate --mps
  bash tool_scripts/cloud_gan.sh smoke
  bash tool_scripts/cloud_gan.sh benchmark --mps
  bash tool_scripts/cloud_gan.sh train --models progan,stylegan --mps
  bash tool_scripts/cloud_gan.sh status
  bash tool_scripts/cloud_gan.sh attach stylegan
  bash tool_scripts/cloud_gan.sh download --apply
  bash tool_scripts/cloud_gan.sh download --host vast-dl-direct --apply
EOF
}

log() {
    printf '[cloud-gan] %s\n' "$*"
}

die() {
    printf '[cloud-gan] ERROR: %s\n' "$*" >&2
    exit 1
}

validate_host() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
        die "--host must be one SSH config host name"
}

require_no_args() {
    local action="$1"
    shift
    (($# == 0)) || die "$action does not accept arguments"
}

validate_model() {
    case "$1" in
        progan|stylegan|stylegan2) ;;
        *) die "unknown model: $1" ;;
    esac
}

is_supported_gpu_name() {
    local gpu_name_upper="${1^^}"
    case "$gpu_name_upper" in
        *B200*|*B300*|*RTX*5080*|*RTX*5090*) return 0 ;;
        *) return 1 ;;
    esac
}

is_rtx_blackwell_gpu_name() {
    local gpu_name_upper="${1^^}"
    case "$gpu_name_upper" in
        *RTX*5080*|*RTX*5090*) return 0 ;;
        *) return 1 ;;
    esac
}

parse_models() {
    local models_csv="$1"
    local model existing
    local requested_models=()
    IFS=',' read -r -a requested_models <<< "$models_csv"
    selected_models=()
    for model in "${requested_models[@]}"; do
        validate_model "$model"
        for existing in "${selected_models[@]}"; do
            [[ "$model" != "$existing" ]] || die "duplicate model: $model"
        done
        selected_models+=("$model")
    done
    ((${#selected_models[@]} >= 1 && ${#selected_models[@]} <= 3)) || \
        die "--models must select one to three models"
}

lesson_path() {
    case "$1" in
        progan) printf '%s\n' 'genai/1.0_generative_adversarial_network/7.0_progan.py' ;;
        stylegan) printf '%s\n' 'genai/1.0_generative_adversarial_network/7.1_stylegan.py' ;;
        stylegan2) printf '%s\n' 'genai/1.0_generative_adversarial_network/7.2_stylegan2.py' ;;
    esac
}

cloud_preflight() {
    command -v uv >/dev/null 2>&1 || die "uv is unavailable; run setup_cloud_gpu.sh"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    [[ -f "$PROJECT_ROOT/data/celeba/list_eval_partition.csv" ]] || \
        die "prepared CelebA data is missing under data/celeba"
    CLOUD_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" || \
        die "cannot query the GPU"
    if ! is_supported_gpu_name "$CLOUD_GPU_NAME"; then
        die "GPU is '$CLOUD_GPU_NAME'; expected B200, B300, RTX 5080, or RTX 5090"
    fi
    cd "$PROJECT_ROOT"
    uv run --locked --no-sync python -c \
        'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"'
}

provision_cloud() {
    local host="vast-dl"
    while (($# > 0)); do
        case "$1" in
            --host)
                (($# >= 2)) || die "--host requires a value"
                host="$2"
                shift 2
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
    ssh "$host" bash -se <<'REMOTE'
set -Eeuo pipefail

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
    apt-get install -y ca-certificates curl git
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
bash tool_scripts/setup_cloud_gpu.sh --profile celeba
uv run --locked --no-sync python tool_scripts/download_dataset.py \
    --dataset celeba
REMOTE
    log "provisioning and direct CelebA download completed"
}

run_smoke() {
    local model script
    local command=()
    local rtx_blackwell=false
    require_no_args smoke "$@"
    cloud_preflight
    if is_rtx_blackwell_gpu_name "$CLOUD_GPU_NAME"; then
        rtx_blackwell=true
        log "smoke: using the RTX Blackwell regularization profile"
    fi
    for model in progan stylegan stylegan2; do
        script="$(lesson_path "$model")"
        command=(uv run --locked --no-sync python -u "$script")
        case "$model" in
            progan)
                command+=(--phase-kimg 1 --num-workers 0 --data-pipeline cuda)
                if [[ "$rtx_blackwell" == true ]]; then
                    command+=(--d-reg-every 2 --reg-batch-shrink 8)
                fi
                ;;
            stylegan)
                command+=(--phase-kimg 1 --num-workers 0 --data-pipeline cuda)
                if [[ "$rtx_blackwell" == true ]]; then
                    command+=(--d-reg-every 1 --reg-batch-shrink 64)
                fi
                ;;
            stylegan2)
                command+=(--total-kimg 1 --num-workers 0 --data-pipeline cuda)
                if [[ "$rtx_blackwell" == true ]]; then
                    command+=(--r1-batch-shrink 64 --path-batch-shrink 64)
                fi
                ;;
        esac
        log "smoke: starting $model"
        DL_OUTPUT_ROOT="$OUTPUT_ROOT/smoke" "${command[@]}"
    done
    log "smoke checks completed; inspect output-vast-dl/smoke"
}

run_validation() {
    if (($# > 0)) && [[ "$1" == -h || "$1" == --help ]]; then
        exec bash "$SCRIPT_DIR/test_gan_concurrency.sh" --help
    fi
    run_smoke
    log "starting the fixed 128x128 concurrency benchmark"
    bash "$SCRIPT_DIR/test_gan_concurrency.sh" "$@"
}

start_mps() {
    command -v nvidia-cuda-mps-control >/dev/null 2>&1 || \
        die "nvidia-cuda-mps-control is unavailable"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
        log "reusing the MPS daemon at $MPS_ROOT"
    else
        nvidia-cuda-mps-control -d
        log "started the MPS daemon at $MPS_ROOT"
    fi
}

tmux_server_exists() {
    tmux -L "$TMUX_SOCKET" list-sessions >/dev/null 2>&1
}

configure_tmux_environment() {
    local use_mps="$1"
    tmux_server_exists || return 0
    if [[ "$use_mps" == true ]]; then
        tmux -L "$TMUX_SOCKET" set-environment -g \
            CUDA_MPS_PIPE_DIRECTORY "$CUDA_MPS_PIPE_DIRECTORY"
        tmux -L "$TMUX_SOCKET" set-environment -g \
            CUDA_MPS_LOG_DIRECTORY "$CUDA_MPS_LOG_DIRECTORY"
    else
        tmux -L "$TMUX_SOCKET" set-environment -gu \
            CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true
        tmux -L "$TMUX_SOCKET" set-environment -gu \
            CUDA_MPS_LOG_DIRECTORY 2>/dev/null || true
    fi
}

train_worker() {
    local model="$1"
    local num_workers="$2"
    local data_pipeline="$3"
    local resume="$4"
    local script checkpoint log_file
    local command=()
    validate_model "$model"
    script="$(lesson_path "$model")"
    checkpoint="$OUTPUT_ROOT/$model/checkpoints/latest.pth"
    log_file="$OUTPUT_ROOT/logs/$model.log"
    command=(
        uv run --locked --no-sync python -u "$script"
        --num-workers "$num_workers"
        --data-pipeline "$data_pipeline"
    )
    if [[ "$resume" == true ]]; then
        command+=(--resume-from "$checkpoint")
    fi
    mkdir -p "$OUTPUT_ROOT/logs"
    cd "$PROJECT_ROOT"
    {
        printf '\n=== %s model=%s resume=%s ===\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" "$resume"
        DL_OUTPUT_ROOT="$OUTPUT_ROOT" "${command[@]}"
    } 2>&1 | tee -a "$log_file"
}

run_train() {
    local models_csv=""
    local use_mps=false
    local resume=false
    local num_workers=0
    local data_pipeline="cuda"
    local model session checkpoint worker_command
    while (($# > 0)); do
        case "$1" in
            --models)
                (($# >= 2)) || die "--models requires a value"
                models_csv="$2"
                shift 2
                ;;
            --mps)
                use_mps=true
                shift
                ;;
            --resume)
                resume=true
                shift
                ;;
            --num-workers)
                (($# >= 2)) || die "--num-workers requires a value"
                num_workers="$2"
                shift 2
                ;;
            --data-pipeline)
                (($# >= 2)) || die "--data-pipeline requires a value"
                data_pipeline="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *) die "unknown train option: $1" ;;
        esac
    done
    [[ -n "$models_csv" ]] || die "train requires --models"
    [[ "$num_workers" =~ ^[0-9]+$ ]] || \
        die "--num-workers must be a non-negative integer"
    case "$data_pipeline" in
        auto|cuda|cpu) ;;
        *) die "--data-pipeline must be one of: auto, cuda, cpu" ;;
    esac
    parse_models "$models_csv"
    cloud_preflight
    command -v tmux >/dev/null 2>&1 || die "tmux is unavailable"
    tmux_server_exists && \
        die "GAN tmux sessions are already active; inspect them with status"

    for model in "${selected_models[@]}"; do
        if [[ "$resume" == true ]]; then
            checkpoint="$OUTPUT_ROOT/$model/checkpoints/latest.pth"
            [[ -f "$checkpoint" ]] || die "resume checkpoint is missing: $checkpoint"
        fi
    done

    if [[ "$use_mps" == true ]]; then
        start_mps
    else
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    fi
    configure_tmux_environment "$use_mps"

    for model in "${selected_models[@]}"; do
        session="gan-$model"
        printf -v worker_command 'bash %q __train_worker %q %q %q %q' \
            "$SCRIPT_PATH" "$model" "$num_workers" "$data_pipeline" "$resume"
        tmux -L "$TMUX_SOCKET" new-session -d \
            -s "$session" -c "$PROJECT_ROOT" "$worker_command"
        configure_tmux_environment "$use_mps"
        log "started $model in tmux session $session"
    done
    log "use 'bash tool_scripts/cloud_gan.sh status' to monitor"
}

show_status() {
    local logs=()
    require_no_args status "$@"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv || log "nvidia-smi query failed"
    fi
    if command -v tmux >/dev/null 2>&1 && tmux_server_exists; then
        tmux -L "$TMUX_SOCKET" list-sessions
    else
        log "no active GAN tmux sessions"
    fi
    shopt -s nullglob
    logs=("$OUTPUT_ROOT"/logs/*.log)
    if ((${#logs[@]} > 0)); then
        tail -n 8 "${logs[@]}"
    fi
}

attach_model() {
    local model="${1:-}"
    (($# == 1)) || die "attach requires exactly one model"
    validate_model "$model"
    command -v tmux >/dev/null 2>&1 || die "tmux is unavailable"
    exec tmux -L "$TMUX_SOCKET" attach-session -t "gan-$model"
}

stop_mps() {
    require_no_args stop-mps "$@"
    command -v nvidia-cuda-mps-control >/dev/null 2>&1 || \
        die "nvidia-cuda-mps-control is unavailable"
    if tmux_server_exists; then
        die "GAN tmux sessions are still active; stop them before MPS"
    fi
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
    if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo quit | nvidia-cuda-mps-control
        log "MPS daemon stopped"
    else
        log "no active MPS daemon found at $MPS_ROOT"
    fi
}

download_output() {
    local host="vast-dl"
    local apply=false
    local remote_source
    local rsync_args=(-aPzvn)
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
    ssh "$host" "test -d $REMOTE_OUTPUT_ROOT" || \
        die "remote output directory is unavailable"
    mkdir -p "$OUTPUT_ROOT"
    remote_source="$host:$REMOTE_OUTPUT_ROOT/"
    if [[ "$apply" == true ]]; then
        rsync_args=(-aPzv)
        log "merging remote output into $OUTPUT_ROOT"
    else
        log "previewing output merge; add --apply to copy files"
    fi
    rsync "${rsync_args[@]}" -- "$remote_source" "$OUTPUT_ROOT/"
    if [[ "$apply" == true ]]; then
        find "$OUTPUT_ROOT" -type f -printf '%T@ %p\n' | \
            sort -n | tail -n 10 | cut -d' ' -f2-
    fi
}

action="${1:-}"
[[ -n "$action" ]] || {
    usage
    exit 2
}
shift

case "$action" in
    provision) provision_cloud "$@" ;;
    validate) run_validation "$@" ;;
    smoke) run_smoke "$@" ;;
    benchmark) exec bash "$SCRIPT_DIR/test_gan_concurrency.sh" "$@" ;;
    train) run_train "$@" ;;
    status) show_status "$@" ;;
    attach) attach_model "$@" ;;
    stop-mps) stop_mps "$@" ;;
    download) download_output "$@" ;;
    __train_worker)
        (($# == 4)) || die "invalid internal worker arguments"
        train_worker "$@"
        ;;
    -h|--help) usage ;;
    *) die "unknown action: $action" ;;
esac
