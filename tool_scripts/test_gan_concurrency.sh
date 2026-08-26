#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

models_csv="progan,stylegan,stylegan2"
progressive_phase_kimg=5
stylegan2_total_kimg=50
num_workers=4
prefetch_factor=2
gpu_index=0
enable_mps=false
output_dir=""
monitor_pid=""
mps_started=false
mps_root=""
child_pids=()

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/test_gan_concurrency.sh [options]

Compare sequential and concurrent execution of two or three BF16 GAN lessons
on one NVIDIA B200 or B300. Benchmark artifacts are isolated from normal model
outputs under output/concurrency-benchmark/ by default.

Options:
  --models LIST                  Comma-separated model names (at least two):
                                 progan, stylegan, stylegan2
  --progressive-phase-kimg N     Per-phase ProGAN/StyleGAN budget (default: 5)
  --stylegan2-total-kimg N       StyleGAN2 budget (default: 50)
  --num-workers N                DataLoader workers per process (default: 4)
  --prefetch-factor N            DataLoader prefetch factor (default: 2)
  --gpu-index N                  GPU index for a non-MPS run (default: 0)
  --mps                          Start a private NVIDIA MPS daemon for the test
  --output-dir PATH              New benchmark directory; must not already exist
  -h, --help                     Show this help

Examples:
  bash tool_scripts/test_gan_concurrency.sh \
    --models progan,stylegan

  bash tool_scripts/test_gan_concurrency.sh \
    --models progan,stylegan,stylegan2 \
    --mps
EOF
}

log() {
    printf '[gan-concurrency] %s\n' "$*"
}

die() {
    printf '[gan-concurrency] ERROR: %s\n' "$*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
    fi
    for pid in "${child_pids[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${child_pids[@]}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
    if [[ "$mps_started" == true ]]; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    if [[ -n "$mps_root" && -d "$mps_root" ]]; then
        rm -rf -- "$mps_root"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

while (($# > 0)); do
    case "$1" in
        --models)
            (($# >= 2)) || die "--models requires a value"
            models_csv="$2"
            shift 2
            ;;
        --progressive-phase-kimg)
            (($# >= 2)) || die "--progressive-phase-kimg requires a value"
            progressive_phase_kimg="$2"
            shift 2
            ;;
        --stylegan2-total-kimg)
            (($# >= 2)) || die "--stylegan2-total-kimg requires a value"
            stylegan2_total_kimg="$2"
            shift 2
            ;;
        --num-workers)
            (($# >= 2)) || die "--num-workers requires a value"
            num_workers="$2"
            shift 2
            ;;
        --prefetch-factor)
            (($# >= 2)) || die "--prefetch-factor requires a value"
            prefetch_factor="$2"
            shift 2
            ;;
        --gpu-index)
            (($# >= 2)) || die "--gpu-index requires a value"
            gpu_index="$2"
            shift 2
            ;;
        --mps)
            enable_mps=true
            shift
            ;;
        --output-dir)
            (($# >= 2)) || die "--output-dir requires a value"
            output_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

is_positive_integer "$progressive_phase_kimg" || \
    die "--progressive-phase-kimg must be a positive integer"
is_positive_integer "$stylegan2_total_kimg" || \
    die "--stylegan2-total-kimg must be a positive integer"
[[ "$num_workers" =~ ^[0-9]+$ ]] || \
    die "--num-workers must be a non-negative integer"
is_positive_integer "$prefetch_factor" || \
    die "--prefetch-factor must be a positive integer"
[[ "$gpu_index" =~ ^[0-9]+$ ]] || die "--gpu-index must be non-negative"

IFS=',' read -r -a requested_models <<< "$models_csv"
models=()
for model in "${requested_models[@]}"; do
    case "$model" in
        progan|stylegan|stylegan2) ;;
        *) die "unknown model in --models: $model" ;;
    esac
    for existing in "${models[@]}"; do
        [[ "$model" != "$existing" ]] || die "duplicate model: $model"
    done
    models+=("$model")
done
((${#models[@]} >= 2 && ${#models[@]} <= 3)) || \
    die "--models must select two or three models"

command -v uv >/dev/null 2>&1 || die "uv is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
[[ -f "$PROJECT_ROOT/data/celeba/list_eval_partition.csv" ]] || \
    die "prepared CelebA data is missing under data/celeba"

gpu_name="$(nvidia-smi -i "$gpu_index" --query-gpu=name --format=csv,noheader)" || \
    die "cannot query GPU $gpu_index"
gpu_name="${gpu_name//$'\n'/ }"
gpu_name_upper="${gpu_name^^}"
if [[ "$gpu_name_upper" != *B200* && "$gpu_name_upper" != *B300* ]]; then
    die "GPU $gpu_index is '$gpu_name'; this benchmark is limited to B200/B300"
fi

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$gpu_index" \
    uv run --locked --no-sync python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'

if [[ -z "$output_dir" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="$PROJECT_ROOT/output/concurrency-benchmark/$timestamp"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$PROJECT_ROOT/$output_dir"
fi
[[ ! -e "$output_dir" ]] || die "output directory already exists: $output_dir"
mkdir -p "$output_dir"

if [[ "$enable_mps" == true ]]; then
    command -v nvidia-cuda-mps-control >/dev/null 2>&1 || \
        die "nvidia-cuda-mps-control is unavailable in this instance"
    mps_root="$(mktemp -d /tmp/dl-gan-mps.XXXXXX)"
    export CUDA_MPS_PIPE_DIRECTORY="$mps_root/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$mps_root/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    gpu_uuid="$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)"
    CUDA_VISIBLE_DEVICES="$gpu_uuid" nvidia-cuda-mps-control -d
    export CUDA_VISIBLE_DEVICES="$gpu_uuid"
    mps_started=true
    log "private MPS daemon started for $gpu_name"
fi

build_command() {
    local model="$1"
    model_command=(uv run --locked --no-sync python -u)
    case "$model" in
        progan)
            model_command+=(
                genai/1.0_generative_adversarial_network/7.0_progan.py
                --phase-kimg "$progressive_phase_kimg"
            )
            ;;
        stylegan)
            model_command+=(
                genai/1.0_generative_adversarial_network/7.1_stylegan.py
                --phase-kimg "$progressive_phase_kimg"
            )
            ;;
        stylegan2)
            model_command+=(
                genai/1.0_generative_adversarial_network/7.2_stylegan2.py
                --total-kimg "$stylegan2_total_kimg"
            )
            ;;
    esac
    model_command+=(
        --num-workers "$num_workers"
        --prefetch-factor "$prefetch_factor"
    )
}

run_model() {
    local mode="$1"
    local model="$2"
    local mode_root="$output_dir/$mode"
    local model_log="$mode_root/logs/$model.log"
    local timing_file="$mode_root/timing/$model.tsv"
    local start_seconds end_seconds status
    mkdir -p "$mode_root/logs" "$mode_root/timing" "$mode_root/artifacts"
    build_command "$model"
    start_seconds="$(date +%s)"
    log "$mode: starting $model"
    if [[ "$mps_started" == true ]]; then
        if DL_OUTPUT_ROOT="$mode_root/artifacts" \
            "${model_command[@]}" >"$model_log" 2>&1; then
            status=0
        else
            status=$?
        fi
    else
        if CUDA_VISIBLE_DEVICES="$gpu_index" \
            DL_OUTPUT_ROOT="$mode_root/artifacts" \
            "${model_command[@]}" >"$model_log" 2>&1; then
            status=0
        else
            status=$?
        fi
    fi
    end_seconds="$(date +%s)"
    printf 'model\telapsed_seconds\texit_code\n%s\t%s\t%s\n' \
        "$model" "$((end_seconds - start_seconds))" "$status" > "$timing_file"
    log "$mode: $model exited with status $status"
    return "$status"
}

start_monitor() {
    local mode="$1"
    mkdir -p "$output_dir/$mode"
    nvidia-smi -i "$gpu_index" \
        --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv -l 2 > "$output_dir/$mode/gpu.csv" &
    monitor_pid=$!
}

stop_monitor() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
        monitor_pid=""
    fi
}

run_sequential() {
    local start_seconds end_seconds status=0
    start_seconds="$(date +%s)"
    start_monitor sequential
    for model in "${models[@]}"; do
        if ! run_model sequential "$model"; then
            status=1
        fi
    done
    stop_monitor
    end_seconds="$(date +%s)"
    printf '%s\n' "$((end_seconds - start_seconds))" > \
        "$output_dir/sequential/elapsed-seconds.txt"
    return "$status"
}

run_parallel() {
    local start_seconds end_seconds status=0
    local model pid
    declare -A model_pids=()
    start_seconds="$(date +%s)"
    start_monitor parallel
    child_pids=()
    for model in "${models[@]}"; do
        run_model parallel "$model" &
        pid=$!
        child_pids+=("$pid")
        model_pids["$model"]="$pid"
    done
    for model in "${models[@]}"; do
        if ! wait "${model_pids[$model]}"; then
            status=1
        fi
    done
    child_pids=()
    stop_monitor
    end_seconds="$(date +%s)"
    printf '%s\n' "$((end_seconds - start_seconds))" > \
        "$output_dir/parallel/elapsed-seconds.txt"
    return "$status"
}

log "GPU: $gpu_name"
log "models: ${models[*]}"
log "results: $output_dir"
run_sequential || die "the sequential baseline failed; inspect its logs"
run_parallel || die "the parallel benchmark failed; inspect its logs"

sequential_seconds="$(<"$output_dir/sequential/elapsed-seconds.txt")"
parallel_seconds="$(<"$output_dir/parallel/elapsed-seconds.txt")"
speedup="$(awk -v sequential="$sequential_seconds" -v parallel="$parallel_seconds" \
    'BEGIN { if (parallel == 0) print "inf"; else printf "%.3f", sequential / parallel }')"

{
    printf 'gpu=%s\n' "$gpu_name"
    printf 'models=%s\n' "${models[*]}"
    printf 'mps=%s\n' "$mps_started"
    printf 'sequential_seconds=%s\n' "$sequential_seconds"
    printf 'parallel_seconds=%s\n' "$parallel_seconds"
    printf 'aggregate_speedup=%s\n' "$speedup"
} | tee "$output_dir/summary.txt"

log "compare per-model timing, logs, generated samples, and gpu.csv before choosing concurrency"
