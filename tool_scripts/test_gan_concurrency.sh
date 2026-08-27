#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

models_csv="progan,stylegan,stylegan2"
timed_batches=16
warmup_batches=1
batch_scale=1
num_workers=0
prefetch_factor=2
data_pipeline="cuda"
gpu_index=0
enable_mps=false
output_dir=""
monitor_pid=""
mps_started=false
mps_root=""
child_pids=()
model_command=()

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/test_gan_concurrency.sh [options]

Compare sequential and concurrent execution of two or three BF16 GAN lessons
on one NVIDIA B200, B300, RTX 5080, or RTX 5090. Each worker skips progressive
growth and runs a short, real 128x128 training burst with lazy regularization
and EMA updates.
Logs, timings, and GPU samples are isolated under
output-vast-dl/concurrency-benchmark/ by default.

Options:
  --models LIST              Comma-separated model names (at least two):
                             progan, stylegan, stylegan2
  --batches N                Timed 128x128 batches per worker (default: 16)
  --warmup-batches N         Untimed warm-up batches per worker (default: 1)
  --batch-scale N            Multiply each lesson's 128x128 batch (default: 1)
  --num-workers N            DataLoader workers per process (default: 0)
  --prefetch-factor N        DataLoader prefetch factor (default: 2)
  --data-pipeline PIPELINE   cuda (default), auto, or cpu
  --gpu-index N              GPU index for a non-MPS run (default: 0)
  --mps                      Start a private NVIDIA MPS daemon for the test
  --output-dir PATH          New benchmark directory; must not already exist
  -h, --help                 Show this help

Examples:
  bash tool_scripts/test_gan_concurrency.sh \
    --models progan,stylegan

  bash tool_scripts/test_gan_concurrency.sh \
    --models progan,stylegan,stylegan2 \
    --mps

Increase --batches only when the default run is too short to stabilize.
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

is_supported_gpu_name() {
    local gpu_name_upper="${1^^}"
    case "$gpu_name_upper" in
        *B200*|*B300*|*RTX*5080*|*RTX*5090*) return 0 ;;
        *) return 1 ;;
    esac
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
        --batches)
            (($# >= 2)) || die "--batches requires a value"
            timed_batches="$2"
            shift 2
            ;;
        --progressive-phase-kimg|--stylegan2-total-kimg)
            die "$1 was removed; use --batches for the fixed 128x128 benchmark"
            ;;
        --warmup-batches)
            (($# >= 2)) || die "--warmup-batches requires a value"
            warmup_batches="$2"
            shift 2
            ;;
        --batch-scale)
            (($# >= 2)) || die "--batch-scale requires a value"
            batch_scale="$2"
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
        --data-pipeline)
            (($# >= 2)) || die "--data-pipeline requires a value"
            data_pipeline="$2"
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

is_positive_integer "$timed_batches" || \
    die "--batches must be a positive integer"
[[ "$warmup_batches" =~ ^[0-9]+$ ]] || \
    die "--warmup-batches must be a non-negative integer"
is_positive_integer "$batch_scale" || \
    die "--batch-scale must be a positive integer"
[[ "$num_workers" =~ ^[0-9]+$ ]] || \
    die "--num-workers must be a non-negative integer"
is_positive_integer "$prefetch_factor" || \
    die "--prefetch-factor must be a positive integer"
case "$data_pipeline" in
    auto|cuda|cpu) ;;
    *) die "--data-pipeline must be one of: auto, cuda, cpu" ;;
esac
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
[[ -f "$PROJECT_ROOT/tool_scripts/benchmark_gan_training.py" ]] || \
    die "tool_scripts/benchmark_gan_training.py is missing"
[[ -f "$PROJECT_ROOT/data/celeba/list_eval_partition.csv" ]] || \
    die "prepared CelebA data is missing under data/celeba"

gpu_name="$(nvidia-smi -i "$gpu_index" --query-gpu=name --format=csv,noheader)" || \
    die "cannot query GPU $gpu_index"
gpu_name="${gpu_name//$'\n'/ }"
if ! is_supported_gpu_name "$gpu_name"; then
    die "GPU $gpu_index is '$gpu_name'; expected B200, B300, RTX 5080, or RTX 5090"
fi

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$gpu_index" \
    uv run --locked --no-sync python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'

if [[ -z "$output_dir" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="$PROJECT_ROOT/output-vast-dl/concurrency-benchmark/$timestamp"
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
    local result_file="$2"
    model_command=(
        uv run --locked --no-sync python -u
        tool_scripts/benchmark_gan_training.py
        "$model"
        --batches "$timed_batches"
        --warmup-batches "$warmup_batches"
        --batch-scale "$batch_scale"
        --num-workers "$num_workers"
        --prefetch-factor "$prefetch_factor"
        --data-pipeline "$data_pipeline"
        --result-file "$result_file"
    )
}

run_model() {
    local mode="$1"
    local model="$2"
    local mode_root="$output_dir/$mode"
    local model_log="$mode_root/logs/$model.log"
    local training_file="$mode_root/timing/$model-training.tsv"
    local process_file="$mode_root/timing/$model-process.tsv"
    local start_seconds end_seconds elapsed_seconds status
    mkdir -p "$mode_root/logs" "$mode_root/timing"
    build_command "$model" "$training_file"
    start_seconds="$EPOCHREALTIME"
    log "$mode: starting $model"
    if [[ "$mps_started" == true ]]; then
        if "${model_command[@]}" >"$model_log" 2>&1; then
            status=0
        else
            status=$?
        fi
    else
        if CUDA_VISIBLE_DEVICES="$gpu_index" \
            "${model_command[@]}" >"$model_log" 2>&1; then
            status=0
        else
            status=$?
        fi
    fi
    end_seconds="$EPOCHREALTIME"
    elapsed_seconds="$(awk -v start="$start_seconds" -v end="$end_seconds" \
        'BEGIN { printf "%.3f", end - start }')"
    printf 'model\telapsed_seconds\texit_code\n%s\t%s\t%s\n' \
        "$model" "$elapsed_seconds" "$status" > "$process_file"
    log "$mode: $model exited with status $status after ${elapsed_seconds}s"
    return "$status"
}

start_monitor() {
    local mode="$1"
    mkdir -p "$output_dir/$mode"
    nvidia-smi -i "$gpu_index" \
        --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv -l 1 > "$output_dir/$mode/gpu.csv" &
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
    local start_seconds end_seconds elapsed_seconds status=0
    start_seconds="$EPOCHREALTIME"
    start_monitor sequential
    for model in "${models[@]}"; do
        if ! run_model sequential "$model"; then
            status=1
        fi
    done
    stop_monitor
    end_seconds="$EPOCHREALTIME"
    elapsed_seconds="$(awk -v start="$start_seconds" -v end="$end_seconds" \
        'BEGIN { printf "%.3f", end - start }')"
    printf '%s\n' "$elapsed_seconds" > \
        "$output_dir/sequential/elapsed-seconds.txt"
    return "$status"
}

run_parallel() {
    local start_seconds end_seconds elapsed_seconds status=0
    local model pid
    declare -A model_pids=()
    start_seconds="$EPOCHREALTIME"
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
    end_seconds="$EPOCHREALTIME"
    elapsed_seconds="$(awk -v start="$start_seconds" -v end="$end_seconds" \
        'BEGIN { printf "%.3f", end - start }')"
    printf '%s\n' "$elapsed_seconds" > \
        "$output_dir/parallel/elapsed-seconds.txt"
    return "$status"
}

log "GPU: $gpu_name"
log "models: ${models[*]}"
log "load: 128x128, batch_scale=$batch_scale, warmup=$warmup_batches, timed=$timed_batches"
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
    printf 'resolution=128\n'
    printf 'batch_scale=%s\n' "$batch_scale"
    printf 'warmup_batches=%s\n' "$warmup_batches"
    printf 'timed_batches=%s\n' "$timed_batches"
    printf 'data_pipeline=%s\n' "$data_pipeline"
    printf 'sequential_seconds=%s\n' "$sequential_seconds"
    printf 'parallel_seconds=%s\n' "$parallel_seconds"
    printf 'aggregate_speedup=%s\n' "$speedup"
    printf '\nmode\tmodel\tresolution\tbatch_size\tpipeline\tprecision\twarmup_batches\ttimed_batches\ttimed_images\ttrain_seconds\timages_per_second\tpeak_allocated_gib\tpeak_reserved_gib\n'
    for mode in sequential parallel; do
        for model in "${models[@]}"; do
            printf '%s\t' "$mode"
            tail -n 1 "$output_dir/$mode/timing/$model-training.tsv"
        done
    done
} | tee "$output_dir/summary.txt"

log "compare aggregate speedup, per-model throughput, peak memory, logs, and gpu.csv"
