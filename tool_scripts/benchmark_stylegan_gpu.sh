#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

timed_batches=22528
warmup_batches=16
batch_size=32
num_workers=0
prefetch_factor=2
data_pipeline="cuda"
precision="bf16"
gpu_index=0
output_dir=""
monitor_pid=""

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/benchmark_stylegan_gpu.sh [options]
       bash tool_scripts/benchmark_stylegan_gpu.sh --compare BASELINE CANDIDATE
       bash tool_scripts/benchmark_stylegan_gpu.sh --compare-precision BASELINE CANDIDATE

Run a fixed-workload, single-GPU StyleGAN benchmark at the progressive model's
maximum 128x128 resolution. The timed loop uses real discriminator/generator
forward and backward paths, EMA, lazy R1 regularization every 16 batches, and
fused Adam parameter updates. BF16 remains the default training precision.
TF32 and strict FP32 are benchmark-only diagnostics. The full defaults are
intended to take roughly one to two hours on an RTX 4070 Ti.

Options:
  --batches N                Timed batches (default: 22528)
  --warmup-batches N         Untimed warm-up batches (default: 16)
  --batch-size N             Images per batch (default: 32)
  --num-workers N            DataLoader workers (default: 0)
  --prefetch-factor N        DataLoader prefetch factor (default: 2)
  --data-pipeline PIPELINE   cuda (default), auto, or cpu
  --precision PRECISION      bf16 (default), tf32, or strict fp32
  --gpu-index N              GPU index (default: 0)
  --output-dir PATH          New result directory; must not already exist
  --compare BASELINE CANDIDATE
                             Compare two training.tsv files from identical runs
  --compare-precision BASELINE CANDIDATE
                             Compare different precisions with identical loads
  -h, --help                 Show this help

Run the same command on both GPUs. Pass the RTX 4070 Ti result first when
comparing so the reported speedup is candidate throughput / baseline throughput.
EOF
}

log() {
    printf '[stylegan-gpu] %s\n' "$*"
}

die() {
    printf '[stylegan-gpu] ERROR: %s\n' "$*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
        monitor_pid=""
    fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

tsv_field() {
    local path="$1"
    local field="$2"
    awk -F '\t' -v field="$field" '
        NR == 1 {
            for (field_index = 1; field_index <= NF; field_index++) {
                if ($field_index == field) column = field_index
            }
            next
        }
        NR == 2 && column { print $column; exit }
    ' "$path"
}

compare_results() {
    local comparison_kind="$1"
    local baseline="$2"
    local candidate="$3"
    local field baseline_value candidate_value
    local baseline_precision candidate_precision
    local baseline_rate candidate_rate speedup
    [[ -f "$baseline" ]] || die "baseline result is missing: $baseline"
    [[ -f "$candidate" ]] || die "candidate result is missing: $candidate"

    for field in model resolution batch_size pipeline \
        warmup_batches timed_batches timed_images; do
        baseline_value="$(tsv_field "$baseline" "$field")"
        candidate_value="$(tsv_field "$candidate" "$field")"
        [[ -n "$baseline_value" && -n "$candidate_value" ]] || \
            die "missing '$field' in one comparison result"
        [[ "$baseline_value" == "$candidate_value" ]] || \
            die "$field differs: baseline=$baseline_value candidate=$candidate_value"
    done

    baseline_precision="$(tsv_field "$baseline" precision)"
    candidate_precision="$(tsv_field "$candidate" precision)"
    [[ -n "$baseline_precision" && -n "$candidate_precision" ]] || \
        die "precision is missing from one comparison result"
    if [[ "$comparison_kind" == "hardware" ]]; then
        [[ "$baseline_precision" == "$candidate_precision" ]] || \
            die "precision differs: baseline=$baseline_precision candidate=$candidate_precision"
    else
        [[ "$baseline_precision" != "$candidate_precision" ]] || \
            die "--compare-precision requires two different precisions"
    fi

    baseline_rate="$(tsv_field "$baseline" images_per_second)"
    candidate_rate="$(tsv_field "$candidate" images_per_second)"
    [[ -n "$baseline_rate" && -n "$candidate_rate" ]] || \
        die "images_per_second is missing from one comparison result"
    speedup="$(awk -v baseline="$baseline_rate" -v candidate="$candidate_rate" \
        'BEGIN { if (baseline <= 0) exit 1; printf "%.3f", candidate / baseline }')" || \
        die "baseline throughput must be positive"

    printf 'baseline=%s\n' "$baseline"
    printf 'candidate=%s\n' "$candidate"
    printf 'baseline_precision=%s\n' "$baseline_precision"
    printf 'candidate_precision=%s\n' "$candidate_precision"
    printf 'baseline_images_per_second=%s\n' "$baseline_rate"
    printf 'candidate_images_per_second=%s\n' "$candidate_rate"
    printf 'candidate_speedup=%sx\n' "$speedup"
}

if (($# > 0)) && [[ "$1" == "--compare" || "$1" == "--compare-precision" ]]; then
    (($# == 3)) || die "$1 requires BASELINE and CANDIDATE"
    if [[ "$1" == "--compare" ]]; then
        compare_results hardware "$2" "$3"
    else
        compare_results precision "$2" "$3"
    fi
    exit 0
fi

while (($# > 0)); do
    case "$1" in
        --batches)
            (($# >= 2)) || die "--batches requires a value"
            timed_batches="$2"
            shift 2
            ;;
        --warmup-batches)
            (($# >= 2)) || die "--warmup-batches requires a value"
            warmup_batches="$2"
            shift 2
            ;;
        --batch-size)
            (($# >= 2)) || die "--batch-size requires a value"
            batch_size="$2"
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
        --precision)
            (($# >= 2)) || die "--precision requires a value"
            precision="$2"
            shift 2
            ;;
        --gpu-index)
            (($# >= 2)) || die "--gpu-index requires a value"
            gpu_index="$2"
            shift 2
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

is_positive_integer "$timed_batches" || die "--batches must be positive"
[[ "$warmup_batches" =~ ^[0-9]+$ ]] || \
    die "--warmup-batches must be non-negative"
is_positive_integer "$batch_size" || die "--batch-size must be positive"
[[ "$num_workers" =~ ^[0-9]+$ ]] || \
    die "--num-workers must be non-negative"
is_positive_integer "$prefetch_factor" || \
    die "--prefetch-factor must be positive"
[[ "$gpu_index" =~ ^[0-9]+$ ]] || die "--gpu-index must be non-negative"
case "$data_pipeline" in
    auto|cuda|cpu) ;;
    *) die "--data-pipeline must be one of: auto, cuda, cpu" ;;
esac
case "$precision" in
    bf16|tf32|fp32) ;;
    *) die "--precision must be one of: bf16, tf32, fp32" ;;
esac

command -v uv >/dev/null 2>&1 || die "uv is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
[[ -f "$PROJECT_ROOT/tool_scripts/benchmark_gan_training.py" ]] || \
    die "benchmark_gan_training.py is missing"
[[ -f "$PROJECT_ROOT/data/celeba/list_eval_partition.csv" ]] || \
    die "prepared CelebA data is missing under data/celeba"

gpu_name="$(nvidia-smi -i "$gpu_index" --query-gpu=name --format=csv,noheader)" || \
    die "cannot query GPU $gpu_index"
driver_version="$(nvidia-smi -i "$gpu_index" \
    --query-gpu=driver_version --format=csv,noheader)" || \
    die "cannot query the NVIDIA driver"
memory_mib="$(nvidia-smi -i "$gpu_index" \
    --query-gpu=memory.total --format=csv,noheader,nounits)" || \
    die "cannot query GPU memory"
gpu_name="${gpu_name//$'\n'/ }"
driver_version="${driver_version//[[:space:]]/}"
memory_mib="${memory_mib//[[:space:]]/}"

cd "$PROJECT_ROOT"
runtime="$(CUDA_VISIBLE_DEVICES="$gpu_index" \
    uv run --locked --no-sync python -c \
    'import sys, torch; requested = sys.argv[1]; assert torch.cuda.is_available(), "CUDA unavailable"; assert requested != "bf16" or torch.cuda.is_bf16_supported(), "BF16 unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} capability={torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]} bf16={torch.cuda.is_bf16_supported()}")' \
    "$precision")" || die "PyTorch precision preflight failed"

if [[ -z "$output_dir" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="$PROJECT_ROOT/output-vast-dl/single-gpu-benchmark/$timestamp"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$PROJECT_ROOT/$output_dir"
fi
[[ ! -e "$output_dir" ]] || die "output directory already exists: $output_dir"
mkdir -p "$output_dir"

result_file="$output_dir/training.tsv"
process_file="$output_dir/process.tsv"
log_file="$output_dir/training.log"
gpu_file="$output_dir/gpu.csv"
monitor_log="$output_dir/gpu-monitor.log"
r1_steps=$((
    (warmup_batches + timed_batches - 1) / 16 -
    (warmup_batches + 15) / 16 + 1
))

printf 'timestamp,name,utilization.gpu,memory.used,memory.total,power.draw\n' > \
    "$gpu_file"
nvidia-smi -i "$gpu_index" \
    --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader -l 1 >> "$gpu_file" 2> "$monitor_log" &
monitor_pid=$!

benchmark_command=(
    uv run --locked --no-sync python -u
    tool_scripts/benchmark_gan_training.py
    stylegan
    --precision "$precision"
    --batches "$timed_batches"
    --warmup-batches "$warmup_batches"
    --batch-size "$batch_size"
    --num-workers "$num_workers"
    --prefetch-factor "$prefetch_factor"
    --data-pipeline "$data_pipeline"
    --result-file "$result_file"
)

log "GPU: $gpu_name (${memory_mib} MiB), driver $driver_version"
log "runtime: $runtime"
log "load: 128x128, batch=$batch_size, precision=$precision, warmup=$warmup_batches, timed=$timed_batches"
log "parameter updates: fused Adam enabled"
log "timed regularization: R1=$r1_steps"
log "results: $output_dir"

start_seconds="$EPOCHREALTIME"
if CUDA_VISIBLE_DEVICES="$gpu_index" \
    "${benchmark_command[@]}" 2>&1 | tee "$log_file"; then
    status=0
else
    status=$?
fi
end_seconds="$EPOCHREALTIME"
elapsed_seconds="$(awk -v start="$start_seconds" -v end="$end_seconds" \
    'BEGIN { printf "%.3f", end - start }')"
cleanup
printf 'elapsed_seconds\texit_code\n%s\t%s\n' \
    "$elapsed_seconds" "$status" > "$process_file"
((status == 0)) || die "benchmark failed; inspect $log_file"

{
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'gpu=%s\n' "$gpu_name"
    printf 'gpu_memory_mib=%s\n' "$memory_mib"
    printf 'driver=%s\n' "$driver_version"
    printf 'runtime=%s\n' "$runtime"
    printf 'precision=%s\n' "$precision"
    printf 'parameter_updates=true\n'
    printf 'r1_every_batches=16\n'
    printf 'timed_r1_steps=%s\n' "$r1_steps"
    printf 'process_elapsed_seconds=%s\n' "$elapsed_seconds"
    grep -E '^(metrics_finite|nonfinite_metrics)=' "$log_file"
    printf '\n'
    cat "$result_file"
} | tee "$output_dir/summary.txt"

log "completed; compare training.tsv files with --compare"
