#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly STYLEGAN_BENCHMARK="$SCRIPT_DIR/benchmark_stylegan_gpu.sh"

timed_batches=512
warmup_batches=16
batch_size=64
num_workers=0
prefetch_factor=2
data_pipeline="cuda"
gpu_index=0
output_dir=""

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/benchmark_h100_precision.sh [options]

Compare strict FP32, TF32-enabled FP32, and BF16 on one NVIDIA H100 with the
same 128x128 StyleGAN training workload. Every timed run includes fused Adam
updates, EMA, and lazy R1 regularization. Defaults should finish in a few
minutes and use the formal lesson's batch size of 64.

Options:
  --batches N                Timed batches per precision (default: 512)
  --warmup-batches N         Untimed warm-up batches (default: 16)
  --batch-size N             Images per batch (default: 64)
  --num-workers N            DataLoader workers (default: 0)
  --prefetch-factor N        DataLoader prefetch factor (default: 2)
  --data-pipeline PIPELINE   cuda (default), auto, or cpu
  --gpu-index N              H100 index (default: 0)
  --output-dir PATH          New result directory; must not already exist
  -h, --help                 Show this help

Inspect bf16_vs_tf32_speedup for the practical precision decision and
bf16_vs_fp32_speedup to isolate the gain over strict FP32 CUDA math.
EOF
}

log() {
    printf '[h100-precision] %s\n' "$*"
}

die() {
    printf '[h100-precision] ERROR: %s\n' "$*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

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

ratio() {
    local numerator="$1"
    local denominator="$2"
    awk -v numerator="$numerator" -v denominator="$denominator" '
        BEGIN {
            if (denominator <= 0) exit 1
            printf "%.3f", numerator / denominator
        }
    '
}

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
        *) die "unknown option: $1" ;;
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

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
[[ -x "$STYLEGAN_BENCHMARK" ]] || die "benchmark_stylegan_gpu.sh is unavailable"

gpu_name="$(nvidia-smi -i "$gpu_index" --query-gpu=name --format=csv,noheader)" || \
    die "cannot query GPU $gpu_index"
gpu_name="${gpu_name//$'\n'/ }"
case "${gpu_name^^}" in
    *H100*) ;;
    *) die "GPU $gpu_index is '$gpu_name'; this comparison requires an H100" ;;
esac

if [[ -z "$output_dir" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="$PROJECT_ROOT/output-vast-dl/h100-precision-benchmark/$timestamp"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$PROJECT_ROOT/$output_dir"
fi
[[ ! -e "$output_dir" ]] || die "output directory already exists: $output_dir"
mkdir -p "$output_dir"

benchmark_args=(
    --batches "$timed_batches"
    --warmup-batches "$warmup_batches"
    --batch-size "$batch_size"
    --num-workers "$num_workers"
    --prefetch-factor "$prefetch_factor"
    --data-pipeline "$data_pipeline"
    --gpu-index "$gpu_index"
)

log "GPU: $gpu_name"
log "results: $output_dir"
for precision in fp32 tf32 bf16; do
    log "starting $precision"
    bash "$STYLEGAN_BENCHMARK" \
        "${benchmark_args[@]}" \
        --precision "$precision" \
        --output-dir "$output_dir/$precision"
done

fp32_file="$output_dir/fp32/training.tsv"
tf32_file="$output_dir/tf32/training.tsv"
bf16_file="$output_dir/bf16/training.tsv"
fp32_rate="$(tsv_field "$fp32_file" images_per_second)"
tf32_rate="$(tsv_field "$tf32_file" images_per_second)"
bf16_rate="$(tsv_field "$bf16_file" images_per_second)"
fp32_memory="$(tsv_field "$fp32_file" peak_reserved_gib)"
tf32_memory="$(tsv_field "$tf32_file" peak_reserved_gib)"
bf16_memory="$(tsv_field "$bf16_file" peak_reserved_gib)"

{
    printf 'git_commit=%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    printf 'gpu=%s\n' "$gpu_name"
    printf 'resolution=128\n'
    printf 'batch_size=%s\n' "$batch_size"
    printf 'warmup_batches=%s\n' "$warmup_batches"
    printf 'timed_batches=%s\n' "$timed_batches"
    printf 'fp32_images_per_second=%s\n' "$fp32_rate"
    printf 'tf32_images_per_second=%s\n' "$tf32_rate"
    printf 'bf16_images_per_second=%s\n' "$bf16_rate"
    printf 'tf32_vs_fp32_speedup=%sx\n' "$(ratio "$tf32_rate" "$fp32_rate")"
    printf 'bf16_vs_fp32_speedup=%sx\n' "$(ratio "$bf16_rate" "$fp32_rate")"
    printf 'bf16_vs_tf32_speedup=%sx\n' "$(ratio "$bf16_rate" "$tf32_rate")"
    printf 'fp32_peak_reserved_gib=%s\n' "$fp32_memory"
    printf 'tf32_peak_reserved_gib=%s\n' "$tf32_memory"
    printf 'bf16_peak_reserved_gib=%s\n' "$bf16_memory"
} | tee "$output_dir/summary.txt"

bash "$STYLEGAN_BENCHMARK" --compare-precision "$fp32_file" "$tf32_file" > \
    "$output_dir/tf32-vs-fp32.txt"
bash "$STYLEGAN_BENCHMARK" --compare-precision "$fp32_file" "$bf16_file" > \
    "$output_dir/bf16-vs-fp32.txt"
bash "$STYLEGAN_BENCHMARK" --compare-precision "$tf32_file" "$bf16_file" > \
    "$output_dir/bf16-vs-tf32.txt"

log "completed; use bf16_vs_tf32_speedup for the practical decision"
