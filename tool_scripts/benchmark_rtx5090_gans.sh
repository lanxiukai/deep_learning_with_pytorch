#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

models_csv="sn_gan,sagan,biggan"
steps=10
warmup_steps=5
batch_size=""
generator_batch_size=""
precision="bf16"
data_dir=""
num_workers=4
output_dir=""

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/benchmark_rtx5090_gans.sh [options]

Run sequential finite-loss optimization checks for the Imagenette-128 SN-GAN,
SAGAN, and BigGAN defaults on one RTX 5090. Synthetic 128x128 batches are used
unless --data-dir points to a prepared Imagenette-128 root.

Options:
  --models LIST          Comma-separated sn_gan,sagan,biggan (default: all)
  --steps N              Timed discriminator steps per model (default: 10)
  --warmup-steps N       Warmup steps per model (default: 5)
  --batch-size N         Override each lesson's RTX 5090 batch default
  --generator-batch-size N
                         Override the generated-image batch independently
  --precision MODE       bf16 (default) or fp32
  --data-dir PATH        Use Imagenette-128 instead of synthetic images
  --num-workers N        Imagenette DataLoader workers (default: 4)
  --output-dir PATH      New result directory (default: timestamped)
  -h, --help             Show this help
EOF
}

die() {
    printf '[rtx5090-benchmark] ERROR: %s\n' "$*" >&2
    exit 1
}

validate_model() {
    case "$1" in
        sn_gan|sagan|biggan) ;;
        *) die "unknown model: $1" ;;
    esac
}

while (($# > 0)); do
    case "$1" in
        --models)
            (($# >= 2)) || die "--models requires a value"
            models_csv="$2"
            shift 2
            ;;
        --steps)
            (($# >= 2)) || die "--steps requires a value"
            steps="$2"
            shift 2
            ;;
        --warmup-steps)
            (($# >= 2)) || die "--warmup-steps requires a value"
            warmup_steps="$2"
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
        --num-workers)
            (($# >= 2)) || die "--num-workers requires a value"
            num_workers="$2"
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

[[ "$steps" =~ ^[1-9][0-9]*$ ]] || die "--steps must be positive"
[[ "$warmup_steps" =~ ^[0-9]+$ ]] || die "--warmup-steps must be non-negative"
[[ "$num_workers" =~ ^[0-9]+$ ]] || die "--num-workers must be non-negative"
if [[ -n "$batch_size" ]]; then
    [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be positive"
fi
if [[ -n "$generator_batch_size" ]]; then
    [[ "$generator_batch_size" =~ ^[1-9][0-9]*$ ]] || \
        die "--generator-batch-size must be positive"
fi
case "$precision" in
    bf16|fp32) ;;
    *) die "--precision must be bf16 or fp32" ;;
esac

IFS=',' read -r -a models <<< "$models_csv"
((${#models[@]} > 0)) || die "--models must not be empty"
declare -A seen=()
for model in "${models[@]}"; do
    validate_model "$model"
    [[ -z "${seen[$model]:-}" ]] || die "duplicate model: $model"
    seen[$model]=1
done

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${gpu_name^^}" == *"RTX 5090"* ]] || die "expected RTX 5090, found: $gpu_name"
command -v uv >/dev/null 2>&1 || die "uv is unavailable"

if [[ -z "$output_dir" ]]; then
    output_dir="$PROJECT_ROOT/output-vast-dl/benchmarks/rtx5090-$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$PROJECT_ROOT/$output_dir"
fi
[[ ! -e "$output_dir" ]] || die "output directory already exists: $output_dir"
mkdir -p "$output_dir"

cd "$PROJECT_ROOT"
printf '[rtx5090-benchmark] GPU: %s\n' "$gpu_name"
printf '[rtx5090-benchmark] Results: %s\n' "$output_dir"
for model in "${models[@]}"; do
    command=(
        uv run --locked --no-sync python -u
        tool_scripts/benchmark_gan_training.py
        --model "$model"
        --steps "$steps"
        --warmup-steps "$warmup_steps"
        --precision "$precision"
        --num-workers "$num_workers"
        --json-output "$output_dir/$model.json"
    )
    [[ -z "$batch_size" ]] || command+=(--batch-size "$batch_size")
    [[ -z "$generator_batch_size" ]] || \
        command+=(--generator-batch-size "$generator_batch_size")
    [[ -z "$data_dir" ]] || command+=(--data-dir "$data_dir")
    printf '[rtx5090-benchmark] Starting %s\n' "$model"
    "${command[@]}" 2>&1 | tee "$output_dir/$model.log"
done

printf '[rtx5090-benchmark] All checks passed. Inspect %s/*.json\n' "$output_dir"
