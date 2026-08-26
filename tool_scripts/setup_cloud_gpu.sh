#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

profile="core"
state_dir="$PROJECT_ROOT/.cache/cloud-gpu"
skip_system_packages=false
skip_gpu_check=false
dry_run=false

usage() {
    cat <<'EOF'
Usage: bash tool_scripts/setup_cloud_gpu.sh [options]

Prepare the locked BF16 project environment on an Ubuntu x86_64 cloud GPU host.
The script is idempotent and does not install NVIDIA drivers, CUDA Toolkit,
Codex, datasets, or editor extensions.

Options:
  --profile PROFILE          core (default), examples, or full
  --state-dir PATH           Persistent uv/Python cache directory
  --skip-system-packages     Do not install missing curl/git/tmux/rsync
  --skip-gpu-check           Allow setup without a visible compatible GPU
  --dry-run                  Validate inputs and print the planned actions
  -h, --help                 Show this help

Profiles:
  core       Base dependencies, including PyTorch and torchvision
  examples   Core plus the optional examples dependencies
  full       Every optional dependency, including examples and tests
EOF
}

log() {
    printf '[cloud-gpu-setup] %s\n' "$*"
}

die() {
    printf '[cloud-gpu-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --profile)
            (($# >= 2)) || die "--profile requires a value"
            profile="$2"
            shift 2
            ;;
        --state-dir)
            (($# >= 2)) || die "--state-dir requires a value"
            state_dir="$2"
            shift 2
            ;;
        --skip-system-packages)
            skip_system_packages=true
            shift
            ;;
        --skip-gpu-check)
            skip_gpu_check=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
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

case "$profile" in
    core|examples|full) ;;
    *) die "profile must be one of: core, examples, full" ;;
esac

case "$state_dir" in
    /*) ;;
    *) state_dir="$PROJECT_ROOT/$state_dir" ;;
esac

[[ "$(uname -s)" == "Linux" ]] || die "Linux is required"
[[ "$(uname -m)" == "x86_64" ]] || die "x86_64 is required"
[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || die "pyproject.toml not found"
[[ -f "$PROJECT_ROOT/uv.lock" ]] || die "uv.lock not found"
[[ -f "$PROJECT_ROOT/.python-version" ]] || die ".python-version not found"

if [[ "$skip_gpu_check" == false ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    gpu_summary="$(nvidia-smi \
        --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader)" || die "nvidia-smi failed"
    driver_version="$(nvidia-smi \
        --query-gpu=driver_version \
        --format=csv,noheader | head -n 1)"
    driver_version="${driver_version//[[:space:]]/}"
    driver_major="${driver_version%%.*}"
    [[ "$driver_major" =~ ^[0-9]+$ ]] || die "cannot parse NVIDIA driver version"
    ((driver_major >= 580)) || die "NVIDIA driver 580 or newer is required"
    log "GPU preflight passed: $gpu_summary"
else
    log "GPU preflight skipped"
fi

missing_packages=()
for command_package in curl:curl git:git tmux:tmux rsync:rsync; do
    command_name="${command_package%%:*}"
    package_name="${command_package##*:}"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        missing_packages+=("$package_name")
    fi
done

if ((${#missing_packages[@]} > 0)); then
    if [[ "$skip_system_packages" == true ]]; then
        die "missing commands and system package installation is disabled: ${missing_packages[*]}"
    fi
    command -v apt-get >/dev/null 2>&1 || die "apt-get is required to install: ${missing_packages[*]}"

    if ((EUID == 0)); then
        privilege=()
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages"
        privilege=(sudo)
    fi

    if [[ "$dry_run" == true ]]; then
        log "would install system packages: ca-certificates ${missing_packages[*]}"
    else
        "${privilege[@]}" apt-get update
        "${privilege[@]}" env DEBIAN_FRONTEND=noninteractive \
            apt-get install -y ca-certificates "${missing_packages[@]}"
    fi
else
    log "required system commands are already available"
fi

python_version=""
IFS= read -r python_version < "$PROJECT_ROOT/.python-version"
python_version="${python_version//[[:space:]]/}"
[[ -n "$python_version" ]] || die ".python-version is empty"

log "project: $PROJECT_ROOT"
log "profile: $profile"
log "persistent state: $state_dir"
log "Python: $python_version"

if [[ "$dry_run" == true ]]; then
    log "would install uv if needed"
    log "would run uv python install and uv sync --locked for profile '$profile'"
    log "dry run completed"
    exit 0
fi

mkdir -p "$state_dir/uv-cache" "$state_dir/uv-python"
export UV_CACHE_DIR="$state_dir/uv-cache"
export UV_PYTHON_INSTALL_DIR="$state_dir/uv-python"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation did not provide an executable"

log "using $(uv --version)"
cd "$PROJECT_ROOT"
uv python install "$python_version"

sync_args=(sync --locked)
case "$profile" in
    core) ;;
    examples) sync_args+=(--extra examples) ;;
    full) sync_args+=(--all-extras) ;;
esac

start_seconds=$SECONDS
uv "${sync_args[@]}"
log "environment sync finished in $((SECONDS - start_seconds)) seconds"

if [[ "$skip_gpu_check" == false ]]; then
    uv run --locked --no-sync python -c \
        'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; name = torch.cuda.get_device_name(0); capability = torch.cuda.get_device_capability(0); bf16 = torch.cuda.is_bf16_supported(); assert bf16, "the selected GPU or PyTorch build does not support BF16"; print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {name}; SM {capability[0]}.{capability[1]}; BF16 {bf16}")'
fi

log "setup completed"
log "next: upload data/celeba, then run 'bash tool_scripts/cloud_gan.sh smoke'"
