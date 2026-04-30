#!/usr/bin/env bash
# cluster/_preflight.sh — shared GPU & environment validation.
# Source this at the top of every cluster training script after conda activate.
#
# Checks:
#   1. Conda environment is activated and has a working Python
#   2. PyTorch can see CUDA
#   3. nvidia-smi reports at least one GPU
#   4. Critical packages are importable

set -euo pipefail

# ── 1. Python exists ─────────────────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    echo "FATAL: python not found in PATH — conda activate may have failed." >&2
    echo "       Expected env: $ENV_PREFIX" >&2
    exit 1
fi

PYTHON_BIN="$(which python)"
echo "==> Python   : $PYTHON_BIN"

# ── 2. PyTorch + CUDA ────────────────────────────────────────────────────────
CUDA_OK=$(python -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>&1) || {
    echo "FATAL: Failed to import torch. Is the environment set up correctly?" >&2
    echo "       Run: bash cluster/setup_env.sh" >&2
    exit 1
}

if [[ "$CUDA_OK" != "YES" ]]; then
    echo "FATAL: PyTorch cannot see CUDA." >&2
    echo "       torch.cuda.is_available() returned False." >&2
    echo "       This means training would run on CPU — aborting." >&2
    echo "       Check: sbatch --gres=gpu:1 and that the PyTorch build is CUDA-enabled." >&2
    exit 1
fi

echo "==> CUDA     : available"

# ── 3. nvidia-smi ────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not found — no GPU driver visible." >&2
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "FATAL: nvidia-smi reports 0 GPUs." >&2
    exit 1
fi

echo "==> GPUs     : $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read -r line; do
    echo "               $line"
done

# ── 4. Critical packages ─────────────────────────────────────────────────────
python -c "import transformers, datasets, accelerate" 2>&1 || {
    echo "FATAL: Missing critical packages (transformers, datasets, accelerate)." >&2
    echo "       Run: bash cluster/setup_env.sh" >&2
    exit 1
}

echo "==> Packages : OK"
echo "==> Preflight checks passed."