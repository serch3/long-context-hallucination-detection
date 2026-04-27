#!/usr/bin/env bash
# cluster/setup_env.sh — one-time environment setup on the iTiger cluster.
# Run from the project root: bash cluster/setup_env.sh
#
# Prerequisites:
#   - Repo cloned to /project/$USER/long-context-hallucination-detection
#   - Conda available in PATH (Miniconda is pre-installed on iTiger)
#
# Everything is kept under /project/$USER/ to avoid the 10 GB $HOME quota:
#   - Conda env  → /project/$USER/envs/hallucination_env  (~8-10 GB with PyTorch)
#   - HF cache   → /project/$USER/.cache/huggingface      (~1-3 GB per model)
#   - Data/ckpts → /project/$USER/long-context-hallucination-detection/

set -euo pipefail

ENV_NAME=${1:-hallucination_env}
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="/project/$USER/envs/$ENV_NAME"

echo "==> Project dir : $PROJ_DIR"
echo "==> Conda env   : $ENV_PREFIX"

# Initialise conda for non-interactive (batch) shells
eval "$(conda shell.bash hook)"

# ── 1. Create environment under /project (avoids $HOME quota) ────────────────
if [ -d "$ENV_PREFIX" ]; then
    echo "==> Environment already exists — skipping creation."
else
    echo "==> Creating Python 3.11 environment under /project..."
    mkdir -p "/project/$USER/envs"
    conda create -y --prefix "$ENV_PREFIX" python=3.11
fi

conda activate "$ENV_PREFIX"

# ── 2. PyTorch with CUDA 12.4 (matches iTiger driver) ────────────────────────
echo "==> Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ── 3. Rest of project dependencies ──────────────────────────────────────────
# torch is already satisfied above; pip will skip re-installing it.
echo "==> Installing project dependencies..."
pip install -r "$PROJ_DIR/requirements.txt"

# ── 4. Create project directories ────────────────────────────────────────────
mkdir -p "$PROJ_DIR/data/raw" \
         "$PROJ_DIR/data/processed" \
         "$PROJ_DIR/results/logs" \
         "$PROJ_DIR/results/metrics" \
         "$PROJ_DIR/checkpoints"

echo ""
echo "==> Setup complete."
echo "    Activate with : conda activate $ENV_PREFIX"
echo "    Download data : bash scripts/download_data.sh"
echo "    Submit jobs   : sbatch cluster/train_distilbert.sh"
echo "                    sbatch cluster/train_modernbert.sh"
