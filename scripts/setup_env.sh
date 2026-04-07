#!/usr/bin/env bash
# scripts/setup_env.sh
# Usage: bash scripts/setup_env.sh [env_name]
#
# Creates (or updates) a conda environment and installs all dependencies
# from requirements.txt. Run from the project root.

set -euo pipefail

ENV_NAME=${1:-hallucination_env}
PYTHON_VERSION=${2:-3.11}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Project root: $REPO_ROOT"
echo "==> Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"

# ── 1. Create or update environment ─────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Environment '$ENV_NAME' already exists — skipping creation."
else
    echo "==> Creating environment..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# ── 2. Install pip dependencies ──────────────────────────────────────────────
echo "==> Installing dependencies from requirements.txt..."
CONDA_ENV_PATH="$(conda env list | awk -v env="$ENV_NAME" '$1 == env {print $NF}')"
if [[ -z "$CONDA_ENV_PATH" ]]; then
    echo "ERROR: Could not locate conda environment '$ENV_NAME'" >&2
    exit 1
fi
"$CONDA_ENV_PATH/bin/pip" install --upgrade pip
"$CONDA_ENV_PATH/bin/pip" install -r "$REPO_ROOT/requirements.txt"

# ── 3. Register Jupyter kernel ───────────────────────────────────────────────
echo "==> Registering Jupyter kernel..."
"$CONDA_ENV_PATH/bin/python" -m ipykernel install \
    --user \
    --name="$ENV_NAME" \
    --display-name "HallucinationEnv (Python $PYTHON_VERSION)"

# ── 4. Create required directories ──────────────────────────────────────────
echo "==> Creating project directories..."
mkdir -p "$REPO_ROOT/data/raw" \
         "$REPO_ROOT/data/processed" \
         "$REPO_ROOT/data/interim" \
         "$REPO_ROOT/results/logs" \
         "$REPO_ROOT/results/metrics" \
         "$REPO_ROOT/results/plots" \
         "$REPO_ROOT/results/tables" \
         "$REPO_ROOT/checkpoints"

echo ""
echo "✅ Setup complete!"
echo "   Activate with:  conda activate $ENV_NAME"
echo "   Download data:  bash scripts/download_data.sh [halueval|libreval]"
