#!/usr/bin/env bash
#SBATCH --job-name=halueval-distilbert
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/%u/long-context-hallucination-detection/results/logs/distilbert_%j.out
#SBATCH --error=/project/%u/long-context-hallucination-detection/results/logs/distilbert_%j.err

# Full DistilBERT training on HaluEval.
# Submit with: sbatch cluster/train_distilbert.sh
# Expected runtime: ~4-6 hours on one GPU.

set -euo pipefail

# Use project-level caches to avoid $HOME quota issues
export XDG_CACHE_HOME="/project/$USER/.cache"
export TRITON_CACHE_DIR="/project/$USER/.cache/triton"
export PIP_CACHE_DIR="/project/$USER/.cache/pip"
export TORCH_EXTENSIONS_DIR="/project/$USER/.cache/torch_extensions"
export HF_HOME="/project/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/project/$USER/.cache/huggingface"
export CONDA_ENVS_PATH="/project/$USER/conda_envs"
export CONDA_PKGS_DIRS="/project/$USER/conda_pkgs"

PROJ_DIR="/project/$USER/long-context-hallucination-detection"
ENV_PREFIX="/project/$USER/envs/hallucination_env"

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJ_DIR"
TRAIN_ARGS=${TRAIN_ARGS:-""}

cd "$PROJ_DIR"

echo "==> Node     : $(hostname)"
echo "==> Started  : $(date)"

# ── Preflight: abort if GPU or environment is broken ─────────────────────────
# shellcheck source=cluster/_preflight.sh
source "$PROJ_DIR/cluster/_preflight.sh"

python -m scripts.train \
    --model-config    configs/distilbert.yaml \
    --training-config configs/training.yaml \
    $TRAIN_ARGS

echo "==> Finished : $(date)"
