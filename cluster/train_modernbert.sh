#!/usr/bin/env bash
#SBATCH --job-name=halueval-modernbert
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/project/%u/long-context-hallucination-detection/results/logs/modernbert_%j.out
#SBATCH --error=/project/%u/long-context-hallucination-detection/results/logs/modernbert_%j.err

# Full ModernBERT training on HaluEval (long sequences up to 8192 tokens).
# Submit with: sbatch cluster/train_modernbert.sh
# Expected runtime: ~12-20 hours on one GPU (grad_accumulation_steps=8).
#
# If you hit OOM, increase gradient_accumulation_steps in configs/modernbert.yaml
# and halve per_device_train_batch_size, or request a second GPU:
#   #SBATCH --gres=gpu:2

set -euo pipefail

PROJ_DIR="/project/$USER/long-context-hallucination-detection"
ENV_PREFIX="/project/$USER/envs/hallucination_env"

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

export HF_HOME="/project/$USER/.cache/huggingface"
export PYTHONPATH="$PROJ_DIR"
TRAIN_ARGS=${TRAIN_ARGS:-""}

cd "$PROJ_DIR"

echo "==> Node     : $(hostname)"
echo "==> GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> Started  : $(date)"

python -m scripts.train \
    --model-config    configs/modernbert.yaml \
    --training-config configs/training.yaml \
    $TRAIN_ARGS

echo "==> Finished : $(date)"
