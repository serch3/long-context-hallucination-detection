#!/usr/bin/env bash
# scripts/download_data.sh
# Usage: bash scripts/download_data.sh [halueval|libreval|all]
#
# Downloads dataset(s) from Hugging Face and saves them to data/raw/{name}/
# from the project root. Run from any directory — paths are always absolute.

set -euo pipefail

DATASET=${1:-halueval}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"

download_halueval() {
    local save_path="$RAW_DIR/halueval"
    if [ -d "$save_path" ]; then
        echo "==> HaluEval already exists at $save_path — skipping."
        return
    fi
    echo "==> Downloading HaluEval → $save_path"
    mkdir -p "$save_path"
    python3 - <<PYEOF
from datasets import load_dataset
dataset = load_dataset("pminervini/HaluEval")
dataset.save_to_disk("$save_path")
print(f"  Saved splits: {list(dataset.keys())}")
PYEOF
    echo "✅ HaluEval saved to $save_path"
}

download_libreval() {
    local save_path="$RAW_DIR/libreval"
    if [ -d "$save_path" ]; then
        echo "==> LibreEval already exists at $save_path — skipping."
        return
    fi
    echo "==> Downloading LibreEval → $save_path"
    mkdir -p "$save_path"
    python3 - <<PYEOF
from datasets import load_dataset
dataset = load_dataset("Arize-ai/LibreEval")
dataset.save_to_disk("$save_path")
print(f"  Saved splits: {list(dataset.keys())}")
PYEOF
    echo "✅ LibreEval saved to $save_path"
}

case "$DATASET" in
    halueval)  download_halueval ;;
    libreval)  download_libreval ;;
    all)       download_halueval; download_libreval ;;
    *)
        echo "❌ Unknown dataset: '$DATASET'"
        echo "   Usage: bash scripts/download_data.sh [halueval|libreval|all]"
        exit 1
        ;;
esac

echo ""
echo "Data directory contents:"
ls -lh "$RAW_DIR"
