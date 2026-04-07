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
    if [ -d "$save_path" ] && [ -n "$(ls -A "$save_path" 2>/dev/null)" ]; then
        echo "==> HaluEval already exists at $save_path — skipping."
        return
    fi
    echo "==> Downloading HaluEval → $save_path"
    mkdir -p "$save_path"
    python3 - <<PYEOF
from datasets import load_dataset

configs = ["dialogue", "dialogue_samples", "general", "qa", "qa_samples", "summarization", "summarization_samples"]
for config in configs:
    print(f"  Downloading config: {config}")
    dataset = load_dataset("pminervini/HaluEval", config)
    dataset.save_to_disk(f"$save_path/{config}")
    print(f"    Saved splits: {list(dataset.keys())}")
PYEOF
    echo "✅ HaluEval saved to $save_path"
}

download_libreval() {
    local save_path="$RAW_DIR/libreval"
    if [ -d "$save_path" ] && [ -n "$(ls -A "$save_path" 2>/dev/null)" ]; then
        echo "==> LibreEval already exists at $save_path — skipping."
        return
    fi
    echo "==> Downloading LibreEval → $save_path"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    git clone --depth=1 https://github.com/Arize-ai/LibreEval.git "$tmp_dir/LibreEval"
    mkdir -p "$save_path"
    cp -r "$tmp_dir/LibreEval/labeled_datasets" "$save_path/"
    cp -r "$tmp_dir/LibreEval/combined_datasets_for_evals" "$save_path/"
    cp -r "$tmp_dir/LibreEval/combined_datasets_for_tuning" "$save_path/"
    rm -rf "$tmp_dir"
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
