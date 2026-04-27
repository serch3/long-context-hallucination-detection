"""Training entry-point: reads YAML configs and runs end-to-end training.

Usage (local):
    python -m scripts.train --model-config configs/distilbert.yaml

Usage (cluster):
    sbatch cluster/train_distilbert.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from datasets import DatasetDict

from src.data.halueval_loader import DEFAULT_TASKS, load_halueval_dataset_dict
from src.data.preprocess import build_tokenizer, preprocess_dataset_dict
from src.models.distilbert import build_distilbert
from src.models.modernbert import build_modernbert
from src.training.trainer import build_trainer, load_config

_MODEL_BUILDERS = {
    "distilbert-base-uncased": build_distilbert,
    "answerdotai/ModernBERT-base": build_modernbert,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a classifier on HaluEval.")
    p.add_argument("--model-config", required=True, help="Model YAML (e.g. configs/distilbert.yaml)")
    p.add_argument("--training-config", default="configs/training.yaml", help="Shared training YAML")
    p.add_argument("--data-dir", default=None, help="Override path to raw HaluEval data")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="HaluEval tasks to include")
    p.add_argument("--eval-split", type=float, default=0.1, help="Fraction held out for eval")
    p.add_argument("--limit", type=int, default=None, help="Cap examples per task (for smoke tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model_config) as f:
        model_yaml: dict = yaml.safe_load(f)

    model_name: str = model_yaml["model_name_or_path"]
    max_length: int = model_yaml["max_length"]

    cfg = load_config(args.model_config, args.training_config)

    loader_kwargs: dict = {}
    if args.data_dir:
        loader_kwargs["data_dir"] = args.data_dir

    print(f"Loading HaluEval tasks: {args.tasks}")
    raw = load_halueval_dataset_dict(
        tasks=args.tasks,
        limit_per_task=args.limit,
        combine_tasks=True,
        **loader_kwargs,
    )
    split = raw["data"].train_test_split(test_size=args.eval_split, seed=cfg.seed)

    print(f"Train: {len(split['train'])}  Eval: {len(split['test'])}")
    print(f"Tokenizing with {model_name} (max_length={max_length})...")

    tokenizer = build_tokenizer(model_name)
    tokenized = preprocess_dataset_dict(
        DatasetDict({"train": split["train"], "eval": split["test"]}),
        tokenizer,
        max_length=max_length,
        padding="longest",
    )

    builder = _MODEL_BUILDERS.get(model_name)
    if builder is None:
        sys.exit(f"Unknown model '{model_name}'. Add it to _MODEL_BUILDERS in scripts/train.py.")
    bundle = builder()

    trainer = build_trainer(bundle, tokenized["train"], tokenized["eval"], cfg)

    print(f"Starting training — output dir: {cfg.output_dir}")
    trainer.train()
    print(f"Done. Checkpoint saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
