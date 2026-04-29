"""Training entry-point: reads YAML configs and runs end-to-end training.

Usage (local):
    python -m scripts.train --model-config configs/distilbert.yaml

Usage (cluster):
    sbatch cluster/train_distilbert.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets

from src.data.halueval_loader import DEFAULT_TASKS, load_halueval_dataset_dict
from src.data.libreval_loader import DEFAULT_SPLITS, load_libreval_dataset_dict
from src.data.preprocess import build_tokenizer, preprocess_dataset_dict
from src.models.distilbert import build_distilbert
from src.models.modernbert import build_modernbert
from src.training.trainer import build_trainer, load_config

_MODEL_BUILDERS = {
    "distilbert-base-uncased": build_distilbert,
    "answerdotai/ModernBERT-base": build_modernbert,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a classifier on HaluEval, LibreEval, or both.")
    p.add_argument("--model-config", required=True, help="Model YAML (e.g. configs/distilbert.yaml)")
    p.add_argument("--training-config", default="configs/training.yaml", help="Shared training YAML")
    p.add_argument(
        "--dataset",
        choices=["halueval", "libreval", "both"],
        default="halueval",
        help="Which dataset source to train on.",
    )
    p.add_argument("--data-dir", default=None, help="Deprecated alias for --halueval-data-dir")
    p.add_argument("--halueval-data-dir", default=None, help="Override path to raw HaluEval data")
    p.add_argument("--libreval-data-dir", default=None, help="Override path to raw LibreEval data")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="HaluEval tasks to include")
    p.add_argument("--libreval-splits", nargs="+", default=list(DEFAULT_SPLITS), help="LibreEval splits to include")
    p.add_argument("--eval-split", type=float, default=0.1, help="Fraction held out for eval")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples per HaluEval task / LibreEval split (for smoke tests)",
    )
    return p.parse_args()


def _load_halueval_data(args: argparse.Namespace) -> Dataset:
    data_dir = args.halueval_data_dir or args.data_dir
    loader_kwargs: dict = {}
    if data_dir:
        loader_kwargs["data_dir"] = data_dir

    print(f"Loading HaluEval tasks: {args.tasks}")
    halueval = load_halueval_dataset_dict(
        tasks=args.tasks,
        limit_per_task=args.limit,
        combine_tasks=True,
        **loader_kwargs,
    )
    return halueval["data"]


def _load_libreval_data(args: argparse.Namespace) -> Dataset:
    loader_kwargs: dict = {}
    if args.libreval_data_dir:
        loader_kwargs["data_dir"] = args.libreval_data_dir

    print(f"Loading LibreEval splits: {args.libreval_splits}")
    libreval = load_libreval_dataset_dict(
        splits=args.libreval_splits,
        limit_per_split=args.limit,
        **loader_kwargs,
    )
    return concatenate_datasets(list(libreval.values()))


def _load_training_data(args: argparse.Namespace) -> Dataset:
    if args.dataset == "halueval":
        data = _load_halueval_data(args)
        print(f"Loaded dataset=halueval with {len(data)} normalized examples")
        return data

    if args.dataset == "libreval":
        data = _load_libreval_data(args)
        print(f"Loaded dataset=libreval with {len(data)} normalized examples")
        return data

    halueval_data = _load_halueval_data(args)
    libreval_data = _load_libreval_data(args)
    merged = concatenate_datasets([halueval_data, libreval_data])
    print(
        "Loaded dataset=both with "
        f"halueval={len(halueval_data)}, libreval={len(libreval_data)}, total={len(merged)} examples"
    )
    return merged


def main() -> None:
    args = parse_args()

    with open(args.model_config) as f:
        model_yaml: dict = yaml.safe_load(f)

    model_name: str = model_yaml["model_name_or_path"]
    max_length: int = model_yaml["max_length"]

    cfg = load_config(args.model_config, args.training_config)

    # Update run_name to include dataset if not the default HaluEval
    if cfg.run_name and args.dataset != "halueval":
        base_name = cfg.run_name.replace("-halueval", "")
        cfg.run_name = f"{base_name}-{args.dataset}"

    all_data = _load_training_data(args)
    split = all_data.train_test_split(test_size=args.eval_split, seed=cfg.seed)

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
    final_metrics = trainer.evaluate(eval_dataset=tokenized["eval"])

    metrics_dir = Path("results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_label = cfg.run_name or Path(cfg.output_dir).name
    metrics_path = metrics_dir / f"{run_label}_{args.dataset}_eval_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "model_name_or_path": model_name,
                "dataset": args.dataset,
                "halueval_tasks": args.tasks,
                "libreval_splits": args.libreval_splits,
                "eval_split": args.eval_split,
                "seed": cfg.seed,
                "output_dir": cfg.output_dir,
                "metrics": final_metrics,
            },
            f,
            indent=2,
        )

    print(f"Evaluation metrics saved to: {metrics_path}")
    print(f"Done. Checkpoint saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
