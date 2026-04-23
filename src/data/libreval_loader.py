"""Load and normalize LibreEval splits into ``input_text`` / ``label`` columns (0=factual, 1=hallucinated)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset

from .halueval_loader import _format_example, _normalize_label

DEFAULT_LIBREVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "libreval"

# Map split name → path relative to DEFAULT_LIBREVAL_DIR.
_SPLIT_PATHS: dict[str, Path] = {
    "non_synthetic": Path("combined_datasets_for_evals/non_synthetic_hallucinations_english.csv"),
    "synthetic":     Path("combined_datasets_for_evals/synthetic_hallucinations_english.csv"),
    "tuning_train":  Path("combined_datasets_for_tuning/english_only/balanced_hallucinations.csv"),
    "tuning_test":   Path("combined_datasets_for_tuning/english_only/test.csv"),
}

DEFAULT_SPLITS: tuple[str, ...] = tuple(_SPLIT_PATHS)


def _build_libreval_examples(dataset: Dataset) -> Dataset:
    rows: list[dict[str, object]] = []
    for ex in dataset:
        if str(ex["label"]).strip().upper() == "NOT_PARSABLE":
            continue
        rows.append({
            "input_text": _format_example((
                f"Reference: {str(ex['reference']).strip()}",
                f"Question: {str(ex['input']).strip()}",
                f"Response: {str(ex['output']).strip()}",
            )),
            "label": _normalize_label(ex["label"]),
        })
    return Dataset.from_list(rows)


def load_libreval_split(
    split_name: str,
    *,
    data_dir: str | Path = DEFAULT_LIBREVAL_DIR,
    limit: int | None = None,
) -> Dataset:
    """Load one LibreEval split and normalize it to ``input_text`` / ``label``."""

    if split_name not in _SPLIT_PATHS:
        raise ValueError(f"Unknown LibreEval split {split_name!r}. Choose from {list(_SPLIT_PATHS)}.")

    csv_path = Path(data_dir) / _SPLIT_PATHS[split_name]
    if not csv_path.exists():
        raise FileNotFoundError(f"LibreEval CSV not found at {csv_path}. Run scripts/download_data.sh first.")

    raw = load_dataset("csv", data_files=str(csv_path), split="train")
    normalized = _build_libreval_examples(raw)
    if limit is not None:
        normalized = normalized.select(range(min(limit, len(normalized))))
    return normalized


def load_libreval_dataset_dict(
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    data_dir: str | Path = DEFAULT_LIBREVAL_DIR,
    limit_per_split: int | None = None,
) -> DatasetDict:
    """Load one or more LibreEval splits into a normalized ``DatasetDict``.

    Each requested split becomes its own key in the returned dict.
    """

    return DatasetDict({
        split_name: load_libreval_split(split_name, data_dir=data_dir, limit=limit_per_split)
        for split_name in splits
    })
