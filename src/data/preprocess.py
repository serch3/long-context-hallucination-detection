"""Tokenization and persistence helpers for normalized HaluEval datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .halueval_loader import DEFAULT_DATA_DIR, DEFAULT_TASKS, load_halueval_dataset_dict

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_tokenizer(model_name_or_path: str, *, use_fast: bool = True) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    text_column: str = "input_text",
    padding: str = "max_length",
) -> Dataset:
    """Tokenize ``text_column``, drop all original columns, and rename ``label`` → ``labels`` for HF Trainer."""

    def tokenize_batch(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding=padding,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns([c for c in dataset.column_names if c != "label"])
    return tokenized_dataset.rename_column("label", "labels")


def preprocess_dataset_dict(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    text_column: str = "input_text",
    padding: str = "max_length",
) -> DatasetDict:
    return DatasetDict(
        {
            split_name: tokenize_dataset(
                split_dataset,
                tokenizer,
                max_length=max_length,
                text_column=text_column,
                padding=padding,
            )
            for split_name, split_dataset in dataset_dict.items()
        }
    )


def preprocess_halueval(
    model_name_or_path: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    tasks: Iterable[str] = DEFAULT_TASKS,
    limit_per_task: int | None = None,
    combine_tasks: bool = True,
    max_length: int = 512,
    use_fast_tokenizer: bool = True,
    save_to_disk: bool = True,
) -> DatasetDict:
    """Load HaluEval, tokenize it, and optionally persist the processed result."""

    tokenizer = build_tokenizer(model_name_or_path, use_fast=use_fast_tokenizer)
    dataset_dict = load_halueval_dataset_dict(
        tasks=tasks,
        data_dir=data_dir,
        limit_per_task=limit_per_task,
        combine_tasks=combine_tasks,
    )
    processed_dataset_dict = preprocess_dataset_dict(dataset_dict, tokenizer, max_length=max_length)

    if save_to_disk:
        output_path = Path(processed_dir) / model_name_or_path.replace("/", "__") / f"maxlen_{max_length}"
        output_path.mkdir(parents=True, exist_ok=True)
        processed_dataset_dict.save_to_disk(str(output_path))

    return processed_dataset_dict
