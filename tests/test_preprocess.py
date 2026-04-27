"""Integrity checks for the tokenization preprocessor (100-sample slice)."""

import pytest
from datasets import Dataset

from data.halueval_loader import load_halueval_task
from src.data.preprocess import build_tokenizer, tokenize_dataset

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 64
LIMIT = 100


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(CHECKPOINT)


@pytest.fixture(scope="module")
def raw_dataset() -> Dataset:
    return load_halueval_task("qa", limit=LIMIT)


@pytest.fixture(scope="module")
def tokenized(tokenizer, raw_dataset) -> Dataset:
    return tokenize_dataset(raw_dataset, tokenizer, max_length=MAX_LENGTH)


def test_output_columns(tokenized: Dataset) -> None:
    assert "input_ids" in tokenized.column_names
    assert "attention_mask" in tokenized.column_names
    assert "labels" in tokenized.column_names, "Column must be 'labels' (plural) for HF Trainer"
    assert "label" not in tokenized.column_names, "'label' must be renamed to 'labels'"
    assert "input_text" not in tokenized.column_names


def test_sequence_length(tokenized: Dataset) -> None:
    for ids in tokenized["input_ids"]:
        assert len(ids) == MAX_LENGTH, f"Expected {MAX_LENGTH} tokens, got {len(ids)}"


def test_label_values_preserved(raw_dataset: Dataset, tokenized: Dataset) -> None:
    assert tokenized["labels"] == raw_dataset["label"], "Labels must survive tokenization unchanged"


def test_row_count_preserved(raw_dataset: Dataset, tokenized: Dataset) -> None:
    assert len(tokenized) == len(raw_dataset)
