"""Integrity checks for the data loaders (100-sample slice)."""

import pytest
from datasets import Dataset, DatasetDict

from data.halueval_loader import DEFAULT_TASKS, load_halueval_dataset_dict, load_halueval_task
from src.data.libreval_loader import DEFAULT_SPLITS, load_libreval_dataset_dict, load_libreval_split

LIMIT = 100


# ---------------------------------------------------------------------------
# HaluEval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", DEFAULT_TASKS)
def test_halueval_task_schema(task: str) -> None:
    ds = load_halueval_task(task, limit=LIMIT)
    assert isinstance(ds, Dataset)
    assert set(ds.column_names) == {"input_text", "label"}


@pytest.mark.parametrize("task", DEFAULT_TASKS)
def test_halueval_task_label_values(task: str) -> None:
    ds = load_halueval_task(task, limit=LIMIT)
    assert all(v in (0, 1) for v in ds["label"]), "Labels must be binary 0/1"


@pytest.mark.parametrize("task", DEFAULT_TASKS)
def test_halueval_task_nonempty_text(task: str) -> None:
    ds = load_halueval_task(task, limit=LIMIT)
    assert all(isinstance(t, str) and t.strip() for t in ds["input_text"]), "input_text must be non-empty strings"


def test_halueval_dataset_dict_combined() -> None:
    ddict = load_halueval_dataset_dict(limit_per_task=LIMIT, combine_tasks=True)
    assert isinstance(ddict, DatasetDict)
    assert list(ddict.keys()) == ["data"]
    assert set(ddict["data"].column_names) == {"input_text", "label"}
    assert len(ddict["data"]) > 0


def test_halueval_dataset_dict_per_task() -> None:
    ddict = load_halueval_dataset_dict(limit_per_task=LIMIT, combine_tasks=False)
    assert set(ddict.keys()) == set(DEFAULT_TASKS)
    for split in ddict.values():
        assert set(split.column_names) == {"input_text", "label"}


def test_halueval_limit_respected() -> None:
    ds = load_halueval_task("qa", limit=10)
    assert len(ds) <= 10


# ---------------------------------------------------------------------------
# LibreEval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", DEFAULT_SPLITS)
def test_libreval_split_schema(split: str) -> None:
    ds = load_libreval_split(split, limit=LIMIT)
    assert isinstance(ds, Dataset)
    assert set(ds.column_names) == {"input_text", "label"}


@pytest.mark.parametrize("split", DEFAULT_SPLITS)
def test_libreval_split_label_values(split: str) -> None:
    ds = load_libreval_split(split, limit=LIMIT)
    assert all(v in (0, 1) for v in ds["label"]), "Labels must be binary 0/1"


@pytest.mark.parametrize("split", DEFAULT_SPLITS)
def test_libreval_split_nonempty_text(split: str) -> None:
    ds = load_libreval_split(split, limit=LIMIT)
    assert all(isinstance(t, str) and t.strip() for t in ds["input_text"])


@pytest.mark.parametrize("split", DEFAULT_SPLITS)
def test_libreval_split_no_not_parsable(split: str) -> None:
    ds = load_libreval_split(split, limit=LIMIT)
    for text in ds["input_text"]:
        assert "NOT_PARSABLE" not in text


def test_libreval_dataset_dict_keys() -> None:
    ddict = load_libreval_dataset_dict(limit_per_split=LIMIT)
    assert set(ddict.keys()) == set(DEFAULT_SPLITS)
