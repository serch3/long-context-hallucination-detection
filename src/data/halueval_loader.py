"""Load and normalize HaluEval tasks into ``input_text`` / ``label`` columns (0=factual, 1=hallucinated)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "halueval"
DEFAULT_TASKS: tuple[str, ...] = ("qa", "dialogue", "summarization", "general")

# Tasks where each source row expands into one factual (label=0) + one hallucinated (label=1) example.
_PAIRED_TASK_CONFIGS: dict[str, dict] = {
    "qa": {
        "context_keys": [("Knowledge", "knowledge"), ("Question", "question")],
        "answer_prefix": "Answer",
        "right_key": "right_answer",
        "hallucinated_key": "hallucinated_answer",
    },
    "dialogue": {
        "context_keys": [("Knowledge", "knowledge"), ("Dialogue history", "dialogue_history")],
        "answer_prefix": "Response",
        "right_key": "right_response",
        "hallucinated_key": "hallucinated_response",
    },
    "summarization": {
        "context_keys": [("Document", "document")],
        "answer_prefix": "Summary",
        "right_key": "right_summary",
        "hallucinated_key": "hallucinated_summary",
    },
}


def _normalize_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"Expected binary label, got {value!r}.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "hallucinated", "hallucination", "hallucinated response"}:
            return 1
        if normalized in {"0", "false", "no", "factual", "grounded", "not hallucinated", "non-hallucinated"}:
            return 0
    raise ValueError(f"Could not normalize label value {value!r}.")


def _format_example(parts: Iterable[str]) -> str:
    return "\n".join(part for part in parts if part and part.strip())


def _build_examples_for_task(task_name: str, dataset: Dataset) -> Dataset:
    if task_name == "general":
        return Dataset.from_list([
            {
                "input_text": _format_example((
                    f"User query: {str(ex['user_query']).strip()}",
                    f"Response: {str(ex['chatgpt_response']).strip()}",
                )),
                "label": _normalize_label(ex["hallucination"]),
            }
            for ex in dataset
        ])

    cfg = _PAIRED_TASK_CONFIGS.get(task_name)
    if cfg is None:
        raise ValueError(f"Unsupported HaluEval task: {task_name!r}.")

    rows: list[dict[str, object]] = []
    for ex in dataset:
        context_parts = [f"{prefix}: {str(ex[field]).strip()}" for prefix, field in cfg["context_keys"]]
        for answer_key, label in [(cfg["right_key"], 0), (cfg["hallucinated_key"], 1)]:
            rows.append({
                "input_text": _format_example(context_parts + [f"{cfg['answer_prefix']}: {str(ex[answer_key]).strip()}"]),
                "label": label,
            })
    return Dataset.from_list(rows)


def _load_task_source(task_name: str, data_dir: Path) -> Dataset:
    local_path = data_dir / task_name
    if not local_path.exists():
        raise FileNotFoundError(f"HaluEval task '{task_name}' not found at {local_path}. Run scripts/download_data.sh first.")
    loaded = load_from_disk(str(local_path))
    if isinstance(loaded, DatasetDict):
        if len(loaded) != 1:
            raise ValueError(f"Expected one split in {local_path}, found {list(loaded.keys())!r}.")
        return loaded[next(iter(loaded))]
    return loaded


def load_halueval_task(
    task_name: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    limit: int | None = None,
) -> Dataset:
    """Load one HaluEval task and normalize it to ``input_text`` / ``label``."""

    source_dataset = _load_task_source(task_name, Path(data_dir))
    normalized_dataset = _build_examples_for_task(task_name, source_dataset)
    if limit is not None:
        normalized_dataset = normalized_dataset.select(range(min(limit, len(normalized_dataset))))
    return normalized_dataset


def load_halueval_dataset_dict(
    *,
    tasks: Iterable[str] = DEFAULT_TASKS,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    limit_per_task: int | None = None,
    combine_tasks: bool = True,
) -> DatasetDict:
    """Load one or more HaluEval tasks into a normalized ``DatasetDict``.

    When ``combine_tasks`` is ``True``, all requested tasks are concatenated into
    a single ``data`` split. When it is ``False``, each task is returned as its
    own split.
    """

    task_names = tuple(tasks)
    normalized_splits = {
        task_name: load_halueval_task(task_name, data_dir=data_dir, limit=limit_per_task)
        for task_name in task_names
    }

    if combine_tasks:
        combined = concatenate_datasets(list(normalized_splits.values())) if normalized_splits else Dataset.from_dict({"input_text": [], "label": []})
        return DatasetDict({"data": combined})

    return DatasetDict(normalized_splits)
