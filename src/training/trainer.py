"""HF Trainer factory for hallucination detection classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.models.base import ModelBundle
from src.evaluation.metrics import compute_metrics


@dataclass
class TrainerConfig:
    output_dir: str
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    report_to: list[str] = field(default_factory=list)
    run_name: Optional[str] = None
    logging_steps: int = 50
    dataloader_num_workers: int = 4
    seed: int = 42
    gradient_accumulation_steps: int = 1


def load_config(model_yaml: str | Path, training_yaml: str | Path | None = None) -> TrainerConfig:
    """Merge training.yaml defaults with a model-specific YAML and return a TrainerConfig."""
    merged: dict[str, Any] = {}

    if training_yaml is not None:
        with open(training_yaml) as f:
            merged.update(yaml.safe_load(f) or {})

    with open(model_yaml) as f:
        merged.update(yaml.safe_load(f) or {})

    # Strip keys not part of TrainerConfig (e.g. model_name_or_path, max_length)
    valid = {f.name for f in TrainerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return TrainerConfig(**{k: v for k, v in merged.items() if k in valid})


def build_training_args(cfg: TrainerConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        logging_steps=cfg.logging_steps,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # GPU safeguards
        no_cuda=False,
        dataloader_pin_memory=True,
    )


def build_trainer(
    bundle: ModelBundle,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: TrainerConfig,
) -> Trainer:
    """Wire together a model bundle and datasets into a ready-to-run Trainer."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(cfg)

    # Use dynamic padding so sequences within each batch are padded to the
    # longest example rather than the global max_length.
    data_collator = DataCollatorWithPadding(tokenizer=bundle.tokenizer)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]

    return Trainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
