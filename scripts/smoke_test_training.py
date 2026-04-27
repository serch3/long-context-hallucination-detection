"""Short training smoke test: 1k samples, 2 epochs, DistilBERT, no GPU required."""

from datasets import DatasetDict
from src.data.halueval_loader import load_halueval_dataset_dict
from src.data.preprocess import preprocess_dataset_dict, build_tokenizer
from src.models.distilbert import build_distilbert, MAX_LENGTH
from src.training.trainer import TrainerConfig, build_trainer

raw = load_halueval_dataset_dict(tasks=["qa", "dialogue"], limit_per_task=250, combine_tasks=True)
data = raw["data"].train_test_split(test_size=0.2, seed=42)
print(f"Train: {len(data['train'])}  Eval: {len(data['test'])}")

tokenizer = build_tokenizer("distilbert-base-uncased")
tokenized = preprocess_dataset_dict(
    DatasetDict({"train": data["train"], "eval": data["test"]}),
    tokenizer,
    max_length=MAX_LENGTH,
    padding="longest",
)

bundle = build_distilbert()

cfg = TrainerConfig(
    output_dir="checkpoints/smoke_test",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    fp16=False,
    early_stopping_patience=2,
    report_to=[],
    logging_steps=20,
    dataloader_num_workers=0,
)

trainer = build_trainer(bundle, tokenized["train"], tokenized["eval"], cfg)
result = trainer.train()

log = trainer.state.log_history
train_losses = [e["loss"] for e in log if "loss" in e]
eval_losses = [e["eval_loss"] for e in log if "eval_loss" in e]

print(f"\nTrain loss  first->last: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
print(f"Eval  loss  first->last: {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}")
print(f"Loss decreased: {train_losses[-1] < train_losses[0]}")
print(f"Runtime: {result.metrics['train_runtime']:.1f}s")
