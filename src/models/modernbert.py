"""ModernBERT sequence classifier for hallucination detection."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .base import ModelBundle

CHECKPOINT = "answerdotai/ModernBERT-base"
MAX_LENGTH = 8192  # Alternating local/global attention is pre-configured in the checkpoint; no extra setup needed.


def build_modernbert(
    checkpoint: str = CHECKPOINT,
    *,
    num_labels: int = 2,
) -> ModelBundle:
    """Load ModernBERT with a binary sequence classification head."""
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return ModelBundle(model=model, tokenizer=tokenizer, config=config)
