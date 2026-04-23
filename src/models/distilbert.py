"""DistilBERT sequence classifier for hallucination detection."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .base import ModelBundle

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 512


def build_distilbert(
    checkpoint: str = CHECKPOINT,
    *,
    num_labels: int = 2,
) -> ModelBundle:
    """Load DistilBERT with a binary sequence classification head."""
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return ModelBundle(model=model, tokenizer=tokenizer, config=config)
