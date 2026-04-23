"""Shared types for model wrappers."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig


@dataclass
class ModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    config: PretrainedConfig
