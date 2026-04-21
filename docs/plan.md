# Project Plan: Long-Context Hallucination Detection

**Team:** Sergio Mandujano · Garrett Sueing  
**Goal:** Fine-tune DistilBERT (baseline) and ModernBERT (main) on HaluEval to classify hallucinated vs. factual LLM outputs.

---

## Modules

The project is split into five modules. Each can be developed and tested in isolation before wiring together via `scripts/run_experiment.py`.

| # | Module | Owner | Key files |
|---|--------|-------|-----------|
| M1 | Data pipeline | — | `src/data/` |
| M2 | Model wrappers | — | `src/models/` |
| M3 | Training loop | — | `src/training/` |
| M4 | Evaluation | — | `src/evaluation/` |
| M5 | Experiment orchestration | — | `scripts/` |

---

### M1 — Data Pipeline
1. **EDA** (`notebooks/eda.ipynb`): inspect dataset splits (qa, dialogue, summarization), label balance, token length distribution.
2. **Loader** (`src/data/loader.py`): wrap HF `datasets` to return a unified `DatasetDict` with columns `[input_text, label]`.
3. **Preprocessor** (`src/data/preprocess.py`): combine context + response, tokenize, truncate to 512 (DistilBERT) / 8192 (ModernBERT), write to `data/processed/`.

**Done when:** `loader.py` and `preprocess.py` each have unit tests that pass on a 100-sample slice (integrity check not training).

---

### M2 — Model Wrappers
1. **DistilBERT** (`src/models/distilbert.py`): sequence classification head on `distilbert-base-uncased`; expose `model`, `tokenizer`, `config`.
2. **ModernBERT** (`src/models/modernbert.py`): same interface on `answerdotai/ModernBERT-base`; enable sliding-window attention for long sequences.

**Done when:** both wrappers do a forward pass on a dummy batch without error.

---

### M3 — Training Loop
1. **Trainer** (`src/training/trainer.py`): HF `Trainer` with early stopping, fp16, and WandB logging; linear warmup + cosine decay via `TrainingArguments`.
2. **Config files** (`configs/distilbert.yaml`, `configs/modernbert.yaml`, `configs/training.yaml`): learning rate, batch size, epochs, max tokens.

**Done when:** DistilBERT trains to completion on a 1k-sample subset; loss decreases.

---

### M4 — Evaluation
1. **Metrics** (`src/evaluation/metrics.py`): accuracy, precision, recall, F1, AUROC via HF `evaluate`.
2. **Evaluator** (`src/evaluation/evaluator.py`): loads a checkpoint, runs on the test split, writes a JSON report to `results/metrics/`.
3. **Error analysis** (`src/evaluation/error_analysis.py`, `notebooks/error_analysis.ipynb`): false-positive/negative inspection bucketed by input length.

**Done when:** evaluator produces a reproducible JSON report for a saved checkpoint.

---

### M5 — Experiment Orchestration
1. **Run experiment** (`scripts/run_experiment.py`): CLI entry that wires M1–M4 together; reads a config YAML.
2. **Compare models** (`scripts/compare_models.py`): load both result JSONs, produce comparison table + plots in `results/`.

**Done when:** one command reproduces all reported numbers from a single config file.
