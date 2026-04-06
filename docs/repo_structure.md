# Repo Structure (Draft)

```text
long-context-hallucination-detection/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- configs/                    # experiment + model configs
|   |-- distilbert.yaml
|   |-- modernbert.yaml
|   |-- training.yaml
|   `-- paths.yaml
|
|-- data/
|   |-- raw/                    # original datasets (HaluEval, LibreEval)
|   |-- processed/              # tokenized / cleaned data
|   |-- interim/                # intermediate transformations
|   `-- README.md
|
|-- notebooks/                  # exploration / prototyping
|   |-- eda.ipynb
|   `-- error_analysis.ipynb
|
|-- src/
|   |-- __init__.py
|   |
|   |-- data/
|   |   |-- loader.py           # load datasets (HF datasets, CSV, etc.)
|   |   |-- preprocess.py       # cleaning, truncation, chunking long context
|   |   `-- collator.py         # batching logic (important for long context)
|   |
|   |-- models/
|   |   |-- distilbert.py       # baseline model wrapper
|   |   |-- modernbert.py       # main model wrapper
|   |   `-- utils.py            # shared model utilities
|   |
|   |-- training/
|   |   |-- trainer.py          # training loop (HF Trainer or custom)
|   |   |-- loss.py             # custom loss if needed
|   |   `-- scheduler.py
|   |
|   |-- evaluation/
|   |   |-- metrics.py          # accuracy, F1, AUROC
|   |   |-- evaluator.py        # evaluation pipeline
|   |   `-- error_analysis.py   # hallucination patterns
|   |
|   |-- inference/
|   |   |-- predict.py          # run model on new text
|   |   `-- pipeline.py         # full inference pipeline
|   |
|   |-- utils/
|   |   |-- logging.py
|   |   |-- seed.py
|   |   `-- helpers.py
|   |
|   `-- experiments/
|       |-- run_experiment.py   # main entrypoint
|       `-- compare_models.py   # distilbert vs modernbert
|
|-- scripts/                    # CLI scripts for reproducibility
|   |-- download_data.sh
|   |-- preprocess.py
|   |-- train_distilbert.py
|   |-- train_modernbert.py
|   |-- evaluate.py
|   `-- benchmark.py
|
|-- results/
|   |-- logs/
|   |-- metrics/
|   |-- plots/
|   `-- tables/
|
|-- checkpoints/                # saved models (gitignore)
|
`-- docs/                       # optional: report, paper draft
    |-- proposal.md
    |-- report.md
    ...
```