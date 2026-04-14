# Repo Structure

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
|   `-- training.yaml
|
|-- data/
|   |-- raw/                    # original HaluEval datasets
|   `-- processed/              # tokenized / cleaned data
|
|-- notebooks/                  # exploration / prototyping
|   |-- eda.ipynb
|   `-- error_analysis.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- utils.py                # seed, logging, shared helpers
|   |
|   |-- data/
|   |   |-- loader.py           # load HaluEval splits → unified DatasetDict
|   |   `-- preprocess.py       # cleaning, tokenization, truncation
|   |
|   |-- models/
|   |   |-- distilbert.py       # baseline model wrapper
|   |   `-- modernbert.py       # main model wrapper
|   |
|   |-- training/
|   |   `-- trainer.py          # HF Trainer config + training loop
|   |
|   `-- evaluation/
|       |-- metrics.py          # accuracy, F1, AUROC
|       |-- evaluator.py        # load checkpoint → JSON report
|       `-- error_analysis.py   # FP/FN analysis bucketed by input length
|
|-- scripts/                    # CLI entrypoints
|   |-- download_data.sh
|   |-- setup_env.sh
|   |-- run_experiment.py       # main entrypoint: wires data→train→eval
|   `-- compare_models.py       # load both result JSONs → comparison table
|
|-- results/
|   |-- logs/
|   |-- metrics/
|   `-- plots/
|
|-- checkpoints/                # saved models (gitignored)
|
`-- docs/
    |-- proposal.md
    |-- plan.md
    `-- repo_structure.md
```
