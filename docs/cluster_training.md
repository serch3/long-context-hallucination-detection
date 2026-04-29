# Running Training on the iTiger Cluster

A guide for running DistilBERT and ModernBERT training jobs on the [iTiger HPC cluster](https://itiger.memphis.edu) at the University of Memphis.

---

## Prerequisites

- SSH access to `itiger.memphis.edu` (UofM credentials; external users need VPN)
- Conda available (Miniconda is pre-installed)

---

## 1. Clone the repo

```bash
ssh your_username@itiger.memphis.edu
cd /project/$USER
git clone <repo-url> long-context-hallucination-detection
cd long-context-hallucination-detection
```

> Use `/project/$USER/` rather than `$HOME` — the home directory quota is only 20 GB, while project storage allows up to 200 GB.

---

## 2. Set up the environment (one-time)

```bash
bash cluster/setup_env.sh
```

This creates a conda environment at `/project/$USER/envs/hallucination_env`, installs PyTorch with CUDA 12.4, and installs all dependencies from `requirements.txt`.

After activating the environment, install the project package so `src` is importable from any script:

```bash
conda activate /project/$USER/envs/hallucination_env
pip install -e .
```

---

## 3. Download the data (one-time)

```bash
conda activate hallucination_env
bash scripts/download_data.sh
```

Raw data lands in `data/raw/halueval/`. Subsequent runs read from there directly.

---

## 4. Submit training jobs
```bash
# DistilBERT on default dataset mode (HaluEval) — ~4–6 hours
sbatch cluster/train_distilbert.sh

# ModernBERT on default dataset mode (HaluEval) — ~12–20 hours
sbatch cluster/train_modernbert.sh

# DistilBERT on LibreEval only
TRAIN_ARGS="--dataset libreval --libreval-splits tuning_train tuning_test" sbatch cluster/train_distilbert.sh

# DistilBERT on combined HaluEval + LibreEval
TRAIN_ARGS="--dataset both" sbatch cluster/train_distilbert.sh

# ModernBERT on combined HaluEval + LibreEval
TRAIN_ARGS="--dataset both" sbatch cluster/train_modernbert.sh
```
Monitor jobs:

```bash
squeue -u $USER
tail -f results/logs/distilbert_<jobid>.out
```

---

## 5. Retrieve results

Checkpoints are written to `checkpoints/distilbert/` and `checkpoints/modernbert/`. Copy them locally with:

```bash
rsync -avz your_username@itiger.memphis.edu:/project/your_username/long-context-hallucination-detection/checkpoints/ ./checkpoints/
rsync -avz your_username@itiger.memphis.edu:/project/your_username/long-context-hallucination-detection/results/ ./results/
```

---

## Resource summary

| Model | Partition | GPUs | RAM | Time limit |
|---|---|---|---|---|
| DistilBERT | bigTiger | 1 | 64 GB | 8 h |
| ModernBERT | bigTiger | 1 | 128 GB | 24 h |

ModernBERT uses `per_device_train_batch_size=4` with `gradient_accumulation_steps=8` (effective batch 32). If you hit OOM, halve the batch size and double accumulation steps in `configs/modernbert.yaml`, or request two GPUs by setting `--gres=gpu:2` in `cluster/train_modernbert.sh`.

---

## Quick smoke test

Run an interactive session before submitting a full job:

```bash
srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=0:30:00 --partition=bigTiger bash
# inside the session:
conda activate /project/$USER/envs/hallucination_env
cd /project/$USER/long-context-hallucination-detection
python -m scripts.train \
    --model-config configs/distilbert.yaml \
    --training-config configs/training.yaml \
    --limit 250
```

This trains on ~500 examples (250 per task × 2 tasks) for a fast end-to-end check.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run `pip install -e .` from the project root (one-time, makes `src` permanently importable) |
| GPU OOM on ModernBERT | Halve `per_device_train_batch_size`, double `gradient_accumulation_steps` in `configs/modernbert.yaml` |
| Conda not found in sbatch | Verify `which conda` works on a login node; job scripts rely on `eval "$(conda shell.bash hook)"` |
| Job exceeds time limit | Increase `--time` in the sbatch header or reduce `num_train_epochs` in `configs/training.yaml` |