# Getting Started

## Prerequisites

| Tool             | Version  | Notes                        |
|------------------|----------|------------------------------|
| Miniconda        | any      | installs conda + Python      |
| git              | any      | to clone the repo            |
| CUDA-capable GPU | optional | required for `fp16` training |

> **No conda?** Run this first:

```bash
# Create the install directory and download the Miniconda installer
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Silently install Miniconda into ~/miniconda3, then remove the installer
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Make conda available in the current session immediately
export PATH=~/miniconda3/bin:$PATH

# Persist conda across future sessions and reload shell config
echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
conda init
source ~/.bashrc

# Disable auto-activation of the base environment on every shell launch
# (macOS enables this by default after conda init)
conda config --set auto_activate_base false
```

---

## Setup

```bash
# 1. Clone the repository and enter the project directory
git clone <repo-url>
cd long-context-hallucination-detection

# 2. Create the conda environment and install all dependencies from requirements.txt
bash scripts/setup_env.sh
# Optional: pass a different Python version as the second argument (default: 3.11)
# bash scripts/setup_env.sh hallucination_env 3.12

# 3. Activate env
conda activate hallucination_env

# 4. Install the project package (makes `src` importable from any script)
pip install -e .

# 5. Download datasets — choose halueval, libreval, or all
#    Data is saved to data/raw/{name}/ from the project root
bash scripts/download_data.sh halueval
```

---

## Verify the setup locally

Run a 2-epoch smoke test on a 500-sample slice to confirm the full pipeline works:

```bash
python scripts/smoke_test_training.py
```

You should see training loss decrease over the two epochs. The checkpoint lands in `checkpoints/smoke_test/`.

---

## Running on the cluster

See **[docs/cluster_training.md](cluster_training.md)** for iTiger HPC setup, job submission, and result retrieval.