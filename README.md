# preprolamu - Preprocessing Latent Multiverse

This repository implements a multiverse analysis pipeline for studying how preprocessing variations influence the latent representation of a dataset, using:
- automatic data cleaning (one-time per dataset),
- flexible preprocessing via `Universe`,
- optional autoencoder embeddings,
- PCA projection + subsampling,
- alpha complex persistent homology and landscapes,
- parallel execution (local + SLURM-ready).

The pipeline is fully modular and allows systematic exploration of combinations of:
- scaling choices
- categorical encodings
- feature subsets
- random seeds
- embedding dimensionality
- TDA hyperparameters

… producing a complete multiverse of latent-space topologies.

## Directory Structure
```yaml
project/
│
├── config
│   └── datasets
│       ├── Merged35.yml
│       └── NF-ToN-IoT-v3.yml
├── data
│   ├── experiments
│   │   ├── latent
│   │   ├── pca
│   │   ├── simple_pd
│   │   ├── simple_pd_grid
│   │   └── subsampling
│   ├── interim
│   │   ├── autoencoder
│   │   ├── embeddings
│   │   ├── landscapes
│   │   └── persistence
│   ├── processed
│   │   └── metrics
│   └── raw
├── logs
├── src
│   └── preprolamu
│       ├── experiments
│       │   └── parameter_sensitivity.py
│       ├── io
│       │   └── storage.py
│       ├── pipeline
│       │   ├── autoencoder.py
│       │   ├── create_embeddings.py
│       │   ├── create_tda.py
│       │   ├── embeddings.py
│       │   ├── landscapes.py
│       │   ├── metrics.py
│       │   ├── parallel.py
│       │   ├── persistence.py
│       │   ├── preprocessing.py
│       │   ├── tda.py
│       │   └── universes.py
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── make_readme_tree.py
│       └── visualization.py
└── README.md
```

## 1. One-Time Dataset Preparation
Before running any Universes, clean your dataset once using the user-specified dataset-specific YAML config file.
In it, specify:
- which columns are non-numerical
- which columns should be dropped a priori
- the label column
- all the label classes
For an example, see the `config/datasets` directory.

Run the cleaning script as follows:
```bash
uv run setup initiate [dataset_id]
```

Alternatively, if using the CLI keybind fails (see FAQ):
```bash
$env:PYTHONPATH="src"
uv run python -m preprolamu.config initiate [dataset_id]
```

This will return a cleaned dataset with the "_cleaned" suffix in `data/raw/`. The function sets specified columns to string values and drops the unwanted columnms.
Currently only Parquet and CSV files are supported.

## 2. Universe Configuration
Multiverse experiments are built from `Universe` objects defined in `pipeline/universes.py`
One `Universe` determines:
- Scaling: Z-score / Min-Max
- Categorical encoding: One-hot / Ordinal
- Feature exclusion: Yes / No
- Seed: 42 / 420 / 4200
- PCA components: 2 / 3 / 4
- AE architecture (fixed)
- TDA configuration (fixed)

`generate_multiverse()` subsequently specifies all possible universes.

## 3. Pipeline Stages
For each `Universe`, the pipeline executes:

### Step 1: Preprocessing
From `pipeline/preprocessing.py`:
- Load cleaned dataset from `data/raw/`
- Drop excluded features
- Scale numeric features (zscore/minmax)
- Encode categorical features (onehot/ordinal)

### Step 2: Autoencoder Training
From `pipeline/autoencoder.py`:
- 🚧<span style="color:orange">**UNDER CONSTRUCTION**</span>🚧

### Step 3: Embedding → PCA → Subsampling
From `pipeline/embeddings.py`:
- 🚧<span style="color:orange">**UNDER CONSTRUCTION**</span>🚧
- Diameter-normalizes embeddings
- PCA projection to lower dimension
- Subsamples N points for TDA computation

### Step 4: Persistent Homology + Landscapes
From `pipeline/tda.py`:
- Compute Alpha complex persistence
- Compute persistence landscapes
- store results to `data/interim/persistence/*.npz`

### Step 5: Metrics
From `pipeline/metrics.py`:
- Summarize TDA output (cumulative persistence, landscape L2 norms)
- Save JSON summaries to `data/processed/metrics/*.json`

## 4. Running the Pipeline (CLI)
The `pyproject.toml` file is set up such that the CLI commands are simple to run. The main Typer CLI is in:
```bash
src/preprolamu/cli.py
```
This can be run from the command line with the following keybind:
```bash
uv run acb [function] [parameters]
```

### List all universes
```bash
uv run acb list-universes
```

### Run a single universe
```bash
uv run acb run-universe 0
```

### Run a batch of universes in parallel
```bash
uv run acb run-universe-batch --start 0 --end 10 --max-workers 4
```

output artifacts will appear in:
```bash
data/interim/
data/processed/
logs/debug.log
```

## 5. Parallel Execution
Local parallelization uses
```bash
concurrent.futures.ProcessPoolExecutor
```
in `pipeline/parallel.py`.
Cluster execution (SLURM job arrays) is straightforward. For a multiverse of size 54:
```bash
#SBATCH --array=0-53
uv run acb run-universe $SLURM_ARRAY_TASK_ID
```
This isolates each universe to one job.
