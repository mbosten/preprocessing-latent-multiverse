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