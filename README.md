# preprolamu - Preprocessing Latent Multiverse

This repository implements a multiverse analysis pipeline for studying how preprocessing variations influence the latent representation of a dataset, using:
- automatic data cleaning (one-time per dataset),
- Preprocessing specification in Universe class,
- Autoencoder training and embedding,
- Flexible PCA projection,normalization and sampling of the embedding space. 
- Modular Alpha complex persistence.
- \[IN PROGRESS\] Extensive comparison with other embedding quality metrics
- \[IN PROGRESS\] Cross-dataset generalization evaluation

The pipeline is fully modular and allows systematic exploration of combinations of:
- \[BEING UPDATED\]

… producing a complete multiverse of latent-space topologies.

## Directory Structure
```yaml
preprolamu
├── CLI
│   ├── analyses.py
│   ├── cli.py
│   └── plots.py
├── experiments
│   ├── PCA_components
│   │   ├── merge_pca_experiment_csvs.py
│   │   └── pca_dim_experiment.py
│   ├── Sample_size
│   │   ├── merge_sample_experiment_csvs.py
│   │   └── sample_size_experiment.py
│   ├── complex_benchmark.py
│   ├── embedding_persistence_experiment.py
│   ├── experiment.py
│   └── single_u_multi_seed_sample_size_experiment.py
├── helpers
│   ├── __init__.py
│   ├── dataset.py
│   ├── results.py
│   ├── statistics.py
│   ├── tabular.py
│   └── tda.py
├── io
│   ├── io.py
│   └── paths.py
├── notebooks
├── pipeline
│   ├── autoencoder.py
│   ├── create_tda.py
│   ├── cross_dataset_evaluation.py
│   ├── embedding_quality_metrics.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── persistence.py
│   ├── preprocessing.py
│   └── universes.py
├── tests
│   ├── ae_anomaly_check.py
│   └── aggregate_ae_anomaly_checks.py
├── __init__.py
├── config.py
└── example_plots.py
```