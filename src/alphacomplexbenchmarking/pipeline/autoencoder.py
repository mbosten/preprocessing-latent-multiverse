# src/alphacomplexbenchmarking/pipeline/autoencoder.py
from __future__ import annotations


def train_autoencoder_for_variant(spec: RunSpec) -> str:
    """
    Train AE on preprocessed data.
    Returns path/ID to saved model.
    """