from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from preprolamu.config import load_dataset_config
from preprolamu.helpers import feature_matrix, labels, load_split
from preprolamu.pipeline.autoencoder import (
    load_autoencoder,
    reconstruction_error,
)
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def summarize_errors(errors: np.ndarray) -> dict:
    errors = errors[np.isfinite(errors)]

    return {
        "n": len(errors),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "p95": float(np.quantile(errors, 0.95)),
    }


def evaluate_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        benign_label: str,
) -> dict:
    errors = reconstruction_error(model, X)

    benign = y == benign_label
    attack = ~benign

    return {
        "reconstruction": summarize_errors(errors),
        "roc_auc": float(roc_auc_score(attack.astype(int), errors)),
        "benign": summarize_errors(errors[benign]),
        "attack": summarize_errors(errors[attack]),
    }


def evaluate_autoencoder(universe: Universe, split: str = "test") -> dict:
    config = load_dataset_config(universe.dataset_id)
    
    df = load_split(universe, config, split)

    y = labels(df, config["label_column"])
    X = feature_matrix(df, config["label_column"])

    return {
        "universe_id": universe.id,
        "dataset_id": universe.dataset_id,
        "split": split,
        **evaluate_model(
            load_autoencoder(universe),
            X,
            y,
            config["benign_label"],
        )
    }


def save_evaluation(universe: Universe, overwrite: bool = False):
    path = universe.paths.eval_metrics(split="test")

    if path.exists() and not overwrite:
        return

    path.write_text(
        json.dumps(evaluate_autoencoder(universe), indent=4),
        encoding="utf-8",
    )
