from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import roc_auc_score

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


def evaluate_autoencoder(universe: Universe, split: str = "test") -> dict:
    df, config = load_split(universe, split)

    y = labels(df, config["label_column"])
    X = feature_matrix(df, config["label_column"])

    errors = reconstruction_error(load_autoencoder(universe), X)

    benign = y == config["benign_label"]
    attack = ~benign

    return {
        "universe_id": universe.id,
        "dataset_id": universe.dataset_id,
        "split": split,
        "reconstruction": summarize_errors(errors),
        "roc_auc": float(roc_auc_score(attack.astype(int), errors)),
        "benign": summarize_errors(errors[benign]),
        "attack": summarize_errors(errors[attack]),
    }


def save_evaluation(universe: Universe, overwrite: bool = False):
    path = universe.paths.eval_metrics(split="test")

    if path.exists() and not overwrite:
        return

    path.write_text(
        json.dumps(evaluate_autoencoder(universe), indent=4),
        encoding="utf-8",
    )
