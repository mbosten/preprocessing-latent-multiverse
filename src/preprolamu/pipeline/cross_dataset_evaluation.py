from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from sklearn.metrics import roc_auc_score

from preprolamu.config import load_dataset_config
from preprolamu.helpers import feature_matrix, labels, load_split
from preprolamu.pipeline.autoencoder import load_autoencoder, reconstruction_error
from preprolamu.pipeline.evaluation import summarize_errors

logger = logging.getLogger(__name__)


BATCH_SIZE = 8192


def evaluate_on_universe(
        model,
        data_universe,
        *,
        split: str = "test",
        feature_var: np.ndarray | None = None,
):
    """Evaluate a trained autoencoder on a target universe."""
    config = load_dataset_config(data_universe.dataset_id)
    
    df = load_split(data_universe, config, split)

    label_col = config["label_column"]
    y = labels(df, label_col)
    X = feature_matrix(df, label_col)

    # Check that the model's input dimension matches the data's feature dimension
    expected_dim = model.encoder[0].in_features
    actual_dim = X.shape[1]
    if expected_dim != actual_dim:
        raise ValueError(f"Model input dimension ({expected_dim}) does not match data feature dimension ({actual_dim}).")
    
    errors = reconstruction_error(model, X, batch_size=BATCH_SIZE, feature_var=feature_var)

    benign = y == config["benign_label"]
    y_true = (~benign).astype(np.uint8)

    result = {
        "data_universe_id": data_universe.id,
        "data_dataset_id": data_universe.dataset_id,
        "n_samples": len(y),
        "n_features": actual_dim,
        "roc_auc": float(roc_auc_score(y_true, errors)) if np.unique(y_true).size > 1 else None,
        "reconstruction": summarize_errors(errors),
        "benign": summarize_errors(errors[benign]),
        "attack": summarize_errors(errors[~benign]),
    }

    return result, y_true, errors


def evaluate_generalization(
        model_universe,
        universes,
        *,
        split: str = "test",
):
    """Evaluate one AE on all universes with the same feature subset."""
    model = load_autoencoder(model_universe)

    targets = [
        u
        for u in universes
        if u.feature_subset == model_universe.feature_subset
        and u.id != model_universe.id
    ]

    logger.info("Evaluating generalization of model from universe %s on %d target universes.",
            model_universe.id,
            len(targets),
        )

    # Required for normalization of the reconstruction error
    config = load_dataset_config(model_universe.dataset_id)
    train_df = load_split(model_universe, config, split="train")
    X_train = feature_matrix(train_df, config["label_column"])
    feature_var = np.maximum(np.var(X_train, axis=0), 1e-6)

    results = []
    raw_evaluations = {}

    for i, target in enumerate(targets, start=1):
        logger.info("[CROSSEVAL] u-%04d [%d/%d] -> u-%04d", model_universe.universe_index, i, len(targets), target.universe_index)
        try:
            result, y_true, scores = evaluate_on_universe(
                    model,
                    target,
                    split=split,
                    feature_var=feature_var,
                )
            
        except ValueError as exc:
            logger.warning(
                "[CROSSEVAL] Skipping u-%04d -> u-%04d: %s",
                model_universe.universe_index,
                target.universe_index,
                exc,
            )
            continue

        prefix = str(target.id)

        raw_evaluations[f"{prefix}__y_true"] = y_true
        raw_evaluations[f"{prefix}__scores"] = scores
        results.append(result)

    metrics = {
        "model_universe_id": model_universe.id,
        "model_dataset_id": model_universe.dataset_id,
        "feature_subset": model_universe.feature_subset,
        "n_features": model.encoder[0].in_features,
        "split": split,
        "n_universes": len(results),
        "results": results,
    }

    return metrics, raw_evaluations


def save_generalization(
        universe,
        universes,
        *,
        split: str = "test",
        overwrite: bool = False,
) -> None:
    metrics_path = universe.paths.cross_eval_metrics(split=split)
    scores_path = universe.paths.cross_eval_scores(split=split)

    if metrics_path.exists() and scores_path.exists() and not overwrite:
        logger.info("Cross-dataset evaluation already exists at %s. Skipping.", metrics_path)
        return

    if not universe.paths.ae_model().exists():
        logger.warning("No autoencoder model found for universe %s. Skipping.", universe.id)
        return

    result, raw_evaluations = evaluate_generalization(
        universe,
        universes,
        split=split,
    )

    metrics_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    np.savez_compressed(scores_path, **raw_evaluations)

    logger.info("Saved cross-dataset evaluation for %s to %s", universe.id, metrics_path)