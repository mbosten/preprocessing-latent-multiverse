from __future__ import annotations

import json
import logging
from typing import Any

from sklearn.metrics import roc_auc_score

from preprolamu.config import load_dataset_config
from preprolamu.helpers import feature_matrix, labels, load_split
from preprolamu.pipeline.autoencoder import load_autoencoder, reconstruction_error
from preprolamu.pipeline.evaluation import summarize_errors

logger = logging.getLogger(__name__)


BATCH_SIZE = 2048


def evaluate_on_universe(
        model,
        data_universe,
        *,
        split: str = "test",
) -> dict[str, Any]:
    """Evaluate a trained autoencoder on a target universe."""
    config = load_dataset_config(data_universe.dataset_id)
    
    df = load_split(data_universe, config, split)

    label_col = config["label_column"]
    y = labels(df, label_col)
    X = feature_matrix(df, label_col)

    # Check that the model's input dimension matches the data's feature dimension
    expected_dim = model.encoder[0].in_features
    if expected_dim != X.shape[1]:
        raise ValueError(f"Model input dimension ({expected_dim}) does not match data feature dimension ({X.shape[1]}).")
    
    errors = reconstruction_error(model, X, batch_size=BATCH_SIZE)

    benign = y == config["benign_label"]
    attack = ~benign

    return {
        "data_universe_id": data_universe.id,
        "data_dataset_id": data_universe.dataset_id,
        "n_samples": len(y),
        "roc_auc": float(roc_auc_score(attack.astype(y), errors)),
        "reconstruction": summarize_errors(errors),
        "benign": summarize_errors(errors[benign]),
        "attack": summarize_errors(errors[attack]),
    }


def evaluate_generalization(
        model_universe,
        universes,
        *,
        split: str = "test",
) -> dict[str, Any]:
    """Evaluate one AE on all universes with the same feature subset."""
    model = load_autoencoder(model_universe)

    targets = [
        u for u in universes
        if u.feature_subset == model_universe.feature_subset
        and u.id != model_universe.id 
    ]

    results = []

    for target in targets:
        try:
            results.append(
                evaluate_on_universe(
                    model,
                    target,
                    split=split,
                )
            )
        except ValueError as exc:
            logger.warning("Skipping target %s: %s", target.id, exc)

    return {
        "model_universe_id": model_universe.id,
        "model_dataset_id": model_universe.dataset_id,
        "feature_subset": model_universe.feature_subset,
        "split": split,
        "n_universes": len(results),
        "results": results,
    }


def save_generalization(
        universe,
        universes,
        *,
        split: str = "test",
        overwrite: bool = False,
) -> None:
    path = universe.paths.cross_eval_metrics(split=split)

    if path.exists() and not overwrite:
        logger.info("Cross-dataset evaluation already exists at %s. Skipping.", path)
        return

    if not universe.paths.ae_model().exists():
        logger.warning("No autoencoder model found for universe %s. Skipping.", universe.id)
        return

    result = evaluate_generalization(
        universe,
        universes,
        split=split,
    )

    path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    logger.info("Saved cross-dataset evaluation for %s to %s", universe.id, path)