from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from preprolamu.config import load_dataset_config
from preprolamu.helpers import feature_matrix_from_df, labels_from_df
from preprolamu.pipeline.autoencoder import load_autoencoder_for_universe
from preprolamu.pipeline.evaluation import _recon_error_per_sample, _summarize_errors
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def _feature_group(universe: Universe) -> str:
    """
    Check whether this universe has the feature subset included.
    Because this needs checking often, it is a separate function.
    """
    return (
        "excluded"
        if universe.get("feature_subset", "without_confounders")
        else "included"
    )


def _compatible_universes(
    source_universe: Universe,
    candidate_universes: Iterable[Universe],
) -> list[Universe]:
    source_group = _feature_group(source_universe)
    return [u for u in candidate_universes if _feature_group(u) == source_group]


def _eval_model_on_universe(
    model_universe: Universe,
    data_universe: Universe,
    split="test",
    batch_size: int = 2048,
    include_stratified: bool = True,
) -> dict[str, Any]:
    ds_cfg = load_dataset_config(model_universe.dataset_id)
    label_col = ds_cfg["label_column"]

    df = pd.read_parquet(model_universe.io.paths.preprocessed(split=split))

    y = labels_from_df(df, label_col=label_col)
    X = feature_matrix_from_df(df, label_col=label_col)

    ds_cfg = load_dataset_config(model_universe.dataset_id)
    model = load_autoencoder_for_universe(model_universe, ds_cfg)

    err = _recon_error_per_sample(model, X, batch_size=batch_size)

    out: dict[str, Any] = {
        "model_universe_id": model_universe.id,
        "data_universe_id": data_universe.id,
        "model_dataset_id": model_universe.dataset_id,
        "data_dataset_id": data_universe.dataset_id,
        "feature_group": _feature_group(model_universe),
        "split": split,
        "label_column": label_col,
        "recon": _summarize_errors(err),
    }

    if y is not None:
        y_true = (y != "Benign").astype(int)
        out["roc_auc"] = float(roc_auc_score(y_true, err))

    if include_stratified and (y is not None) and (len(y) == len(err)):
        benign_mask = y == "Benign"
        attack_mask = ~benign_mask

        out["recon_benign"] = _summarize_errors(err[benign_mask])
        out["recon_attack"] = _summarize_errors(err[attack_mask])

        out["n_benign"] = int(benign_mask.sum())
        out["n_attack"] = int(attack_mask.sum())

    return out


def evaluate_autoencoder_cross_reconstruction(
    model_universe: Universe,
    all_universes: Iterable[Universe],
    split: str = "test",
    batch_size: int = 2048,
    include_stratified: bool = True,
) -> dict[str, Any]:

    compatible = _compatible_universes(model_universe, all_universes)

    results = []
    for data_universe in compatible:
        result = _eval_model_on_universe(
            model_universe=model_universe,
            data_universe=data_universe,
            split=split,
            batch_size=batch_size,
            include_stratified=include_stratified,
        )
        results.append(result)

    return {
        "model_universe_id": model_universe.id,
        "model_dataset_id": model_universe.dataset_id,
        "feature_group": _feature_group(model_universe),
        "split": split,
        "n_evaluated_universes": len(results),
        "cross_recon": results,
    }


def evalue_cross(
    universe: Universe,
    all_universes: Iterable[Universe],
    batch_size: int = 2048,
    overwrite: bool = False,
    include_stratified: bool = True,
):
    out_path = universe.paths.cross_eval_metrics(split="test")

    model_path = universe.paths.ae_model()

    if out_path.exists() and not overwrite:
        logger.info(
            "[AE-CROSS-EVAL] Skipping existing eval for universe %s", universe.id
        )
        return

    if not model_path.exists():
        logger.info("[AE-CROSS-EVAL] Missing model for universe %s", universe.id)
        return

    try:
        result = evaluate_autoencoder_cross_reconstruction(
            model_universe=universe,
            all_universes=all_universes,
            split="test",
            batch_size=batch_size,
            include_stratified=include_stratified,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    except FileNotFoundError as e:
        logger.warning("[AE-CROSS-EVAL] Missing file for %s: %s", universe.id, e)

    except Exception as e:
        logger.warning("[AE-CROSS-EVAL] Failed for %s: %s", universe.id, e)

    logger.info("[AE-CROSS-EVAL] Evaluation for %s complete.", universe.id)
