from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

import numpy as np

from matplotlib.figure import Figure
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from preprolamu.config import load_dataset_config
from preprolamu.helpers import feature_matrix, labels, load_split
from preprolamu.pipeline.autoencoder import load_autoencoder, reconstruction_error
from preprolamu.pipeline.evaluation import summarize_errors
from preprolamu.pipeline.preprocessing import Preprocessor
from preprolamu.pipeline.universes import DATASET_IDS

logger = logging.getLogger(__name__)


BATCH_SIZE = 8192


def evaluate_on_universe(
        model,
        data_universe,
        *,
        split: str = "test",
        feature_var: np.ndarray | None = None,
        df=None,
):
    """Evaluate a trained autoencoder on a target universe."""
    config = load_dataset_config(data_universe.dataset_id)

    if df is None:
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
        "auprc": float(average_precision_score(y_true, errors)) if np.unique(y_true).size > 1 else None,
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
    """Evaluate an AE on the remaining original datasets using source-fitted preprocessing."""
    model = load_autoencoder(model_universe)

    # Redo the preprocessing for the model universe to retrieve the parameters needed to apply these to the target universes.
    source = Preprocessor(model_universe)
    source.process()


    # Required for normalization of the reconstruction error
    config = load_dataset_config(model_universe.dataset_id)
    X_train = feature_matrix(source.train, config["label_column"])
    feature_var = np.maximum(np.var(X_train, axis=0), 1e-6)

    del X_train

    results = []
    raw_evaluations = {}

    for dataset_id in DATASET_IDS:
        if dataset_id == model_universe.dataset_id:
            continue

        logger.info("[CROSSEVAL] %s -> %s", model_universe.id, dataset_id)
        target_universe = replace(
            model_universe, dataset_id=dataset_id, universe_index=None,
        )
        target = Preprocessor(target_universe, source=source)
        train, val, test = target.process()
        df = {"train": train, "val": val, "test": test}[split]

        result, y_true, scores = evaluate_on_universe(
            model, target_universe, split=split,
            feature_var=feature_var, df=df,
        )
        if result["roc_auc"] is None:
            raise ValueError(f"AUROC undefined for target dataset {dataset_id}")

        raw_evaluations[f"{dataset_id}__y_true"] = y_true
        raw_evaluations[f"{dataset_id}__scores"] = scores
        results.append(result)

    metrics = {
        "model_universe_id": model_universe.id,
        "model_dataset_id": model_universe.dataset_id,
        "feature_subset": model_universe.feature_subset,
        "n_features": model.encoder[0].in_features,
        "split": split,
        "n_universes": len(results),
        "results": results,
        "mean_roc_auc": float(np.mean([r["roc_auc"] for r in results])),
    }
    return metrics, raw_evaluations


def save_curve_figures(universe, raw_evaluations: dict, *, split: str = "test") -> None:
    """Save ROC and precision-recall figures"""
    target_ids = sorted({key.split("__")[0] for key in raw_evaluations})

    roc_fig, pr_fig = Figure(figsize=(7, 6)), Figure(figsize=(7, 6))
    roc_ax, pr_ax = roc_fig.subplots(), pr_fig.subplots()

    for target_id in target_ids:
        y_true = raw_evaluations[f"{target_id}__y_true"]
        scores = raw_evaluations[f"{target_id}__scores"]
        if np.unique(y_true).size < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_ax.plot(fpr, tpr, lw=1, label=f"{target_id} (AUROC={auc(fpr, tpr):.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        pr_ax.plot(recall, precision, lw=1, label=f"{target_id} (AUPRC={ap:.3f})")

    roc_ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"Cross-dataset ROC: model {universe.id} ({split})")
    pr_ax.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02),
              title=f"Cross-dataset PR: model {universe.id} ({split})")

    for fig, ax, kind in ((roc_fig, roc_ax, "roc"), (pr_fig, pr_ax, "prc")):
        ax.legend(fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(
            universe.paths.figures(f"cross_eval_{kind}/{universe.id}_cross_eval_{kind}_{split}"),
            dpi=150,
        )


def save_generalization(
        universe,
        universes,
        *,
        split: str = "test",
        overwrite: bool = False,
) -> None:
    metrics_path = universe.paths.cross_eval_metrics(split=split)
    scores_path = universe.paths.cross_eval_scores(split=split)

    figure_paths = [
        universe.paths.figures(f"cross_eval_{kind}/{universe.id}_cross_eval_{kind}_{split}")
        for kind in ("roc", "prc")
    ]

    if metrics_path.exists() and scores_path.exists() and not overwrite:
        if all(p.exists() for p in figure_paths):
            logger.info("Cross-dataset evaluation already exists at %s. Skipping.", metrics_path)
        else:
            # Backfill figures from the stored raw scores without re-evaluating.
            with np.load(scores_path) as data:
                save_curve_figures(universe, dict(data), split=split)
            logger.info("Saved missing cross-dataset curve figures for %s.", universe.id)
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
    np.savez(scores_path, **raw_evaluations)
    save_curve_figures(universe, raw_evaluations, split=split)

    logger.info("Saved cross-dataset evaluation for %s to %s", universe.id, metrics_path)