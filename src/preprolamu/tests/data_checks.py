# src/preprolamu/tests/data_checks.py
from __future__ import annotations

import logging

import numpy as np

from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def log_feature_stats(X: np.ndarray, split: str, universe: Universe) -> None:
    X = X.astype(np.float64)

    # Global stats
    finite_mask = np.isfinite(X)
    n_total = X.size
    n_finite = finite_mask.sum()
    n_nan = np.isnan(X).sum()
    n_posinf = np.isposinf(X).sum()
    n_neginf = np.isneginf(X).sum()

    logger.info(
        "[AE][%s][%s] global stats: n_total=%d, n_finite=%d, n_nan=%d, +inf=%d, -inf=%d",
        universe.to_id_string(),
        split,
        n_total,
        n_finite,
        n_nan,
        n_posinf,
        n_neginf,
    )

    if n_finite == 0:
        logger.warning(
            "[AE][%s][%s] no finite values in X!", universe.to_id_string(), split
        )
        return

    X_finite = X[finite_mask]

    logger.info(
        "[AE][%s][%s] finite values: min=%.4e, max=%.4e, mean=%.4e, std=%.4e",
        universe.to_id_string(),
        split,
        np.min(X_finite),
        np.max(X_finite),
        np.mean(X_finite),
        np.std(X_finite),
    )

    # Rough idea of scale per feature
    sample = X[: min(100000, X.shape[0]), :]  # sample rows
    feature_means = np.nanmean(sample, axis=0)
    feature_stds = np.nanstd(sample, axis=0)
    feature_max_abs = np.nanmax(np.abs(sample), axis=0)

    # Log top-5 “worst” features by max abs value
    worst_idx = np.argsort(-feature_max_abs)[:5]
    for idx in worst_idx:
        logger.info(
            "[AE][%s][%s] feature %d: mean=%.4e, std=%.4e, max_abs=%.4e",
            universe.to_id_string(),
            split,
            idx,
            feature_means[idx],
            feature_stds[idx],
            feature_max_abs[idx],
        )
