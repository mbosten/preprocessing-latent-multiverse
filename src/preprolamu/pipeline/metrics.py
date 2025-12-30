# src/preprolamu/pipeline/metrics.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprolamu.io.storage import load_landscapes, load_metrics
from preprolamu.pipeline.universes import Universe
from preprolamu.tests.landscape_checks import top_landscape_outliers

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """
    Simple scalar summaries of TDA, per homology dimension.
    Extend this as needed.
    """

    total_persistence_per_dim: Dict[int, float]
    landscape_l2_per_dim: Dict[int, float]


def load_landscapes_with_provenance(
    universes: Iterable[Universe],
    split: str = "test",
) -> tuple[List[Tuple[Universe, Dict[int, np.ndarray]]], int, int]:
    """
    Load landscapes for universes while preserving provenance (Universe object),
    and keep the same robust missing/empty checks you already had.
    """
    pairs: List[Tuple[Universe, Dict[int, np.ndarray]]] = []
    skipped_missing = 0
    skipped_empty = 0

    for u in universes:
        path = u.landscapes_path(split=split)
        if not path.exists():
            skipped_missing += 1
            continue

        L = load_landscapes(u, split=split)
        if not L:
            skipped_empty += 1
            continue

        pairs.append((u, L))

    return pairs, skipped_missing, skipped_empty


def compute_total_persistence(intervals: np.ndarray) -> float:
    """
    Sum of (death - birth) over all finite intervals.
    intervals: array of shape (k, 2)
    """
    if intervals.size == 0:
        return 0.0
    births = intervals[:, 0]
    deaths = intervals[:, 1]
    lengths = np.clip(deaths - births, a_min=0.0, a_max=None)
    return float(lengths.sum())


# PRESTO function, slightly adapted
def compute_landscape_norm(
    landscape: Dict[int, np.array],
    score_type: str = "aggregate",
) -> Dict[int, float] | float:
    norms = {
        k: float(np.linalg.norm(v)) if v is not None else 0.0
        for k, v in landscape.items()
    }
    if score_type == "aggregate":
        return sum(norms.values())
    elif score_type == "average":
        return sum(norms.values()) / len(norms.values())
    elif score_type == "separate":
        return norms
    else:
        raise NotImplementedError(score_type)


def compute_landscape_norm_means(
    landscapes: List[Dict[int, np.array]],
    homology_dims: List[int] | None = None,
    return_norms: bool = False,
):
    if not landscapes:
        raise ValueError("landscapes list is empty.")

    # per-landscape norms, missing dims -> 0
    landscape_norms: List[Dict[int, float]] = []
    dims_seen: set[int] = set()

    for L in landscapes:
        norms = {
            d: float(np.linalg.norm(arr)) for d, arr in L.items() if arr is not None
        }
        landscape_norms.append(norms)
        dims_seen.update(norms.keys())

    dims = list(homology_dims) if homology_dims is not None else sorted(dims_seen)
    N = len(landscapes)

    # Debugging functionality: log top outliers
    logger.info("Length landscape norms: %d", len(landscape_norms))
    for d in dims:
        top = top_landscape_outliers(landscape_norms, dims=[d], top_k=10)
        logger.info(
            "[TDA] Top landscape norm outliers for dim %d (dim, norm, idx_in_list): %s",
            d,
            top,
        )

    means = {d: sum(n.get(d, 0.0) for n in landscape_norms) / N for d in dims}

    return (means, landscape_norms) if return_norms else means


def compute_presto_variance(
    landscapes: List[Dict[int, np.ndarray]],
    homology_dims: Iterable[int] | None = None,
) -> float:

    if not landscapes:
        raise ValueError("landscapes list is empty.")

    if homology_dims is None:
        homology_dims = range(max(landscapes[0].keys()) + 1)
    homology_dims = list(homology_dims)

    mean_norms, norms_per_landscape = compute_landscape_norm_means(
        landscapes, homology_dims=homology_dims, return_norms=True
    )
    logger.info("[TDA] Mean landscape norms per dim: %s", mean_norms)
    dims = list(mean_norms.keys())
    N = len(norms_per_landscape)

    sse = 0.0
    for d in dims:
        mu = mean_norms[d]
        for n in norms_per_landscape:
            sse += (n.get(d, 0.0) - mu) ** 2

    return sse / N


def compute_presto_variance_from_metrics_table(
    df: pd.DataFrame,
    *,
    homology_dims: Iterable[int] = (0, 1, 2),
) -> float:
    """
    Compute PRESTO variance using precomputed landscape L2 norms stored in a metrics table.

    Expects df to contain columns l2_dim{d} for each d in homology_dims.
    Uses the same definition as compute_presto_variance():
      var = (1/N) * Σ_u Σ_d (n_{u,d} - μ_d)^2
    where μ_d is the mean across universes for dimension d, and N is number of universes.
    """
    dims = list(homology_dims)

    if df.empty:
        raise ValueError("metrics table is empty.")

    cols = [f"l2_dim{d}" for d in dims]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for PRESTO variance: {missing}")

    X = df[cols].to_numpy(dtype=float)

    # Treat non-finite as 0.0 (consistent with build_metrics_table's "missing dims -> 0.0" intent)
    # If you prefer to drop rows with non-finite, we can do that instead.
    X = np.where(np.isfinite(X), X, 0.0)

    N = X.shape[0]
    if N == 0:
        raise ValueError("No rows available for PRESTO variance.")

    mu = X.mean(axis=0)
    sse = ((X - mu) ** 2).sum()
    return float(sse / N)


def compute_presto_variance_across_universes(
    universes: Iterable[Universe],
    split: str = "test",
    homology_dims: Iterable[int] | None = (0, 1, 2),
) -> float:
    pairs, skipped_missing, skipped_empty = load_landscapes_with_provenance(
        universes, split=split
    )

    if not pairs:
        raise RuntimeError(
            f"No landscapes found (split={split}, skipped_missing={skipped_missing}, "
            f"skipped_empty={skipped_empty}). Cannot compute PRESTO variance."
        )

    landscapes_list = [L for (_, L) in pairs]

    var = compute_presto_variance(
        landscapes=landscapes_list, homology_dims=homology_dims
    )

    logger.info(
        "[TDA] PRESTO variance across %d universes (split=%s); skipped missing=%d, empty=%d; var=%.6g",
        len(landscapes_list),
        split,
        skipped_missing,
        skipped_empty,
        var,
    )

    return var


def compute_metrics_from_tda(
    persistence_per_dimension: Dict[int, np.ndarray],
    landscapes_per_dimension: Dict[int, Optional[np.ndarray]],
) -> MetricsResult:
    total_persistence_per_dim: Dict[int, float] = {}
    landscape_l2_per_dim: Dict[int, float] = {}

    for dim, intervals in persistence_per_dimension.items():
        total_persistence_per_dim[dim] = compute_total_persistence(intervals)

    landscape_l2_per_dim = compute_landscape_norm(
        landscapes_per_dimension,
        score_type="separate",
    )

    return MetricsResult(
        total_persistence_per_dim=total_persistence_per_dim,
        landscape_l2_per_dim=landscape_l2_per_dim,
    )


def build_metrics_table(
    universes: Iterable[Universe],
    split: str = "test",
    require_exists: bool = True,
    homology_dims: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """
    One row per universe, using stored metrics JSON only.
    Missing dims are treated as 0.0 (your earlier decision).
    """
    rows: List[Dict[str, Any]] = []

    for u in universes:
        path = u.metrics_path(split=split)
        if require_exists and not path.exists():
            continue

        row = u.to_param_dict()
        row["universe_id"] = u.id
        row["split"] = split
        row["metrics_path"] = str(path)

        try:
            payload = load_metrics(u, split=split)  # <-- dict from JSON
            if not payload:
                row["metrics_status"] = "empty_metrics"
                row["failure_reason"] = "metrics JSON is empty"
                row["l2_aggregate"] = np.nan
                row["l2_average"] = np.nan
                rows.append(row)
                continue

            # keys may be strings in JSON: {"0": 1.23, "1": 4.56}
            l2_raw = payload.get("landscape_l2_per_dim", {}) or {}
            tp_raw = payload.get("total_persistence_per_dim", {}) or {}

            l2 = {int(k): float(v) for k, v in l2_raw.items()}
            tp = {int(k): float(v) for k, v in tp_raw.items()}

            row["metrics_status"] = "ok"
            row["failure_reason"] = None

            l2_vals: List[float] = []
            for d in homology_dims:
                v = float(l2.get(d, 0.0))
                row[f"l2_dim{d}"] = v
                l2_vals.append(v)

                row[f"tp_dim{d}"] = float(tp.get(d, 0.0))

            row["l2_aggregate"] = float(sum(l2_vals))
            row["l2_average"] = float(sum(l2_vals) / max(len(l2_vals), 1))

        except FileNotFoundError as e:
            row["metrics_status"] = "missing_metrics"
            row["failure_reason"] = str(e)
            row["l2_aggregate"] = np.nan
            row["l2_average"] = np.nan

        except Exception as e:
            row["metrics_status"] = "metrics_load_error"
            row["failure_reason"] = f"{type(e).__name__}: {e}"
            row["l2_aggregate"] = np.nan
            row["l2_average"] = np.nan

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics files found for split={split!r}.")
    return pd.DataFrame(rows)
