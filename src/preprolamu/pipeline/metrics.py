# src/preprolamu/pipeline/metrics.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from preprolamu.io.storage import load_landscapes
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


# Check whether landscape file is non-empty
def npz_has_keys(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path) as data:
        return len(data.files) > 0


def load_landscapes_with_provenance(
    universes: Iterable[Universe],
    split: str = "test",
    skip_empty: bool = True,
) -> List[Tuple[Universe, dict[int, np.ndarray]]]:
    """
    Load landscapes and keep the Universe object attached.
    This prevents 'index in filtered list' confusion later.
    """
    out: List[Tuple[Universe, dict[int, np.ndarray]]] = []
    for u in universes:
        L = load_landscapes(u, split=split)
        if skip_empty and not L:
            continue
        out.append((u, L))
    return out


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


def compute_presto_variance_across_universes(
    universes: Iterable[Universe],
    split: str = "test",
    homology_dims: Iterable[int] | None = (0, 1, 2),
) -> float:
    """
    Load landscapes for all given universes and compute PRESTO-style variance
    of landscape norms across them.

    Each universe must already have TDA results saved.
    """
    landscapes_list: List[Dict[int, np.ndarray]] = []
    skipped_missing = 0
    skipped_empty = 0

    for u in universes:
        path = u.landscapes_path(split=split)
        if not path.exists():
            skipped_missing += 1
            continue

        # Empty NPZ => empty dict after load
        if not npz_has_keys(path):
            skipped_empty += 1
            continue

        landscapes = load_landscapes(u, split=split)
        if not landscapes:
            skipped_empty += 1
            continue

        landscapes_list.append(landscapes)

    if not landscapes_list:
        raise RuntimeError(
            f"No landscapes found (split={split}, skipped_missing={skipped_missing}, "
            f"skipped_empty={skipped_empty}). Cannot compute PRESTO variance."
        )

    var = compute_presto_variance(
        landscapes=landscapes_list,
        homology_dims=homology_dims,
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


def build_landscape_norm_table(
    universes: Iterable[Universe],
    split: str = "test",
    require_exists: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for u in universes:
        path = u.landscapes_path(split=split)
        if require_exists and not path.exists():
            continue

        row = u.to_param_dict()
        row["universe_id"] = getattr(u, "id", u.to_id_string())
        row["split"] = split
        row["landscapes_path"] = str(path)

        try:
            landscapes = load_landscapes(u, split=split)

            if not landscapes:
                row["landscape_status"] = "empty_landscapes"
                row["failure_reason"] = (
                    "load_landscapes returned no landscapes (empty dict)."
                )
                row["norm_aggregate"] = np.nan
                row["norm_average"] = np.nan
            else:
                per_dim = compute_landscape_norm(landscapes, score_type="separate")
                agg = compute_landscape_norm(landscapes, score_type="aggregate")
                avg = compute_landscape_norm(
                    landscapes, score_type="average"
                )  # <-- NEW

                row["landscape_status"] = "ok"
                row["failure_reason"] = None
                row["norm_aggregate"] = agg
                row["norm_average"] = avg  # <-- NEW
                for d, v in per_dim.items():
                    row[f"norm_dim{d}"] = v

        except FileNotFoundError as e:
            row["landscape_status"] = "missing_landscapes"
            row["failure_reason"] = str(e)
            row["norm_aggregate"] = np.nan
            row["norm_average"] = np.nan

        except Exception as e:
            row["landscape_status"] = "landscape_load_error"
            row["failure_reason"] = f"{type(e).__name__}: {e}"
            row["norm_aggregate"] = np.nan
            row["norm_average"] = np.nan

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No landscape files found for split={split!r}.")
    return pd.DataFrame(rows)
