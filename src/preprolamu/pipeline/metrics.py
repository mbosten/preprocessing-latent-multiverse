# src/preprolamu/pipeline/metrics.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from preprolamu.io.storage import load_tda_output_for_universe
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """
    Simple scalar summaries of TDA, per homology dimension.
    Extend this as needed.
    """

    total_persistence_per_dim: Dict[int, float]
    landscape_l2_per_dim: Dict[int, float]


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
    landscapes: List[Dict[int, np.array]], return_norms: bool = False
):
    if not landscapes:
        raise ValueError("landscapes list is empty.")

    N = len(landscapes)
    max_homology_dimension = max(landscapes[0].keys())

    landscape_norms: List[Dict[int, float]] = [
        compute_landscape_norm(L, score_type="separate") for L in landscapes
    ]

    landscape_norm_means = {
        i: sum(L[i] for L in landscape_norms) / N
        for i in range(max_homology_dimension + 1)
    }
    if return_norms:
        return landscape_norm_means, landscape_norms
    else:
        return landscape_norm_means


def compute_presto_variance(
    landscapes: List[Dict[int, np.ndarray]],
) -> float:

    if not landscapes:
        raise ValueError("landscapes list is empty.")

    N = len(landscapes)
    homology_dims = range(max(landscapes[0]))

    homology_dims = list(homology_dims)

    mean_norms, norms_per_landscape = compute_landscape_norm_means(
        landscapes, return_norms=True
    )

    dim_sums = 0.0

    for d in homology_dims:
        if d not in mean_norms:
            continue
        mu_d = mean_norms[d]
        for L_norms in norms_per_landscape:
            if d not in L_norms:
                continue
            dim_sums += (L_norms[d] - mu_d) ** 2

    return dim_sums / N


def compute_presto_variance_across_universes(
    universes: Iterable[Universe],
    homology_dims: Iterable[int] | None = (0, 1, 2),
) -> float:
    """
    Load landscapes for all given universes and compute PRESTO-style variance
    of landscape norms across them.

    Each universe must already have TDA results saved via `save_metrics_from_tda_output`.
    """
    landscapes_list: List[Dict[int, np.ndarray]] = []

    for u in universes:
        _, landscapes = load_tda_output_for_universe(u)
        landscapes_list.append(landscapes)

    var = compute_presto_variance(
        landscapes=landscapes_list,
        homology_dims=homology_dims,
    )
    logger.info(
        "[TDA] Computed PRESTO variance across %d universes: %.6f",
        len(landscapes_list),
        var,
    )
    return var


def compute_metrics_from_tda(
    persistence_per_dimension: Dict[int, np.ndarray],
    landscapes_per_dimension: Dict[int, Optional[np.ndarray]],
):
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
