# src/alphacomplexbenchmarking/pipeline/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


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


def compute_metrics_from_tda(
        persistence_per_dimension: Dict[int, np.ndarray],
        landscapes_per_dimension: Dict[int, Optional[np.ndarray]]):
    total_persistence_per_dim: Dict[int, float] = {}
    landscape_l2_per_dim: Dict[int, float] = {}

    for dim, intervals in persistence_per_dimension.items():
        total_persistence_per_dim[dim] = compute_total_persistence(intervals)

    for dim, landscapes in landscapes_per_dimension.items():
        if landscapes is None:
            landscape_l2_per_dim[dim] = 0.0
        else:
            # landscapes might be shape (1, num_landscapes, resolution) or similar
            landscape_l2_per_dim[dim] = float(np.linalg.norm(landscapes))

    return MetricsResult(
        total_persistence_per_dim=total_persistence_per_dim,
        landscape_l2_per_dim=landscape_l2_per_dim,
    )

