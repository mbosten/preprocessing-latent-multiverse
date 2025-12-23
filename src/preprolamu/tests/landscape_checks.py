# src/preprolamu/tests/landscape_checks.py
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# Identify which landscapes have extremely high norms
def top_landscape_outliers(
    landscapes_list: list[dict[int, np.ndarray]],
    dims: list[int],
    top_k: int = 10,
):
    rows = []
    for i, L in enumerate(landscapes_list):
        for d in dims:
            arr = L.get(d)
            if arr is None:
                continue
            n = float(np.linalg.norm(arr))
            rows.append((d, n, i))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]
