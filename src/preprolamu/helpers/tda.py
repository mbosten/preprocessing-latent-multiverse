from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mask_infinities(array: np.ndarray) -> np.ndarray:
    """
    Masks infinities from persistence intervals as returned by simplex_tree.persistence_intervals_in_dimension.
    """
    masked = array[array[:, 1] < np.inf]
    return masked
