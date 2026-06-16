from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# This should be moved to utilities
def mask_infinities(array: np.ndarray) -> np.ndarray:
    masked = array[array[:, 1] < np.inf]
    return masked
