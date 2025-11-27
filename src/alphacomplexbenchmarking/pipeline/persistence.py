# src/alphacomplexbenchmarking/pipeline/persistence.py
from __future__ import annotations
import numpy as np
import gudhi as gd
import logging

logger = logging.getLogger(__name__)


def mask_infinities(array: np.ndarray) -> np.ndarray:
    logger.debug(f"Masking infinities in array with shape {array.shape}")
    masked = array[array[:, 1] < np.inf]
    logger.debug(f"Resulting masked array shape: {masked.shape}")
    return masked


def compute_alpha_complex_persistence(
    data: np.ndarray,
    homology_dimensions: list[int] = [0, 1, 2]
) -> dict[int, np.ndarray]:
    """
    Compute alpha-complex persistence for the given point cloud.
    Returns a dict dim -> intervals (numpy array).
    """
    logger.info(f"Computing alpha complex persistence for data of shape {data.shape}, dims={homology_dimensions}")
    alpha_complex = gd.AlphaComplex(points=data, precision="fast")
    st = alpha_complex.create_simplex_tree()
    st.compute_persistence()

    per_dim: dict[int, np.ndarray] = {}
    for dim in homology_dimensions:
        per_dim[dim] = mask_infinities(st.persistence_intervals_in_dimension(dim))
        logger.debug(f"Dim {dim}: {per_dim[dim].shape[0]} intervals after masking")
    return per_dim
