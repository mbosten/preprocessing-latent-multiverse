from __future__ import annotations

import logging

import gudhi as gd
import numpy as np

logger = logging.getLogger(__name__)


def mask_infinities(array: np.ndarray) -> np.ndarray:
    logger.info(f"Masking infinities in array with shape {array.shape}")
    masked = array[array[:, 1] < np.inf]
    logger.info(f"Resulting masked array shape: {masked.shape}")
    return masked


def build_alpha_complex_simplex_tree(points: np.ndarray) -> gd.SimplexTree:

    logger.info(f"Computing alpha complex persistence for data of shape {points.shape}")
    alpha_complex = gd.AlphaComplex(points=points, precision="safe")
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.compute_persistence()
    return simplex_tree


def compute_alpha_complex_persistence(
    data: np.ndarray, homology_dimensions: list[int] = [0, 1, 2]
):

    st = build_alpha_complex_simplex_tree(data)

    logger.info(f"Computed persistence with {len(st.persistence_pairs())} intervals")

    per_dim: dict[int, np.ndarray] = {}
    for dim in homology_dimensions:
        per_dim[dim] = mask_infinities(st.persistence_intervals_in_dimension(dim))
        logger.info(f"Dim {dim}: {per_dim[dim].shape[0]} intervals after masking")
    return per_dim
