# src/alphacomplexbenchmarking/pipeline/persistence.py
import numpy as np
import gudhi as gd



def mask_infinities(array: np.ndarray) -> np.ndarray:
    return array[array[:, 1] < np.inf]


def compute_alpha_complex_persistence(
    data: np.ndarray,
    homology_dimensions: list[int] = [0, 1, 2]
) -> dict[int, np.ndarray]:
    """
    Compute alpha-complex persistence for the given point cloud.
    Returns a dict dim -> intervals (numpy array).
    """
    alpha_complex = gd.AlphaComplex(points=data, precision="fast")
    st = alpha_complex.create_simplex_tree()
    st.compute_persistence()

    per_dim: dict[int, np.ndarray] = {}
    for dim in homology_dimensions:
        per_dim[dim] = mask_infinities(st.persistence_intervals_in_dimension(dim))
    return per_dim
