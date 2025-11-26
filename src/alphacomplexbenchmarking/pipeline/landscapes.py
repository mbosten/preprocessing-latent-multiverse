# src/alphacomplexbenchmarking/pipeline/landscapes.py
import numpy as np
from gudhi.representations import Landscape


def compute_landscapes(
    persistence_per_dimension: dict[int, np.ndarray], 
    num_landscapes: int = 5,
    resolution: int = 1000, 
    homology_dimensions: list[int] = [0, 1, 2]
) -> dict[int, np.ndarray | None]:
    """
    Compute persistence landscapes per dimension.
    Returns a dict dim -> landscapes array or None if no intervals.
    """
    LS = Landscape(resolution=resolution, keep_endpoints=False, num_landscapes=num_landscapes)

    landscapes_per_dimension: dict[int, np.ndarray | None] = {}

    for dim in homology_dimensions:
        persistence_pairs = persistence_per_dimension.get(dim, [])
        if len(persistence_pairs) == 0:
            landscapes_per_dimension[dim] = None
            continue

        landscapes_per_dimension[dim] = LS.fit_transform([persistence_pairs])

    return landscapes_per_dimension