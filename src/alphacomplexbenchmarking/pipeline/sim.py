# src/alphacomplexbenchmarking/pipeline/sim.py
import numpy as np


def generate_simulation_matrix(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    """Generate a random matrix for simulation."""
    rng = np.random.default_rng(seed)
    data = rng.random((n_samples, n_dims))
    return data
