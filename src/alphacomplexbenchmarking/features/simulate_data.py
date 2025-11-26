import numpy as np


def generate_simulation_matrix(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    """Generate a random matrix for simulation."""
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, n_dims))