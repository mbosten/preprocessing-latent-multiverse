
import numpy as np

def generate_simulation_matrix(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    """
    Generate a simulation matrix with random values.

    Parameters:
    n_samples (int): Number of samples (rows).
    n_dims (int): Number of dimensions (columns).
    seed (int): Seed for the random number generator.

    Returns:
    np.ndarray: A matrix of shape (n_samples, n_dims) with random values.
    """
    rng = np.random.default_rng(seed)
    simulation_matrix = rng.random((n_samples, n_dims))
    return simulation_matrix
