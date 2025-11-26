# src/alphacomplexbenchmarking/pipeline/sim.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

def generate_simulation_matrix(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    """Generate a random matrix for simulation."""
    logger.debug(f"Generating simulation matrix with n_samples={n_samples}, n_dims={n_dims}, seed={seed}")
    rng = np.random.default_rng(seed)
    data = rng.random((n_samples, n_dims))
    logger.debug(f"Generated matrix shape: {data.shape}")
    return data
