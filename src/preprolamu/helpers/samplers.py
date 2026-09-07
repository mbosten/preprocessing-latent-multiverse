from __future__ import annotations

import numpy as np


def torus_surface(n, R, r, seed=42):
    """Sample n points from the surface of a 3D torus."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2*np.pi, n)
    v = rng.uniform(0, 2*np.pi, n)
    return np.column_stack((
        (R + r*np.cos(v)) * np.cos(u),
        (R + r*np.cos(v)) * np.sin(u),
        r * np.sin(v)
    ))


def d_torus(n, d, radii=None, seed=42, return_angles=False):
    """Sample n points from a d-torus embedded in 2*d dimensions."""
    rng = np.random.default_rng(seed)

    radii = np.ones(d) if radii is None else np.asarray(radii)
    # Sample angles uniformly
    theta = rng.uniform(0, 2*np.pi, size=(n, d))

    X = np.stack((radii*np.cos(theta), radii*np.sin(theta)), axis=-1).reshape(n, -1)
    return (X, theta) if return_angles else X


def swiss_roll(n, turns=1.5, noise=0.0, seed=42):
    """Sample n points from a Swiss roll embedded in 3D."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5*np.pi, (1.5 + 2*turns)*np.pi, n)
    X = np.column_stack((t*np.cos(t), rng.uniform(-1, 1, n), t*np.sin(t)))
    X /= X.std(axis=0)
    return X + rng.normal(0, noise, X.shape)


def hypercube_surface(n: int, d: int, half_side: float = 1.0, seed: int | None = 42) -> np.ndarray:
    """Sample n points uniformly from the surface of a d-dimensional hypercube."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-half_side, half_side, size=(n, d))
    X[np.arange(n), rng.integers(0, d, size=n)] = rng.choice((-half_side, half_side), n)
    return X


def d_gaussian(n, d, mean=None, std=1.0, seed=42):
    """Sample n points from a d-dimensional Gaussian distribution."""
    rng = np.random.default_rng(seed)

    return rng.normal(np.zeros(d) if mean is None else mean, std, (n, d))


def d_gaussian_correlated(n, d, seed=None):
    """Sample n points from a d-dimensional Gaussian with random covariance."""
    rng = np.random.default_rng(seed)
    
    A = rng.normal(size=(d, d))
    return rng.multivariate_normal(np.zeros(d), A @ A.T, n)
