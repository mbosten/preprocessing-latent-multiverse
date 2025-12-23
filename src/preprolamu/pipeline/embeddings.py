# src/preprolamu/pipeline/embeddings.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from preprolamu.io.storage import save_projected
from preprolamu.pipeline.autoencoder import (
    get_feature_matrix_from_universe,
    load_autoencoder_for_universe,
)
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def normalize_space(X, seed: int, diameter_iterations=1000):
    """
    Normalize a space based on an approximate diameter.
    """
    rng = np.random.default_rng(seed)
    subset = [rng.choice(len(X))]
    for _ in range(diameter_iterations - 1):
        distances = cdist([X[subset[-1]]], X).ravel()
        new_point = np.argmax(distances)
        subset.append(new_point)

    pairwise_distances = cdist(X[subset], X[subset])
    diameter = np.max(pairwise_distances)

    eps = 1e-8
    if not np.isfinite(diameter) or diameter < eps:
        logger.info(
            "[EMB] Computed diameter is non-finite or too small; defaulting to 1 (diameter=%s).",
            diameter,
        )
        diameter = 1.0

    return X / diameter, diameter


def project_PCA(normalized_latent_space: np.ndarray, n_components: int, seed: int):
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(normalized_latent_space)
    logger.info(f"[EMB] PCA projection shape: {projected.shape}")
    return projected


def downsample_latent(X: np.ndarray, target_size: int, seed: int):
    n = X.shape[0]
    target = min(target_size, n) if target_size > 0 else n
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=target, replace=False)
    downsampled = X[indices]
    logger.info(f"[EMB] Downsampled latent shape: {downsampled.shape}")
    return downsampled


def from_latent_to_point_cloud(
    X: np.ndarray,
    pca_dim: int,
    target_size: int,
    seed: int,
    normalize: bool = True,
    save_projected_to: Optional[tuple[Universe, str]] = None,
    save_projected_raw_to: Optional[tuple[Universe, str]] = None,
    dtype: type = np.float32,
):
    # Store unnormalized PCA projection
    if save_projected_raw_to is not None:
        u, split = save_projected_raw_to
        X_pca_raw = project_PCA(X, n_components=pca_dim, seed=seed)
        save_projected(
            universe=u, split=f"{split}_raw", arr=X_pca_raw.astype(dtype, copy=False)
        )

    diameter = None
    if normalize:
        X, diameter = normalize_space(X, diameter_iterations=1000, seed=seed)

    X_pca = project_PCA(X, n_components=pca_dim, seed=seed)

    if save_projected_to is not None:
        universe, split = save_projected_to
        save_projected(
            universe=universe, split=split, arr=X_pca.astype(dtype, copy=False)
        )

    if target_size < X_pca.shape[0]:
        X_pca_sample = downsample_latent(X_pca, target_size=target_size, seed=seed)
    else:
        X_pca_sample = X_pca

    return X_pca_sample, diameter


def compute_embeddings_for_universe(universe: Universe):
    logger.info(f"[EMB] Computing embeddings for universe={universe.id}")

    X, _, ds_cfg = get_feature_matrix_from_universe(universe)

    # Load AE model
    logger.info("[EMB] Loading autoencoder model.")
    model = load_autoencoder_for_universe(universe, ds_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        latent_tensor = model.encode(X_tensor)
        latent = latent_tensor.cpu().numpy()

    logger.info(f"[EMB] Computed latent representation with shape {latent.shape}")

    return latent
