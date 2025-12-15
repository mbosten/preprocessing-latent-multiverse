# src/preprolamu/pipeline/embeddings.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from preprolamu.io.storage import get_embedding_path, save_numpy_array
from preprolamu.pipeline.autoencoder import (
    get_feature_matrix_from_universe,
    load_autoencoder_for_universe,
)
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def normalize_space(X, diameter_iterations=1000, seed=42):
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
    return X / diameter


def project_PCA(normalized_latent_space: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components)
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
    X: np.ndarray, pca_dim: int, target_size: int, seed: int, normalize: bool = True
):
    if normalize:
        X = normalize_space(X, diameter_iterations=1000, seed=42)

    X_pca = project_PCA(X, n_components=pca_dim)

    if target_size < X_pca.shape[0]:
        X_pca_sample = downsample_latent(X_pca, target_size=target_size, seed=seed)
    else:
        X_pca_sample = X_pca

    return X_pca_sample


def compute_embeddings_for_universe(universe: Universe):
    logger.info(f"[EMB] Computing embeddings for universe={universe.to_id_string()}")

    X, ds_cfg = get_feature_matrix_from_universe(universe)

    # Load AE model
    logger.debug("[EMB] Loading autoencoder model.")
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


def compute_embeddings_and_subsample_for_tda(universe: Universe) -> Path:
    """
    For this Universe:
      - load preprocessed data,
      - use trained encoder to compute embeddings,
      - normalize embeddings,
      - PCA to universe.pca_dim,
      - subsample TDA points according to universe.tda_config.subsample_size,
      - save resulting N x d array for TDA.

    Returns path to the saved embedding array.
    """
    latent = compute_embeddings_for_universe(universe)

    points_for_tda = from_latent_to_point_cloud(
        latent,
        pca_dim=universe.pca_dim,
        target_size=universe.tda_config.subsample_size,
        seed=universe.seed,
        normalize=True,
    )

    emb_path = get_embedding_path(universe)
    save_numpy_array(emb_path, points_for_tda)
    return emb_path
