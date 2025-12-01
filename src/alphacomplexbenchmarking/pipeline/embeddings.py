# src/alphacomplexbenchmarking/pipeline/embeddings.py
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import torch

from alphacomplexbenchmarking.config import load_dataset_config, DatasetConfig
from alphacomplexbenchmarking.pipeline.universes import Universe
from alphacomplexbenchmarking.pipeline.autoencoder import load_autoencoder_for_universe, _get_feature_matrix_for_ae
from alphacomplexbenchmarking.io.storage import (
    get_preprocessed_path,
    get_embedding_path,
    save_numpy_array,
)

logger = logging.getLogger(__name__)


def normalize_space(
        X,
        diameter_iterations=1000,
        seed=42,
    ):
        """
        Normalize a space based on an approximate diameter.

        Parameters:
        - X : np.ndarray
            The input space to be normalized.
        - diameter_iterations : int, optional
            The number of iterations to approximate the space diameter. Default is 1000.
        - seed : int, optional
            Seed for the random number generator. Default is 42.

        Returns:
        - np.ndarray
            The normalized space based on approximate diameter.
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
    logger.info(f"[EMB] Computing embeddings for universe={universe.to_id_string()}")

    preprocessed_path = get_preprocessed_path(universe)
    df = pd.read_parquet(preprocessed_path)

    ds_cfg = load_dataset_config(universe.dataset_id)
    X = _get_feature_matrix_for_ae(df, ds_cfg)

    # Load AE model
    logger.debug("[EMB] Loading autoencoder model.")
    model = load_autoencoder_for_universe(universe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
         X_tensor = torch.from_numpy(X).to(device)
         latent_tensor = model.encode(X_tensor)
         latent = latent_tensor.cpu().numpy()
    
    logger.info(f"[EMB] Computed latent representation with shape {latent.shape}")


    normalized_latent = normalize_space(latent, diameter_iterations=1000, seed=42)

    # PCA to low dimension
    pca = PCA(n_components=universe.pca_dim)
    latent_pca = pca.fit_transform(normalized_latent)
    logger.debug(f"[EMB] PCA projection shape: {latent_pca.shape}")

    # Subsample for TDA
    n = latent_pca.shape[0]
    target = min(universe.tda_config.subsample_size, n)
    rng = np.random.default_rng(universe.seed)
    indices = rng.choice(n, size=target, replace=False)
    points_for_tda = latent_pca[indices]

    logger.info(
        f"[EMB] Subsampled {target} points (from {n}) for TDA, dim={universe.pca_dim}"
    )

    emb_path = get_embedding_path(universe)
    save_numpy_array(emb_path, points_for_tda)
    return emb_path