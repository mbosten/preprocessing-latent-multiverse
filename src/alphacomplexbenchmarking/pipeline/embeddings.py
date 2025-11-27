# src/alphacomplexbenchmarking/pipeline/embeddings.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from alphacomplexbenchmarking.pipeline.specs import RunSpec
from alphacomplexbenchmarking.pipeline.autoencoder import load_autoencoder_for_variant
from alphacomplexbenchmarking.io.storage import (
    get_preprocessed_path,
    get_embedding_path,
    save_numpy_array,
)

logger = logging.getLogger(__name__)


def compute_embeddings_and_subsample_for_tda(spec: RunSpec) -> Path:
    """
    For this RunSpec:
      - load preprocessed data,
      - use trained encoder to compute embeddings,
      - normalize embeddings,
      - PCA to spec.pca_dim,
      - subsample TDA points according to spec.tda_config.subsample_size,
      - save resulting N x d array for TDA.

    Returns path to the saved embedding array.
    """
    logger.info(f"[EMB] Computing embeddings for spec={spec.to_id_string()}")

    preprocessed_path = get_preprocessed_path(spec)
    df = pd.read_parquet(preprocessed_path)
    X = df.to_numpy(dtype=float)

    # Load AE model (once implemented)
    logger.debug("[EMB] Loading autoencoder model (NotImplemented placeholder).")
    # model = load_autoencoder_for_variant(spec)
    # latent = model.encode(X)  # shape (N, latent_dim)
    # For now: just pretend the preprocessed data IS the embedding:
    latent = X

    logger.debug(f"[EMB] Got latent representation with shape {latent.shape}")

    # Normalize (z-score)
    latent_mean = latent.mean(axis=0, keepdims=True)
    latent_std = latent.std(axis=0, keepdims=True) + 1e-8
    latent_norm = (latent - latent_mean) / latent_std

    # PCA to low dimension
    pca = PCA(n_components=spec.pca_dim)
    latent_pca = pca.fit_transform(latent_norm)
    logger.debug(f"[EMB] PCA projection shape: {latent_pca.shape}")

    # Subsample for TDA
    n = latent_pca.shape[0]
    target = min(spec.tda_config.subsample_size, n)
    rng = np.random.default_rng(spec.seed)
    indices = rng.choice(n, size=target, replace=False)
    points_for_tda = latent_pca[indices]

    logger.info(
        f"[EMB] Subsampled {target} points (from {n}) for TDA, dim={spec.pca_dim}"
    )

    emb_path = get_embedding_path(spec)
    save_numpy_array(emb_path, points_for_tda)
    return emb_path