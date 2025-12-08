# src/alphacomplexbenchmarking/pipeline/create_embeddings.py
from __future__ import annotations

import logging
import numpy as np
import torch
import pandas as pd
from typing import List, Dict
from alphacomplexbenchmarking.config import load_dataset_config, DatasetConfig
from alphacomplexbenchmarking.pipeline.universes import Universe
from alphacomplexbenchmarking.pipeline.autoencoder import train_autoencoder_for_universe, load_autoencoder_for_universe, _get_feature_matrix_for_ae
from alphacomplexbenchmarking.io.storage import (
    get_latent_cache_path,
    get_preprocessed_path
)


logger = logging.getLogger(__name__)


def get_or_compute_latent(
    universe: Universe,
    retrain_if_missing: bool = True,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Canonical function for:
      Universe -> latent embeddings (full dataset, no PCA, no subsampling).

    Behavior:
      - if cached latent exists and not force_recompute: load and return
      - otherwise: ensure AE trained, compute latent, save cache, return
    """
    cache_path = get_latent_cache_path(universe)

    if cache_path.exists() and not force_recompute:
        logger.info("[Embedding] Loading cached latent from %s for %s", cache_path, universe)
        return np.load(cache_path)

    logger.info("[Embedding] Computing latent for %s (force_recompute=%s)", universe, force_recompute)

    # 1. Optionally (re)train AE
    if retrain_if_missing:
        logger.info("[Embedding] Training AE for universe %s (if needed).", universe)
        train_autoencoder_for_universe(universe)

    # 2. Load preprocessed data
    ds_cfg: DatasetConfig = load_dataset_config(universe.dataset_id)
    preprocessed_path = get_preprocessed_path(universe)
    logger.info("[Embedding] Loading preprocessed data from %s", preprocessed_path)
    df = pd.read_parquet(preprocessed_path)

    # 3. Feature matrix
    X = _get_feature_matrix_for_ae(df, ds_cfg)

    # 4. Encode with AE
    model = load_autoencoder_for_universe(universe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("[Embedding] Encoding %d samples into latent space.", X.shape[0])
    with torch.no_grad():
        tensor_X = torch.from_numpy(X).to(device)
        latent_tensor = model.encode(tensor_X)
        latent = latent_tensor.cpu().numpy()

    # 5. Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, latent)
    logger.info("[Embedding] Saved latent cache to %s", cache_path)

    return latent


def collect_latents_for_universes(
    universes: List[Universe],
    retrain_if_missing: bool = True,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Return a dict: universe_id_string -> latent array.
    Useful for PRESTO-style analyses where you want a list/dict of embeddings.
    """
    latents: Dict[str, np.ndarray] = {}
    for u in universes:
        uid = u.to_id_string()
        latents[uid] = get_or_compute_latent(
            u,
            retrain_if_missing=retrain_if_missing,
            force_recompute=force_recompute,
        )
    return latents