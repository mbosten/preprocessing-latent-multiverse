# src/preprolamu/pipeline/create_embeddings.py
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch

from preprolamu.io.storage import get_ae_model_path, get_embedding_path
from preprolamu.pipeline.autoencoder import (
    get_feature_matrix_from_universe,
    load_autoencoder_for_universe,
    train_autoencoder_for_universe,
)
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def get_or_compute_latent(
    universe: Universe,
    split: str = "test",
    retrain_regardless: bool = False,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Canonical function for:
      Universe -> latent embeddings (full dataset, no PCA, no subsampling).

    - retrain_regardless: Should a new AE be trained, regardless of existing saved checkpoints?
    - force_recompute: Should the AE be trained/evaluated again, regardless of an existing embedding space?

    Behavior:s
      - if cached latent exists and not force_recompute: load and return
      - otherwise: ensure AE trained, compute latent, save cache, return
    """
    # Specify split-specific file name depending on which dataset we retrieve the embedding space from.
    latent_path = get_embedding_path(universe, split=split)

    if latent_path.exists() and not force_recompute:
        logger.info(
            "[Embedding] Loading cached latent (%s) from %s for %s",
            split,
            latent_path,
            universe.to_id_string(),
        )
        return np.load(latent_path)

    # Retrieve model path to see if a checkpoint exists.
    model_path = get_ae_model_path(universe)

    # If model does not exist yet, or training should be overwritten.
    if not model_path.exists() or retrain_regardless:
        train_autoencoder_for_universe(universe)

    # Feature matrix
    X, ds_cfg = get_feature_matrix_from_universe(universe, split=split)

    model = load_autoencoder_for_universe(universe, ds_cfg)

    # Encode with trained AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info(
        "[Embedding] Encoding %d samples into latent space (%s split).",
        X.shape[0],
        split,
    )

    with torch.no_grad():
        tensor_X = torch.from_numpy(X).to(device)
        latent_tensor = model.encode(tensor_X)
        latent = latent_tensor.cpu().numpy()

    # 5. Cache
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(latent_path, latent)
    logger.info("[Embedding] Saved latent (%s split) to %s", split, latent_path)

    return latent


# CURRENTLY NOT USED, REDUNDANT OR DEPRECATED
def collect_latents_for_universes(
    universes: List[Universe],
    retrain_regardless: bool = False,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Write embedding space to object for list of universes.
    """
    latents: Dict[str, np.ndarray] = {}
    for u in universes:
        uid = u.to_id_string()
        latents[uid] = get_or_compute_latent(
            u,
            retrain_regardless=retrain_regardless,
            force_recompute=force_recompute,
        )
    return latents
