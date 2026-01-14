from __future__ import annotations

import logging

import numpy as np
import torch

from preprolamu.io.storage import load_embedding
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

    if not force_recompute:
        try:
            return load_embedding(
                universe, split=split, force_recompute=force_recompute
            )
        except FileNotFoundError:
            logger.info(
                "[Embedding] No existing embedding (%s split) found for %s; will compute.",
                split,
                universe.id,
            )

    latent_path = universe.embedding_path(split=split)

    # Retrieve model path to see if a checkpoint exists.
    model_path = universe.ae_model_path()

    # Feature matrix
    X, _, ds_cfg = get_feature_matrix_from_universe(universe, split=split)

    logger.info(
        "[Embedding] Retrieved feature matrix of shape %s for %s (%s split).",
        X.shape,
        universe.id,
        split,
    )

    # If model does not exist yet, or training should be overwritten.
    if not model_path.exists() or retrain_regardless:
        train_autoencoder_for_universe(universe)

    model = load_autoencoder_for_universe(universe, ds_cfg)

    # Encode with trained AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Found device: %s", device)
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
