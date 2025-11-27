# src/alphacomplexbenchmarking/pipeline/autoencoder.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from alphacomplexbenchmarking.pipeline.specs import RunSpec
from alphacomplexbenchmarking.io.storage import get_preprocessed_path, get_ae_model_path

logger = logging.getLogger(__name__)


def train_autoencoder_for_variant(spec: RunSpec) -> Path:
    """
    Train an autoencoder on the preprocessed data for this spec.
    For now, this is a placeholder you should replace with your
    actual PyTorch / TF training code.

    Returns path to saved model.
    """
    logger.info(f"[AE] Training autoencoder for spec={spec.to_id_string()}")

    # Load preprocessed data (you'll likely use a DataLoader instead).
    preprocessed_path = get_preprocessed_path(spec)
    df = pd.read_parquet(preprocessed_path)

    # TODO: implement actual AE model and training.
    # For now, just log and pretend we did.
    logger.warning("[AE] train_autoencoder_for_variant is a placeholder. Implement me!")

    model_path = get_ae_model_path(spec)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy save so path exists (remove once you save a real model)
    np.save(model_path.with_suffix(".npy"), np.zeros(1))
    logger.info(f"[AE] (placeholder) Saved dummy model marker to {model_path.with_suffix('.npy')}")

    return model_path


def load_autoencoder_for_variant(spec: RunSpec):
    """
    Load the trained autoencoder model for this spec.
    Implement this with your model library of choice.
    """
    # TODO: implement real model loading
    raise NotImplementedError("load_autoencoder_for_variant must be implemented with your model.")