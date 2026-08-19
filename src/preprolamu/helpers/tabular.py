from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from preprolamu.config import load_dataset_config
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def load_split(universe: Universe, split: str, *, benign_only: bool = False):
    config = load_dataset_config(universe.dataset_id)
    df = pd.read_parquet(universe.paths.preprocessed(split=split))

    if benign_only:
        label = config["label_column"]
        df = df[df[label] == config["benign_label"]]

    return df, config


def feature_matrix(df: pd.DataFrame, label_column: str) -> np.ndarray:
    return (df.drop(columns=label_column).to_numpy(dtype=np.float32))


def labels(df: pd.DataFrame, label_column: str) -> np.ndarray:
    return df[label_column].to_numpy()