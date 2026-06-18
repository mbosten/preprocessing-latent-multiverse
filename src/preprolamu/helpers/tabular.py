from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def labels_from_df(df: pd.DataFrame, label_col: str) -> np.ndarray | None:
    if label_col not in df.columns:
        return None
    y = df[label_col].astype(str).to_numpy()
    return y


def feature_matrix_from_df(df: pd.DataFrame, label_col: str) -> np.ndarray:
    cols_to_drop = [label_col]
    if "Label" in df.columns and "Label" not in cols_to_drop:
        cols_to_drop.append("Label")

    df_features = df.drop(columns=cols_to_drop, errors="ignore")
    X = df_features.to_numpy(dtype=np.float32)
    return X
