# src/alphacomplexbenchmarking/pipeline/preprocessing.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd  # or polars, etc.

from alphacomplexbenchmarking.pipeline.specs import RunSpec, Scaling, FeatureSubset, CatEncoding
from alphacomplexbenchmarking.io.storage import get_preprocessed_path, ensure_parent_dir

logger = logging.getLogger(__name__)


def load_raw_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Load the full raw dataset (17M x 53).
    Replace this with your real loading (e.g. parquet).
    """
    # TODO: implement actual loading
    raise NotImplementedError("load_raw_dataset must be implemented for your data.")


def apply_feature_subset(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    if spec.feature_subset == FeatureSubset.ALL:
        return df

    # Example: drop a hard-coded subset; adjust to your use case.
    special_features = ["feat_a", "feat_b"]  # TODO: real names
    logger.debug(f"Dropping special features: {special_features}")
    return df.drop(columns=[c for c in special_features if c in df.columns])


def apply_scaling(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    if spec.scaling == Scaling.NONE:
        return df

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    numeric_cols = df.select_dtypes(include="number").columns
    scaler_cls = StandardScaler if spec.scaling == Scaling.ZSCORE else MinMaxScaler
    scaler = scaler_cls()
    logger.debug(f"Applying {spec.scaling.value} scaling to {len(numeric_cols)} numeric columns.")
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled


def apply_cat_encoding(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    cat_cols = df.select_dtypes(exclude="number").columns

    if spec.cat_encoding == CatEncoding.NONE or not cat_cols:
        return df

    logger.debug(f"Applying {spec.cat_encoding.value} encoding to categorical columns: {list(cat_cols)}")

    if spec.cat_encoding == CatEncoding.ONEHOT:
        return pd.get_dummies(df, columns=cat_cols, drop_first=False)

    if spec.cat_encoding == CatEncoding.ORDINAL:
        from sklearn.preprocessing import OrdinalEncoder

        df_encoded = df.copy()
        enc = OrdinalEncoder()
        df_encoded[cat_cols] = enc.fit_transform(df[cat_cols])
        return df_encoded

    raise ValueError(f"Unknown cat_encoding: {spec.cat_encoding}")


def preprocess_variant(spec: RunSpec) -> Path:
    """
    End-to-end preprocessing for one RunSpec.
    Returns path to preprocessed dataset.
    """
    logger.info(f"Preprocessing dataset for spec={spec.to_id_string()}")
    df = load_raw_dataset(spec.dataset_id)
    df = apply_feature_subset(df, spec)
    df = apply_scaling(df, spec)
    df = apply_cat_encoding(df, spec)

    path = get_preprocessed_path(spec)
    ensure_parent_dir(path)
    # Parquet is better for big datasets
    df.to_parquet(path)
    logger.info(f"Saved preprocessed data to {path}")
    return path