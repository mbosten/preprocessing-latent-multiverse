# src/alphacomplexbenchmarking/pipeline/preprocessing.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from alphacomplexbenchmarking.pipeline.specs import RunSpec, Scaling, FeatureSubset, CatEncoding
from alphacomplexbenchmarking.io.storage import get_raw_dataset_path, get_preprocessed_path, ensure_parent_dir

logger = logging.getLogger(__name__)

LABEL_COLUMN = "Label"
TARGET_LABEL_VALUE = "BENIGN"
MAX_ROWS_FOR_LABEL = 1000

def load_raw_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Currently loads only a subset of benign traffic for debugging purposes.
    Returns a pandas DataFrame to ensure further compatibility.
    """
    logger.info(
        "Loading raw dataset from %s using Polars: %s == %r (limit=%d)",
        dataset_id,
        LABEL_COLUMN,
        TARGET_LABEL_VALUE,
        MAX_ROWS_FOR_LABEL,
    )
    path = get_raw_dataset_path(dataset_id, extension="csv")
    lf = (
        pl.scan_csv(path)
        .filter(pl.col(LABEL_COLUMN) == TARGET_LABEL_VALUE)
        .limit(MAX_ROWS_FOR_LABEL)
    )

    # Collect into an in-memory Polars DataFrame, then convert to pandas
    df_polars = lf.collect()
    df = df_polars.to_pandas()

    logger.info("Loaded %d rows x %d columns after Polars filter+limit.", *df.shape)
    logger.debug(f"Column dtypes: {dict(df.dtypes)}")
    return df


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

    if cat_cols.empty:
        return df

    logger.debug(f"Applying {spec.cat_encoding.value} encoding to categorical columns: {list(cat_cols)}")

    if spec.cat_encoding == CatEncoding.ONEHOT:
        return pd.get_dummies(df, columns=cat_cols, drop_first=False)

    if spec.cat_encoding == CatEncoding.LABEL:
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