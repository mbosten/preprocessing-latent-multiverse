# src/preprolamu/pipeline/preprocessing.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from preprolamu.config import load_dataset_config
from preprolamu.io.storage import (
    ensure_parent_dir,
    get_clean_dataset_path,
    get_preprocessed_test_path,
    get_preprocessed_train_path,
)
from preprolamu.pipeline.universes import CatEncoding, FeatureSubset, Scaling, Universe

logger = logging.getLogger(__name__)


# Load the cleaned dataset
def load_raw_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Load the cleaned dataset (all classes) as a pandas DataFrame.
    """
    path = get_clean_dataset_path(dataset_id, extension="parquet")
    logger.info("Loading full cleaned dataset from %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows x %d columns.", *df.shape)
    logger.debug("Column dtypes: %s", dict(df.dtypes))
    return df


def split_train_test(
    df: pd.DataFrame,
    label_col: str,
    benign_label: str,
    train_frac: float = 0.7,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random train/test split, stratified by 'Attack' if present,
    otherwise by the label column.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in dataframe.")

    stratify_col = df["Attack"] if "Attack" in df.columns else df[label_col]

    df_train, df_test = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify_col,
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Keep only benign samples for training. Otherwise preprocessing is affected by soft leakage.
    df_train = df_train[df_train[label_col] == benign_label].reset_index(drop=True)

    logger.info(
        "Data split into train (benign-only: %d rows) and test (all classes: %d rows).",
        len(df_train),
        len(df_test),
    )

    return df_train, df_test


# Drop or keep the set of features as indicated in the Universe class
def apply_feature_subset(df: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    if universe.feature_subset == FeatureSubset.ALL:
        return df

    if universe.to_id_string().startswith(
        ("ds-NF-ToN-IoT-v3", "ds-NF-UNSW-NB15-v3", "df-NF-CICIDS2018-v3")
    ):  # These are NF-ToN-IoT-v3 specific
        special_features = [
            "IPV4_SRC_ADDR",
            "IPV4_DST_ADDR",
            "L4_SRC_PORT",
            "L4_DST_PORT",
        ]
    elif universe.to_id_string().startswith(
        "ds-Merged"
    ):  # Just a dummy variable to keep the pipeline similar.
        special_features = ["Protocol Type"]
    else:
        raise ValueError(
            f"Unknown universe for feature subseting: {universe.to_id_string()}"
        )
    logger.debug(f"Dropping special features: {special_features}")
    return df.drop(columns=[c for c in special_features if c in df.columns])


def fit_scaler(df_train: pd.DataFrame, universe: Universe, ds_cfg):
    # numeric features = numeric cols except label
    numeric_cols = [
        c
        for c in df_train.select_dtypes(include="number").columns
        if c != ds_cfg.label_column
    ]

    if not numeric_cols:
        return df_train

    if universe.scaling == Scaling.ZSCORE:
        scaler = StandardScaler()
    elif universe.scaling == Scaling.MINMAX:
        scaler = MinMaxScaler()
    elif universe.scaling == Scaling.ROBUST:
        scaler = RobustScaler()
    elif universe.scaling == Scaling.QUANTILE:
        scaler = QuantileTransformer(output_distribution="normal")
    else:
        raise ValueError(f"Unknown scaling: {universe.scaling}")

    logger.debug(
        "Applying %s scaling to %d numeric columns.",
        universe.scaling.value,
        len(numeric_cols),
    )

    scaler.fit(df_train[numeric_cols])
    return scaler, numeric_cols


def transform_with_scaler(df: pd.DataFrame, scaler, numeric_cols) -> pd.DataFrame:
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
    return df_scaled


def fit_cat_encoder(df_train: pd.DataFrame, universe: Universe, ds_cfg):
    cat_cols = [
        c
        for c in df_train.select_dtypes(exclude="number").columns
        if c != ds_cfg.label_column
    ]

    if not cat_cols or universe.cat_encoding is None:
        return None, []

    if universe.cat_encoding == CatEncoding.ONEHOT:
        return None, cat_cols

    if universe.cat_encoding == CatEncoding.LABEL:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        enc.fit(df_train[cat_cols])
        return enc, cat_cols

    raise ValueError(f"Unknown cat_encoding: {universe.cat_encoding}")


def transform_with_cat_encoder(
    df: pd.DataFrame, universe: Universe, encoder, cat_cols
) -> pd.DataFrame:
    if universe.cat_encoding is None or not cat_cols:
        return df

    df_encoded = df.copy()

    logger.debug(
        "Applying %s encoding to categorical columns: %s",
        universe.cat_encoding.value,
        list(cat_cols),
    )

    if universe.cat_encoding == CatEncoding.ONEHOT:
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=False)
        return df_encoded

    if universe.cat_encoding == CatEncoding.LABEL and encoder is not None:
        df_encoded[cat_cols] = encoder.transform(df[cat_cols])
        return df_encoded
    raise ValueError(f"Unknown cat_encoding: {universe.cat_encoding}")


# Orchestrator function to call in cli.py
def preprocess_variant(universe: Universe) -> Path:
    """
    End-to-end preprocessing for one Universe.
    Returns path to preprocessed dataset.
    """
    logger.info(f"Preprocessing dataset for universe={universe.to_id_string()}")
    df = load_raw_dataset(universe.dataset_id)
    ds_cfg = load_dataset_config(universe.dataset_id)
    df = apply_feature_subset(df, universe)
    df_train, df_test = split_train_test(
        df,
        label_col=ds_cfg.label_column,
        benign_label=ds_cfg.benign_label,
        train_frac=0.7,
        seed=42,
    )

    df = None  # free memory, just to be sure

    scaler, numeric_cols = fit_scaler(df_train, universe, ds_cfg)
    encoder, cat_cols = fit_cat_encoder(df_train, universe, ds_cfg)

    df_train = transform_with_scaler(df_train, scaler, numeric_cols)
    df_train = transform_with_cat_encoder(df_train, universe, encoder, cat_cols)

    df_test = transform_with_scaler(df_test, scaler, numeric_cols)
    df_test = transform_with_cat_encoder(df_test, universe, encoder, cat_cols)

    path_train = get_preprocessed_train_path(universe)
    path_test = get_preprocessed_test_path(universe)

    ensure_parent_dir(path_train)
    ensure_parent_dir(path_test)

    df_train.to_parquet(path_train)
    df_test.to_parquet(path_test)

    logger.info("Saved preprocessed TRAIN data to %s", path_train)
    logger.info("Saved preprocessed TEST data to %s", path_test)

    return path_train, path_test
