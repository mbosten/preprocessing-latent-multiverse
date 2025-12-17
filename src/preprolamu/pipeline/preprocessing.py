# src/preprolamu/pipeline/preprocessing.py
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
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
    get_preprocessed_validation_path,
    get_preprocessing_status_path,
)
from preprolamu.pipeline.universes import (
    DuplicateHandling,
    FeatureSubset,
    Missingness,
    Scaling,
    Universe,
)

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
    logger.info("Column dtypes: %s", dict(df.dtypes))
    return df


def split_train_test(
    df: pd.DataFrame,
    label_col: str,
    benign_label: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random train/test split, stratified by 'Attack' if present,
    otherwise by the label column.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in dataframe.")

    stratify_col = df["Attack"] if "Attack" in df.columns else df[label_col]

    df_train, df_temp = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify_col,
    )

    # get correct proportions of val and test set in the remainder of the data
    remaining_frac = val_frac + (1 - train_frac - val_frac)
    relative_val_frac = val_frac / remaining_frac

    stratify_temp = (
        df_temp["Attack"] if "Attack" in df_temp.columns else df_temp[label_col]
    )

    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val_frac,
        random_state=seed,
        stratify=stratify_temp,
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Keep only benign samples for training. Otherwise preprocessing is affected by soft leakage.
    df_train = df_train[df_train[label_col] == benign_label].reset_index(drop=True)

    logger.info(
        "Data split: train_benign=%d, val_all=%d, test_all=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    return df_train, df_val, df_test


# Drop or keep the set of features as indicated in the Universe class
def apply_feature_subset(df: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    if universe.feature_subset == FeatureSubset.ALL:
        return df

    if universe.to_id_string().startswith(
        ("ds-NF-ToN-IoT-v3", "ds-NF-UNSW-NB15-v3", "ds-NF-CICIDS2018-v3")
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
    logger.info(f"Dropping special features: {special_features}")
    return df.drop(columns=[c for c in special_features if c in df.columns])


def apply_duplicate_handling(df: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    if universe.duplicate_handling == DuplicateHandling.KEEP:
        return df

    before = len(df)
    df_nodup = df.drop_duplicates()
    after = len(df_nodup)
    logger.info(
        "Dropped %d duplicate rows for universe %s",
        before - after,
        universe.to_id_string(),
    )
    return df_nodup


def apply_missingness(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    universe: Universe,
    ds_cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_col = ds_cfg.label_column

    # Numeric columns only
    numeric_cols = [
        c for c in df_train.select_dtypes(include="number").columns if c != label_col
    ]

    if not numeric_cols:
        logger.info("[PREP] No numeric columns found for missingness handling.")
        return df_train, df_test

    # ---- Replace infinities with NaN ----
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train[numeric_cols] = df_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_test[numeric_cols] = df_test[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Count missing
    missing_train_before = df_train[numeric_cols].isna().sum().sum()
    missing_test_before = df_test[numeric_cols].isna().sum().sum()

    logger.info(
        "[PREP] Missingness before handling: train=%d, test=%d",
        missing_train_before,
        missing_test_before,
    )

    # ---- DROP ROWS ----
    if universe.missingness == Missingness.DROP_ROWS:
        before_train = len(df_train)
        before_test = len(df_test)

        df_train = df_train.dropna(subset=numeric_cols).reset_index(drop=True)
        df_test = df_test.dropna(subset=numeric_cols).reset_index(drop=True)

        logger.info(
            "[PREP] Missingness DROP_ROWS: train %d→%d, test %d→%d",
            before_train,
            len(df_train),
            before_test,
            len(df_test),
        )

        return df_train, df_test

    # ---- IMPUTE MEDIAN based on training data only ----
    if universe.missingness == Missingness.IMPUTE_MEDIAN:
        medians = df_train[numeric_cols].median()

        df_train[numeric_cols] = df_train[numeric_cols].fillna(medians)
        df_test[numeric_cols] = df_test[numeric_cols].fillna(medians)

        logger.info(
            "[PREP] Missingness IMPUTE_MEDIAN: filled missing values in %d numeric cols.",
            len(numeric_cols),
        )

        return df_train, df_test

    raise ValueError(f"Unknown missingness setting: {universe.missingness}")


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

    logger.info(
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


# Orchestrator function to call in cli.py
def preprocess_variant(
    universe: Universe, overwrite: bool = False
) -> Tuple[Path, Path]:
    """
    End-to-end preprocessing for one Universe.
    Returns path to preprocessed dataset.
    """
    logger.info(f"Preprocessing dataset for universe={universe.to_id_string()}")

    path_train = get_preprocessed_train_path(universe)
    path_val = get_preprocessed_validation_path(universe)
    path_test = get_preprocessed_test_path(universe)
    status_path = get_preprocessing_status_path(universe)

    if not overwrite and path_train.exists() and path_test.exists():
        logger.info(
            "Preprocessed files for %s already exist at %s and %s. Skipping preprocessing.",
            universe.to_id_string(),
            path_train,
            path_test,
        )

        if status_path.exists():
            try:
                status_path.unlink()
            except OSError:
                logger.warning(
                    "Could not remove existing preprocessing status file at %s",
                    status_path,
                    exc_info=True,
                )

        return path_train, path_test

    ensure_parent_dir(status_path)
    status_path.write_text("IN_PROGRESS\n", encoding="utf-8")

    try:
        df = load_raw_dataset(universe.dataset_id)
        ds_cfg = load_dataset_config(universe.dataset_id)
        df = apply_feature_subset(df, universe)
        df = apply_duplicate_handling(df, universe)
        df_train, df_val, df_test = split_train_test(
            df,
            label_col=ds_cfg.label_column,
            benign_label=ds_cfg.benign_label,
            train_frac=0.6,
            val_frac=0.2,
            seed=42,
        )

        df = None  # free memory, just to be sure
        gc.collect()

        # Remove or impute missing values (e.g., NaN, inf) from val and test, respectively
        df_train, df_val = apply_missingness(
            df_train,
            df_val,
            universe,
            ds_cfg,
        )

        _, df_test = apply_missingness(
            df_train,
            df_test,
            universe,
            ds_cfg,
        )

        scaler, numeric_cols = fit_scaler(df_train, universe, ds_cfg)

        df_train = transform_with_scaler(df_train, scaler, numeric_cols)
        df_val = transform_with_scaler(df_val, scaler, numeric_cols)
        df_test = transform_with_scaler(df_test, scaler, numeric_cols)

        ensure_parent_dir(path_train)
        ensure_parent_dir(path_val)
        ensure_parent_dir(path_test)

        df_train.to_parquet(path_train)
        df_val.to_parquet(path_val)
        df_test.to_parquet(path_test)

        del df_train, df_val, df_test
        gc.collect()

        logger.info("Saved preprocessed TRAIN data to %s", path_train)
        logger.info("Saved preprocessed VALIDATION data to %s", path_val)
        logger.info("Saved preprocessed TEST data to %s", path_test)

        try:
            if status_path.exists():
                status_path.unlink()
        except OSError:
            logger.warning(
                "Could not remove status file %s after success",
                status_path,
                exc_info=True,
            )

        return path_train, path_val, path_test

    except Exception:
        logger.exception(
            "Preprocessing failed for universe=%s", universe.to_id_string()
        )
        try:
            status_path.write_text("FAILED\n", encoding="utf-8")
        except OSError:
            logger.warning(
                "Could not write FAILED status to %s", status_path, exc_info=True
            )
        raise
