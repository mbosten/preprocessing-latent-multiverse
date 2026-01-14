from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from preprolamu.config import load_dataset_config
from preprolamu.io.storage import ensure_parent_dir, get_clean_dataset_path
from preprolamu.pipeline.universes import (
    DuplicateHandling,
    FeatureSubset,
    LogTransform,
    Missingness,
    Scaling,
    Universe,
)

logger = logging.getLogger(__name__)


# Load the cleaned dataset
def load_raw_dataset(dataset_id: str) -> pd.DataFrame:

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
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:

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

    # Keep only benign samples for training.
    df_train = df_train[df_train[label_col] == benign_label].reset_index(drop=True)

    logger.info(
        "Data split:\ntrain (benign only)=%d,\nvalidation (all classes)=%d,\ntest (all classes)=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    return df_train, df_val, df_test


# Drop or keep the set of features as indicated in the Universe class
def apply_feature_subset(df: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    if universe.feature_subset == FeatureSubset.ALL:
        return df

    if universe.id.startswith(
        ("ds-NF-ToN-IoT-v3", "ds-NF-UNSW-NB15-v3", "ds-NF-CICIDS2018-v3")
    ):  # These are NF-ToN-IoT-v3 specific
        special_features = [
            "IPV4_SRC_ADDR",
            "IPV4_DST_ADDR",
            "L4_SRC_PORT",
            "L4_DST_PORT",
        ]
    # Remnant of debug dataset used earlier to set up the pipeline
    elif universe.id.startswith(
        "ds-Merged"
    ):  # Just a dummy variable to keep the pipeline similar.
        special_features = ["Protocol Type"]
    else:
        raise ValueError(f"Unknown universe for feature subseting: {universe.id}")
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
        universe.id,
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
        c
        for c in df_train.select_dtypes(include="number").columns
        if c not in (label_col, "Label")
    ]

    if not numeric_cols:
        logger.info("[PREP] No numeric columns found for missingness handling.")
        return df_train, df_test

    df_train = df_train.copy()
    df_test = df_test.copy()

    # handle nans and infs
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

    # drop rows universes
    if universe.missingness == Missingness.DROP_ROWS:
        before_train = len(df_train)
        before_test = len(df_test)

        df_train = df_train.dropna(subset=numeric_cols).reset_index(drop=True)
        df_test = df_test.dropna(subset=numeric_cols).reset_index(drop=True)

        logger.info(
            "[PREP] Missingness DROP_ROWS: train %d-->%d, test %d-->%d",
            before_train,
            len(df_train),
            before_test,
            len(df_test),
        )

        return df_train, df_test

    # impute median universes.
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


def apply_log_transform(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    universe: Universe,
    ds_cfg,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if universe.log_transform == LogTransform.NONE:
        return df_train, df_val, df_test

    label_col = ds_cfg.label_column
    numeric_cols = [
        c
        for c in df_train.select_dtypes(include="number").columns
        if c not in (label_col, "Label")
    ]

    if not numeric_cols:
        return df_train, df_val, df_test

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    logger.info(
        "[PREP] Applying %s transform to %d numeric columns.",
        universe.log_transform.value,
        len(numeric_cols),
    )

    for col in numeric_cols:
        # train split motivates the choice for handling negatives
        x_train = df_train[col].astype(float)
        all_nonneg = (x_train >= 0).all()

        if universe.log_transform == LogTransform.LOG1P:
            if all_nonneg:
                # log1p on non-negative features
                for df_split in (df_train, df_val, df_test):
                    df_split[col] = np.log1p(
                        df_split[col].astype(float).clip(lower=0.0)
                    )
            else:
                # signed log1p for features with negatives
                logger.info(
                    "[PREP] Feature '%s' contains negatives; using signed log1p.", col
                )
                for df_split in (df_train, df_val, df_test):
                    x = df_split[col].astype(float)
                    df_split[col] = np.sign(x) * np.log1p(np.abs(x))

    return df_train, df_val, df_test


def fit_scaler(df_train: pd.DataFrame, universe: Universe, ds_cfg):
    # numeric features = numeric cols except label column (attack) and 'Label' if present
    numeric_cols = [
        c
        for c in df_train.select_dtypes(include="number").columns
        if c != ds_cfg.label_column and c != "Label"
    ]
    seed = universe.seed

    if not numeric_cols:
        return df_train

    if universe.scaling == Scaling.ZSCORE:
        scaler = StandardScaler()
    elif universe.scaling == Scaling.MINMAX:
        scaler = MinMaxScaler()
    elif universe.scaling == Scaling.QUANTILE:
        scaler = QuantileTransformer(output_distribution="normal", random_state=seed)
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

    logger.info(f"Preprocessing dataset for universe={universe.id}")

    path_train = universe.preprocessed_train_path()
    path_val = universe.preprocessed_validation_path()
    path_test = universe.preprocessed_test_path()
    status_path = universe.preprocessing_status_path()

    if not overwrite and path_train.exists() and path_test.exists():
        logger.info(
            "Preprocessed files for %s already exist at %s and %s. Skipping preprocessing.",
            universe.id,
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
            seed=universe.seed,
        )

        df = None  # free memory, just to be sure
        gc.collect()

        # Remove or impute missing values (e.g., nan, inf) from val and test, respectively
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

        # Log transform
        df_train, df_val, df_test = apply_log_transform(
            df_train,
            df_val,
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
        logger.exception("Preprocessing failed for universe=%s", universe.id)
        try:
            status_path.write_text("FAILED\n", encoding="utf-8")
        except OSError:
            logger.warning(
                "Could not write FAILED status to %s", status_path, exc_info=True
            )
        raise
