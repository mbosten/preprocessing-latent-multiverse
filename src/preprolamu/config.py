from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import typer
import yaml
from project_utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


DatasetConfig = dict[str, Any]


# set up logging.
@app.callback()
def main():
    setup_logging(
        log_dir=Path("logs"),
        suppress_loggers=[
            "PIL",
            "matplotlib.font_manager",
            "matplotlib.texmanager",
            "matplotlib.dviread",
        ],
    )
    logger.info("CLI started ...")


def load_dataset_config(dataset_id: str) -> DatasetConfig:
    config_path = Path("config") / "datasets" / f"{dataset_id}.yml"

    if not config_path.exists():
        raise FileNotFoundError("Dataset config not found: %s", config_path)

    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    cfg["dataset_id"] = cfg.get("dataset_id", dataset_id)
    cfg["raw_path"] = Path(cfg["raw_path"])
    cfg["clean_path"] = Path("data") / "raw" / f"{dataset_id}_clean.parquet"
    cfg["non_numerical_columns"] = list(cfg.get("non_numerical_columns", []) or [])
    cfg["features_to_exclude"] = list(cfg.get("features_to_exclude", []) or [])
    cfg["label_column"] = cfg.get("label_column")
    cfg["benign_label"] = cfg.get("benign_label")
    cfg["label_classes"] = list(cfg.get("label_classes", []) or []) or None
    cfg["autoencoder"] = cfg.get("autoencoder", {}) or {}

    return cfg


# load OG data
def load_raw_source(cfg: DatasetConfig) -> pd.DataFrame:
    path = path = cfg["raw_path"]

    logger.info(f"[PREP] Loading raw source from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Raw source file not found at {path}")

    if path.suffix.lower() == ".csv":

        pl.Config.set_verbose(True)
        with pl.Config(verbose=True):
            logger.info("[PREP] Scanning csv with Polars")
            lf = pl.scan_csv(path)
            logger.info("[PREP] Collecting LazyFrame")
            df_polars = lf.collect()
            logger.info("[PREP] Converting Polars to Pandas")
            df = df_polars.to_pandas()
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported source_format: {path.suffix.lower()}")

    logger.info(f"[PREP] Loaded raw data with shape {df.shape}")
    return df


# drop columns if these are indicated in config (but aint the case)
def apply_drop_columns(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    cols_to_drop = [c for c in cfg["features_to_exclude"] if c in df.columns]
    if cols_to_drop:
        logger.info(f"[PREP] Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


# cat encoding
def apply_one_time_categorical_encoding(
    df: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    df_out = df.copy()

    label_col = cfg["label_column"]
    non_num_cols = cfg["non_numerical_columns"]

    cols_to_encode: list[str] = [
        c for c in non_num_cols if c in df_out.columns and c != label_col
    ]

    if not cols_to_encode:
        logger.info("[PREP] No categorical columns to encode in one-time setup.")
        return df_out

    changed: list[str] = []

    for col in cols_to_encode:
        old_dtype = df_out[col].dtype

        df_out[col] = df_out[col].astype("category")
        codes = df_out[col].cat.codes
        logger.info("[PREP] Category codes range %s-%s", codes.min(), codes.max())
        df_out[col] = codes.astype("int32")

        new_dtype = df_out[col].dtype
        changed.append(f"{col}: {old_dtype} -> {new_dtype}")

    logger.info(
        "[PREP] One-time categorical encoding applied to %d columns: %s",
        len(changed),
        ", ".join(changed),
    )

    return df_out


# Fastai function copied from https://github.com/fastai/fastai/blob/main/fastai/tabular/core.py#L99 to avoid installing the whole library
def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    "Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion."

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int16, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint16, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }
    if obj2cat:
        typemap["object"] = (
            "category"  # User wants to categorify dtype('Object'), which may not always save space
        )
    else:
        excl_types.add("object")

    new_dtypes = {}

    def _exclude_dtype(dt):
        return dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(_exclude_dtype, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t
    return new_dtypes


# Fastai function copied to avoid installing the whole library. Thank you fastai!
def df_shrink(df, skip=[], obj2cat=True, int2uint=False):
    "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
    dt = df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
    logger.info("[PREP] Shrunk dtypes of dataframe columns: %s", dt)
    return df.astype(dt)


def compute_dataset_profile(
    df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    label_col = cfg.get("label_column")
    features_to_exclude = cfg.get("features_to_exclude", []) or []

    profile: dict[str, dict[str, Any]] = {}

    for feature_subset in ["all", "without_confounders"]:
        if feature_subset == "without_confounders":
            df_subset = df.drop(columns=features_to_exclude, errors="ignore")
        else:
            df_subset = df

        numeric_cols = [
            col
            for col in df_subset.select_dtypes(include="number").columns
            if col not in {label_col, "Label"}
        ]

        numeric = df_subset[numeric_cols].replace([np.inf, -np.inf], np.nan)

        profile[feature_subset] = {
            "has_duplicates": bool(df_subset.duplicated().any()),
            "has_missing_numeric": bool(numeric.isna().any().any()),
        }

    return profile


def save_dataset_profile(
    dataset_id: str,
    profile: dict[str, dict[str, Any]],
) -> None:
    path = Path("data") / "metadata" / f"{dataset_id}_profile.yml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(profile, file, sort_keys=False)

    logger.info("[PREP] Saved dataset profile to %s", path)


def save_clean_dataset(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    out_path = cfg["clean_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[PREP] Saving cleaned dataset to %s", out_path)
    df.to_parquet(out_path)
    logger.info("[PREP] Saved cleaned dataset with shape %s", df.shape)


@app.command()
def prepare_dataset(
    dataset_id: str = typer.Argument(
        ..., help="Dataset id to prepare, e.g. NF-CICIDS2018-v3"
    ),
) -> None:

    cfg = load_dataset_config(dataset_id)
    logger.info("[PREP] Preparing dataset_id=%s", cfg["dataset_id"])

    df = load_raw_source(cfg)
    df = df_shrink(df, obj2cat=False, int2uint=False)
    df = apply_drop_columns(df, cfg)
    df = apply_one_time_categorical_encoding(df, cfg)

    profile = compute_dataset_profile(df, cfg)

    save_clean_dataset(df, cfg)
    save_dataset_profile(cfg["dataset_id"], profile)

    df = None
    gc.collect()


if __name__ == "__main__":
    app()
