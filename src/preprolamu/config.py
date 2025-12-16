# src/preprolamu/config.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer
import yaml

from preprolamu.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer()


@dataclass
class AutoencoderConfig:
    latent_dim: int
    hidden_dims: tuple[int, ...]
    epochs: int
    batch_size: int
    dropout: float
    regularization: float


@dataclass
class DatasetConfig:
    dataset_id: str
    raw_path: Path
    non_numerical_columns: List[str]
    features_to_exclude: Optional[List[str]] = None
    label_column: Optional[str] = None
    benign_label: Optional[str] = None
    label_classes: Optional[List[str]] = None
    autoencoder: AutoencoderConfig = AutoencoderConfig

    @property
    def output_path(self) -> Path:
        return Path("data") / "raw" / f"{self.dataset_id}_clean.parquet"


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging",
    ),
):
    """
    Global CLI options, executed before any subcommand.
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_dir=Path("logs"), level=level)
    logger = logging.getLogger(__name__)
    logger.debug("CLI started with verbose=%s", verbose)


def load_dataset_config(dataset_id: str) -> DatasetConfig:
    """
    Load dataset-specific config from config/datasets/{dataset_id}.yml
    """
    config_path = Path("config") / "datasets" / f"{dataset_id}.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {config_path}. "
            f"Create it to specify dataset-specific settings for {dataset_id}."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    ae_raw = raw_cfg.get("autoencoder", {}) or {}

    # Default to Ton-IoT-v3 AE config if not specified
    ae_cfg = AutoencoderConfig(
        latent_dim=ae_raw.get("latent_dim", 12),
        hidden_dims=tuple(ae_raw.get("hidden_dims", (26,))),
        epochs=ae_raw.get("epochs", 10),
        batch_size=ae_raw.get("batch_size", 256),
        dropout=ae_raw.get("dropout", 0.0377),
        regularization=ae_raw.get("regularization", 0.0019),
    )

    cfg = DatasetConfig(
        dataset_id=raw_cfg.get("dataset_id", dataset_id),
        raw_path=Path(raw_cfg["raw_path"]),
        non_numerical_columns=list(raw_cfg.get("non_numerical_columns", []) or []),
        features_to_exclude=list(raw_cfg.get("features_to_exclude", []) or []),
        label_column=raw_cfg.get("label_column"),
        benign_label=raw_cfg.get("benign_label"),
        label_classes=list(raw_cfg.get("label_classes", [])) or None,
        autoencoder=ae_cfg,
    )
    return cfg


# ----------------- Core cleaning logic ----------------- #
def load_raw_source(cfg: DatasetConfig) -> pd.DataFrame:
    path = cfg.raw_path

    logger.info(f"[PREP] Loading raw source from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Raw source file not found at {path}")

    if path.suffix.lower() == ".csv":
        # df = pd.read_csv(path)
        pl.Config.set_verbose(True)
        with pl.Config(verbose=True):
            logger.debug("[PREP] Scanning csv with Polars")
            lf = pl.scan_csv(path)
            logger.debug("[PREP] Collecting LazyFrame")
            df_polars = lf.collect()
            logger.debug("[PREP] Converting Polars to Pandas")
            df = df_polars.to_pandas()
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported source_format: {path.suffix.lower()}")

    logger.info(f"[PREP] Loaded raw data with shape {df.shape}")
    return df


def apply_drop_columns(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    cols_to_drop = [c for c in cfg.features_to_exclude if c in df.columns]
    if cols_to_drop:
        logger.info(f"[PREP] Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def apply_one_time_categorical_encoding(
    df: pd.DataFrame, cfg: DatasetConfig
) -> pd.DataFrame:
    df_out = df.copy()

    label_col = cfg.label_column
    non_num_cols = cfg.non_numerical_columns or []

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
        codes = df_out[col].cat.codes  # int64 by default
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


# THis function is likely redundant due to categorical encoding above.
def apply_dtypes(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Convert columns listed in non_numerical_columns to pandas string dtype.
    Leave all other columns unchanged, and log which columns actually changed dtype.
    """
    df_out = df.copy()

    # Record original dtypes
    original_dtypes = df_out.dtypes.to_dict()

    changed_cols: list[str] = []
    non_num_cols = cfg.non_numerical_columns

    for col in non_num_cols:
        if col not in df_out.columns:
            logger.warning(
                "[PREP] non-numerical column '%s' listed in config but not found in dataset.",
                col,
            )
            continue

        old_dtype = original_dtypes.get(col)
        df_out[col] = df_out[col].astype("string")
        new_dtype = df_out[col].dtype

        if new_dtype != old_dtype:
            changed_cols.append(f"{col}: {old_dtype} -> {new_dtype}")

    if changed_cols:
        logger.info(
            "[PREP] Changed dtypes for %d non-numerical columns: %s",
            len(changed_cols),
            ", ".join(changed_cols),
        )
    else:
        logger.info("[PREP] No dtype changes were applied to non-numerical columns.")

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


def apply_remove_infinite(df: pd.DataFrame) -> pd.DataFrame:

    df_out = df.copy()

    # Work only on numeric columns
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    if not len(num_cols):
        logger.info("[PREP] No numeric columns found; skipping infinity check.")
        return df_out

    # Build mask of inf/-inf
    inf_mask = np.isinf(df_out[num_cols])
    total_inf = int(inf_mask.values.sum())

    if total_inf == 0:
        logger.info("[PREP] No +/- infinity values found in numeric columns.")
        return df_out

    # Log per-column counts
    col_counts = inf_mask.sum()
    detailed_counts = ", ".join(
        f"{col}: {int(cnt)}" for col, cnt in col_counts.items() if cnt > 0
    )

    logger.info(
        "[PREP] Found %d +/- infinity values in numeric columns. Breakdown: %s",
        total_inf,
        detailed_counts,
    )

    # Replace infinities with NaN, only in numeric cols
    df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], np.nan)

    return df_out


def apply_remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing NaN values.
    Logs total NaN count and per-column NaN counts before removal.
    """
    df_out = df.copy()

    # Count NaNs per column
    nan_counts = df_out.isna().sum()
    total_nans = int(nan_counts.sum())

    if total_nans == 0:
        logger.info("[PREP] No NaN values found in dataset.")
        return df_out

    # Build detailed log string
    detailed_counts = ", ".join(
        f"{col}: {int(cnt)}" for col, cnt in nan_counts.items() if cnt > 0
    )

    logger.info(
        "[PREP] Found %d NaN values across %d columns. Breakdown: %s",
        total_nans,
        (nan_counts > 0).sum(),
        detailed_counts,
    )

    # Drop rows with any NaN
    before = df_out.shape[0]
    df_out = df_out.dropna()
    after = df_out.shape[0]

    logger.info(
        "[PREP] Dropped %d rows containing NaN values. New shape: %s",
        before - after,
        df_out.shape,
    )

    return df_out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[PREP] Removing duplicate rows from dataset with shape %s.", df.shape)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("[PREP] Removed duplicates. New shape: %s", df.shape)
    return df


def save_clean_dataset(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    out_path = cfg.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[PREP] Saving cleaned dataset to {out_path}")
    df.to_parquet(out_path)
    logger.info(f"[PREP] Saved cleaned dataset with shape {df.shape}")


def prepare_dataset(dataset_id: str) -> None:
    cfg = load_dataset_config(dataset_id)
    logger.info(f"[PREP] Preparing dataset_id={cfg.dataset_id}")

    df = load_raw_source(cfg)
    df = df_shrink(df, obj2cat=False, int2uint=False)
    df = apply_drop_columns(df, cfg)
    # df = apply_dtypes(df, cfg)
    df = apply_one_time_categorical_encoding(df, cfg)
    # Functions below will be incorporated into the multiverse pipeline
    # df = apply_remove_infinite(df)
    # df = apply_remove_nans(df)
    # df = remove_duplicates(df)

    save_clean_dataset(df, cfg)


# ----------------- Typer CLI ----------------- #
# uv run setup initiate Merged35
@app.command()
def initiate(
    dataset_id: str = typer.Argument(..., help="Dataset id to prepare, e.g. base_v1"),
):
    """
    One-time cleaning: read raw file, fix dtypes, save to data/raw/{dataset_id}.parquet.
    """

    prepare_dataset(dataset_id)


if __name__ == "__main__":
    app()
