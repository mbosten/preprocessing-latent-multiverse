from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import typer
import yaml
import json
from project_utils import setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer()


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


CONFIG_DIR = Path("config") / "datasets"
CLEAN_DIR = Path("data") / "raw"
PROFILE_PATH = Path("data/interim/metadata/profiles.json")

DatasetConfig = dict[str, Any]


def load_dataset_config(dataset_id: str) -> DatasetConfig:
    path = CONFIG_DIR / f"{dataset_id}.yml"

    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return {
        "dataset_id": dataset_id,
        "raw_path": Path(config["raw_path"]),
        "label_column": config.get("label_column"),
        "benign_label": config.get("benign_label"),
        "categorical_columns": config.get("categorical_columns", []),
        "confounders": config.get("confounders", []),
        "autoencoder": config.get("autoencoder", {}),
    }


def load_raw(path: Path) -> pd.DataFrame:
    match path.suffix.lower():
        case ".csv":
            return pl.read_csv(path).to_pandas()
        case ".parquet":
            return pd.read_parquet(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix.lower()}")


def encode_categoricals(df: pd.DataFrame, columns: list[str]):
    df = df.copy()

    for column in columns:
        if column in df:
            df[column] = df[column].astype("category").cat.codes.astype("int32")

    return df


def profile(df: pd.DataFrame, config: DatasetConfig) -> dict[str, dict[str, bool]]:
    variants = {
        "all": df,
        "without_confounders": df.drop(
            columns=config.get("confounders", []),
            errors="ignore"
        ),
    }

    return {
        name: {
            "has_duplicates": bool(frame.duplicated().any()),
            "has_missing_numeric": has_missing_numeric(
                frame,
                config["label_column"],
            ),
        }
        for name, frame in variants.items()
    }


def has_missing_numeric(df: pd.DataFrame, label_col: str):
    numeric = df.select_dtypes(include="number").drop(columns=label_col, errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return bool(numeric.isna().any().any())


def update_profiles(dataset_id, profile: dict[str, dict[str, bool]]):
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    profiles = (
        json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        if PROFILE_PATH.exists()
        else {}
    )

    profiles[dataset_id] = profile
    PROFILE_PATH.write_text(json.dumps(profiles, indent=4), encoding="utf-8")


@app.command()
def prepare_dataset(dataset_id: str = typer.Argument(..., help="Dataset id to prepare, e.g. NF-CICIDS2018-v3")):
    config = load_dataset_config(dataset_id)

    df = load_raw(config["raw_path"])
    df = df.drop(columns="Attack", errors="ignore") 
    df = encode_categoricals(df, config["categorical_columns"])

    dataset_profile = profile(df, config)

    clean_path = CLEAN_DIR / f"{dataset_id}_clean.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_path, index=False)

    update_profiles(dataset_id, dataset_profile)
    logger.info("Prepared %s: %d rows, %d columns", dataset_id, *df.shape)


if __name__ == "__main__":
    app()
