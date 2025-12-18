# src/preprolamu/io/storage.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


# Data base directory
BASE_DATA_DIR = Path("data")

# Experiment directories
EXPERIMENTS_ROOT = BASE_DATA_DIR / "experiments"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------- low-level save/load helpers ----------


# PATHS
def get_raw_dataset_path(dataset_id: str, extension: str) -> Path:
    return BASE_DATA_DIR / "raw" / f"{dataset_id}.{extension}"


def get_clean_dataset_path(dataset_id: str, extension: str) -> Path:
    return BASE_DATA_DIR / "raw" / f"{dataset_id}_clean.{extension}"


def get_preprocessed_train_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR
        / "processed"
        / "train"
        / f"{universe.to_id_string()}_preprocessed_train.parquet"
    )


def get_preprocessed_validation_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR
        / "processed"
        / "validation"
        / f"{universe.to_id_string()}_preprocessed_validation.parquet"
    )


def get_preprocessed_test_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR
        / "processed"
        / "test"
        / f"{universe.to_id_string()}_preprocessed_test.parquet"
    )


def get_preprocessing_status_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR
        / "interim"
        / "preprocessing_status"
        / f"{universe.to_id_string()}.status"
    )


def get_ae_model_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR / "interim" / "autoencoder" / f"{universe.to_id_string()}_ae.pt"
    )


def get_embedding_path(universe: Universe, split: str = "test") -> Path:
    """
    Path for the PCA-projected embedding used for TDA.
    """
    return (
        BASE_DATA_DIR
        / "interim"
        / "embeddings"
        / f"{universe.to_id_string()}_latent_{split}.npy"
    )


def get_tda_result_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR / "interim" / "persistence" / f"{universe.to_id_string()}_tda.npz"
    )


def get_metrics_path(universe: Universe) -> Path:
    return (
        BASE_DATA_DIR
        / "processed"
        / "metrics"
        / f"{universe.to_id_string()}_metrics.json"
    )


def get_latent_cache_path(universe: Universe) -> Path:
    root = ensure_dir(EXPERIMENTS_ROOT / "latent")
    return root / f"{universe.to_id_string()}_latent.npy"


def save_metrics_from_tda_output(
    universe: Universe,
    per_dim: dict[int, np.ndarray],
    landscapes: dict[int, np.ndarray | None],
    metrics: dict[str, float],
):
    tda_path = get_tda_result_path(universe)
    arrays_for_npz = {}
    for d, arr in per_dim.items():
        arrays_for_npz[f"dim{d}_intervals"] = arr
    for d, ls in landscapes.items():
        if ls is not None:
            arrays_for_npz[f"dim{d}_landscapes"] = ls

    save_tda_npz(tda_path, **arrays_for_npz)

    metrics_path = get_metrics_path(universe)
    metrics_dict = asdict(metrics)
    save_json(metrics_path, metrics_dict)
    logger.info(
        "[TDA] Saved TDA results to %s and metrics to %s", tda_path, metrics_path
    )


def load_tda_output_for_universe(
    universe: Universe,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray | None]]:
    """
    Inverse of `save_metrics_from_tda_output`:
    load per-dimension persistence intervals and landscapes for a universe.

    Returns:
        per_dim:    dim -> intervals array
        landscapes: dim -> landscapes array (or None if not present)
    """
    tda_path = get_tda_result_path(universe)
    raw = load_tda_npz(tda_path)  # dict[str, np.ndarray]

    per_dim: Dict[int, np.ndarray] = {}
    landscapes: Dict[int, np.ndarray | None] = {}

    for key, arr in raw.items():
        if key.startswith("dim") and key.endswith("_intervals"):
            # key like "dim1_intervals" -> extract 1
            dim_str = key[3:-10]  # strip "dim" and "_intervals"
            dim = int(dim_str)
            per_dim[dim] = arr
        elif key.startswith("dim") and key.endswith("_landscapes"):
            # key like "dim1_landscapes"
            dim_str = key[3:-11]  # strip "dim" and "_landscapes"
            dim = int(dim_str)
            landscapes[dim] = arr

    return per_dim, landscapes


def save_tda_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_parent_dir(path)
    logger.info(f"Saving TDA npz to {path} with keys {list(arrays.keys())}")
    np.savez(path, **arrays)


def load_tda_npz(path: Path) -> Dict[str, np.ndarray]:
    logger.info(f"Loading TDA npz from {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    logger.info(f"Saving JSON to {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    logger.info(f"Loading JSON from {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
