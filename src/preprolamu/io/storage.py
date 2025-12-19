# src/preprolamu/io/storage.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

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


def get_latent_cache_path(universe: Universe) -> Path:
    root = ensure_dir(EXPERIMENTS_ROOT / "latent")
    return root / f"{universe.to_id_string()}_latent.npy"


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


def get_persistence_path(universe: Universe, split: str = "test") -> Path:
    return (
        BASE_DATA_DIR
        / "interim"
        / "persistence"
        / f"{universe.to_id_string()}_persistence_{split}.npz"
    )


def get_landscapes_path(universe: Universe, split: str = "test") -> Path:
    return (
        BASE_DATA_DIR
        / "interim"
        / "landscapes"
        / f"{universe.to_id_string()}_landscapes_{split}.npz"
    )


def get_metrics_path(universe: Universe, split: str = "test") -> Path:
    return (
        BASE_DATA_DIR
        / "processed"
        / "metrics"
        / f"{universe.to_id_string()}_metrics_{split}.json"
    )


# IO functions
def load_embedding(
    universe: Universe,
    split: str,
    force_recompute: bool = False,
):
    """
    Loads embedding from disk. If the file is not found or if force_recompute=True,
    raises FileNotFoundError.
    """
    embed_path = get_embedding_path(universe, split=split)
    if embed_path.exists() and not force_recompute:
        logger.info(
            "[Embedding] Loading cached latent (%s) from %s for %s",
            split,
            embed_path,
            universe.to_id_string(),
        )
        return np.load(embed_path)
    raise FileNotFoundError(
        f"No embedding found at {embed_path} for {universe.to_id_string()}"
    )


def save_tda_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_parent_dir(path)
    logger.info("Saving npz to %s with keys %s", path, list(arrays.keys()))
    np.savez(path, **arrays)


def load_tda_npz(path: Path) -> Dict[str, np.ndarray]:
    logger.info("Loading npz from %s", path)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ---------- Persistence ----------


def save_persistence(
    universe: Universe,
    split: str,
    per_dim: dict[int, np.ndarray],
) -> None:
    path = get_persistence_path(universe, split=split)
    arrays = {f"dim{d}_intervals": arr for d, arr in per_dim.items()}
    save_tda_npz(path, **arrays)


def load_persistence(
    universe: Universe,
    split: str = "test",
) -> dict[int, np.ndarray]:
    path = get_persistence_path(universe, split=split)
    raw = load_tda_npz(path)
    per_dim: dict[int, np.ndarray] = {}
    for key, arr in raw.items():
        if key.startswith("dim") and key.endswith("_intervals"):
            dim_str = key[3:-10]  # strip "dim" and "_intervals"
            dim = int(dim_str)
            per_dim[dim] = arr
    if not per_dim:
        raise FileNotFoundError(f"No persistence intervals found in {path}")
    return per_dim


# ---------- Landscapes ----------


def save_landscapes(
    universe: Universe,
    split: str,
    landscapes: dict[int, np.ndarray | None],
) -> None:
    path = get_landscapes_path(universe, split=split)
    arrays = {
        f"dim{d}_landscapes": arr for d, arr in landscapes.items() if arr is not None
    }
    save_tda_npz(path, **arrays)


def load_landscapes(
    universe: Universe,
    split: str = "test",
) -> dict[int, np.ndarray | None]:
    path = get_landscapes_path(universe, split=split)
    raw = load_tda_npz(path)
    landscapes: dict[int, np.ndarray | None] = {}
    for key, arr in raw.items():
        if key.startswith("dim") and key.endswith("_landscapes"):
            dim_str = key[3:-11]  # strip "dim" and "_landscapes"
            dim = int(dim_str)
            landscapes[dim] = arr
    if not landscapes:
        raise FileNotFoundError(f"No landscapes found in {path}")
    return landscapes


# ---------- Metrics ----------


def save_metrics(
    universe: Universe,
    split: str,
    metrics: Any,
) -> None:
    path = get_metrics_path(universe, split=split)
    if hasattr(metrics, "__dataclass_fields__"):
        payload = asdict(metrics)
    else:
        payload = dict(metrics)
    save_json(path, payload)


def load_metrics(
    universe: Universe,
    split: str = "test",
) -> Dict[str, Any]:
    path = get_metrics_path(universe, split=split)
    return load_json(path)


# ---------- JSON ----------


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    logger.info(f"Saving JSON to {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    logger.info(f"Loading JSON from {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
