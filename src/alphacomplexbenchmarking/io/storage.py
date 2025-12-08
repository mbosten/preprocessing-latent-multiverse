# src/alphacomplexbenchmarking/io/storage.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import logging
import json
from typing import Any, Dict

from alphacomplexbenchmarking.pipeline.universes import Universe

logger = logging.getLogger(__name__)

# Probably obsolete
RAW_ROOT = Path("data/raw")
INTERIM_ROOT = Path("data/interim")

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


def raw_matrix_path(run_id: str) -> Path:
    return RAW_ROOT / f"{run_id}.npz"


def persistence_path(run_id: str) -> Path:
    return INTERIM_ROOT / "persistence" / f"{run_id}.npz"


def landscapes_path(run_id: str) -> Path:
    return INTERIM_ROOT / "landscapes" / f"{run_id}.npz"


def get_preprocessed_path(universe: Universe) -> Path:
    return BASE_DATA_DIR / "processed" / f"{universe.to_id_string()}_preprocessed.parquet"


def get_ae_model_path(universe: Universe) -> Path:
    return BASE_DATA_DIR / "interim" / "autoencoder" / f"{universe.to_id_string()}_ae.pt"


def get_embedding_path(universe: Universe) -> Path:
    """
    Path for the PCA-projected embedding used for TDA.
    """
    return BASE_DATA_DIR / "interim" / "embeddings" / f"{universe.to_id_string()}_pca.npy"


def get_tda_result_path(universe: Universe) -> Path:
    return BASE_DATA_DIR / "interim" / "persistence" / f"{universe.to_id_string()}_tda.npz"


def get_metrics_path(universe: Universe) -> Path:
    return BASE_DATA_DIR / "processed" / "metrics" / f"{universe.to_id_string()}_metrics.json"

def get_latent_cache_path(universe: Universe) -> Path:
    root = ensure_dir(EXPERIMENTS_ROOT / "latent")
    return root / f"{universe.to_id_string()}_latent.npy"


# SAVE 'N LOAD
def save_matrix(data: np.ndarray, run_id: str) -> Path:
    ensure_dir(RAW_ROOT)
    path = raw_matrix_path(run_id)
    np.savez_compressed(path, data=data)
    return path


def load_matrix(run_id: str) -> np.ndarray:
    path = raw_matrix_path(run_id)
    return np.load(path)["data"]

def load_latent_from_cache(universe: Universe) -> np.ndarray:
    cache_path = get_latent_cache_path(universe)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No latent cache found for {universe}. Expected at {cache_path}. "
            "Run the embedding preparation command first."
        )
    logger.info("[Embedding] Loading latent from %s for %s", cache_path, universe)
    return np.load(cache_path)

def save_persistence(per_dim: dict[int, np.ndarray], run_id: str) -> Path:
    out_dir = ensure_dir(INTERIM_ROOT / "persistence")
    path = out_dir / f"{run_id}.npz"
    np.savez_compressed(path, **{f"dim{d}": arr for d, arr in per_dim.items()})
    return path


def load_persistence(run_id: str) -> dict[int, np.ndarray]:
    path = persistence_path(run_id)
    data = np.load(path)
    out: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("dim"):
            dim = int(key[3:])
            out[dim] = data[key]
    return out


def save_landscapes(landscapes: dict[int, np.ndarray | None], run_id: str) -> Path:
    out_dir = ensure_dir(INTERIM_ROOT / "landscapes")
    path = out_dir / f"{run_id}.npz"
    np.savez_compressed(
        path,
        **{f"dim{d}": arr for d, arr in landscapes.items() if arr is not None},
    )
    return path


def save_numpy_array(path: Path, array: np.ndarray) -> None:
    ensure_parent_dir(path)
    logger.debug(f"Saving numpy array with shape {array.shape} to {path}")
    np.save(path, array)


def load_numpy_array(path: Path) -> np.ndarray:
    logger.debug(f"Loading numpy array from {path}")
    return np.load(path)


def save_tda_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_parent_dir(path)
    logger.debug(f"Saving TDA npz to {path} with keys {list(arrays.keys())}")
    np.savez(path, **arrays)


def load_tda_npz(path: Path) -> Dict[str, np.ndarray]:
    logger.debug(f"Loading TDA npz from {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    logger.debug(f"Saving JSON to {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    logger.debug(f"Loading JSON from {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
