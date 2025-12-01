# src/alphacomplexbenchmarking/io/storage.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import logging
import json
from typing import Any, Dict

from .run_id import make_run_id
from alphacomplexbenchmarking.pipeline.sim import generate_simulation_matrix
from alphacomplexbenchmarking.pipeline.persistence import compute_alpha_complex_persistence
from alphacomplexbenchmarking.pipeline.landscapes import compute_landscapes
from alphacomplexbenchmarking.pipeline.universes import Universe

logger = logging.getLogger(__name__)

# Probably obsolete
RAW_ROOT = Path("data/raw")
INTERIM_ROOT = Path("data/interim")

# new vars
BASE_DATA_DIR = Path("data")


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


# SAVE 'N LOAD
def save_matrix(data: np.ndarray, run_id: str) -> Path:
    ensure_dir(RAW_ROOT)
    path = raw_matrix_path(run_id)
    np.savez_compressed(path, data=data)
    return path


def load_matrix(run_id: str) -> np.ndarray:
    path = raw_matrix_path(run_id)
    return np.load(path)["data"]


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


# ---------- convenience pipeline wrappers with storage ----------
def generate_and_store(
    n_samples: int,
    n_dims: int,
    seed: int,
) -> tuple[np.ndarray, str, Path]:
    """
    1) generate matrix
    2) build run_id
    3) save to data/raw/<run_id>.npz
    Returns (data, run_id, path).
    """
    data = generate_simulation_matrix(n_samples, n_dims, seed)
    run_id = make_run_id(seed, n_samples, n_dims)
    path = save_matrix(data, run_id)
    return data, run_id, path

def compute_and_store_persistence_for_run(
    data: np.ndarray,
    run_id: str,
    homology_dimensions: list[int],
) -> tuple[dict[int, np.ndarray], Path]:
    """
    1) compute persistence from in-memory data
    2) save to data/interim/persistence/<run_id>.npz
    """
    per_dim = compute_alpha_complex_persistence(data, homology_dimensions)
    path = save_persistence(per_dim, run_id)
    return per_dim, path

def compute_and_store_landscapes_for_run(
    persistence_per_dimension: dict[int, np.ndarray],
    run_id: str,
    homology_dimensions: list[int],
    num_landscapes: int = 5,
    resolution: int = 1000,
) -> tuple[dict[int, np.ndarray | None], Path]:
    """
    1) compute landscapes
    2) save to data/interim/landscapes/<run_id>.npz
    """
    landscapes = compute_landscapes(
        persistence_per_dimension,
        num_landscapes=num_landscapes,
        resolution=resolution,
        homology_dimensions=homology_dimensions,
    )
    path = save_landscapes(landscapes, run_id)
    return landscapes, path