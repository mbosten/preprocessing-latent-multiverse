# src/preprolamu/io/storage.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
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


def get_latent_cache_path(universe: Universe) -> Path:
    root = ensure_dir(EXPERIMENTS_ROOT / "latent")
    return root / f"{universe.to_id_string()}_latent.npy"


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
    embed_path = universe.embedding_path(split=split)
    if embed_path.exists() and not force_recompute:
        logger.info(
            "[Embedding] Loading cached latent (%s) from %s for %s",
            split,
            embed_path,
            universe.id,
        )
        return np.load(embed_path)
    raise FileNotFoundError(f"No embedding found at {embed_path} for {universe.id}")


def save_tda_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_parent_dir(path)
    logger.info("Saving npz to %s with keys %s", path, list(arrays.keys()))
    np.savez(path, **arrays)


def load_tda_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_projected(universe: Universe, split: str, arr: np.ndarray) -> None:
    path = universe.projected_path(split=split)
    ensure_parent_dir(path)
    logger.info("[IO] Saving projected point cloud (%s) to %s", split, path)
    np.save(path, arr)


def load_projected(universe: Universe, split: str, normalized: bool) -> np.ndarray:
    path = universe.projected_path(split=split, normalized=normalized)
    if path.exists():
        logger.info(
            "[IO] Loading projection (split=%s, normalized=%s) from %s",
            split,
            normalized,
            path,
        )
        return np.load(path)
    raise FileNotFoundError(f"No projected point cloud found at {path}")


# ---------- Persistence ----------


def save_persistence(
    universe: Universe,
    split: str,
    per_dim: dict[int, np.ndarray],
) -> None:
    path = universe.persistence_path(split=split)
    arrays = {f"dim{d}_intervals": arr for d, arr in per_dim.items()}
    save_tda_npz(path, **arrays)


def load_persistence(
    universe: Universe,
    split: str = "test",
) -> dict[int, np.ndarray]:
    path = universe.persistence_path(split=split)
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
    path = universe.landscapes_path(split=split)
    arrays = {
        f"dim{d}_landscapes": arr for d, arr in landscapes.items() if arr is not None
    }
    save_tda_npz(path, **arrays)


def load_landscapes(
    universe: Universe,
    split: str = "test",
) -> dict[int, np.ndarray | None]:
    path = universe.landscapes_path(split=split)
    raw = load_tda_npz(path)
    landscapes: dict[int, np.ndarray | None] = {}
    for key, arr in raw.items():
        if key.startswith("dim") and key.endswith("_landscapes"):
            dim_str = key[3:-11]  # strip "dim" and "_landscapes"
            dim = int(dim_str)
            landscapes[dim] = arr
    if not landscapes:
        print(raw)
        print(landscapes)
        raise FileNotFoundError(f"No landscapes found in {path}")
    return landscapes


# ---------- Metrics ----------


def save_metrics(
    universe: Universe,
    split: str,
    metrics: Any,
) -> None:
    path = universe.metrics_path(split=split)
    if hasattr(metrics, "__dataclass_fields__"):
        payload = asdict(metrics)
    else:
        payload = dict(metrics)
    save_json(path, payload)


def load_metrics(
    universe: Universe,
    split: str = "test",
) -> Dict[str, Any]:
    path = universe.metrics_path(split=split)
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
