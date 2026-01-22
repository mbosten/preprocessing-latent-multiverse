from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .paths import Split

if TYPE_CHECKING:
    from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseIO:
    u: Universe

    def load_metrics(self, split: Split = "test") -> dict[str, Any]:
        path = self.u.paths.metrics(split)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_metrics(self, split: Split, metrics: Any) -> None:
        path = self.u.paths.metrics(split)
        if hasattr(metrics, "__dataclass_fields__"):
            payload = asdict(metrics)
        else:
            payload = dict(metrics)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_landscapes(self, split: Split = "test") -> dict[int, np.ndarray]:
        path = self.u.paths.landscapes(split)
        with np.load(path) as data:
            raw = {k: data[k] for k in data.files}

        landscapes: dict[int, np.ndarray] = {}
        for key, arr in raw.items():
            if key.startswith("dim") and key.endswith("_landscapes"):
                dim = int(key[3:-11])  # "dim{d}_landscapes"
                landscapes[dim] = arr

        if not landscapes:
            raise FileNotFoundError(f"No landscapes found in {path}")
        return landscapes

    def save_landscapes(
        self, split: Split, landscapes: dict[int, np.ndarray | None]
    ) -> None:
        path = self.u.paths.landscapes(split)
        arrays = {
            f"dim{d}_landscapes": arr
            for d, arr in landscapes.items()
            if arr is not None
        }
        logger.info("Saving npz to %s with keys %s", path, list(arrays.keys()))
        np.savez(path, **arrays)

    def load_persistence(self, split: Split = "test") -> dict[int, np.ndarray]:
        path = self.u.paths.persistence(split)
        with np.load(path) as data:
            raw = {k: data[k] for k in data.files}

        per_dim: dict[int, np.ndarray] = {}
        for key, arr in raw.items():
            if key.startswith("dim") and key.endswith("_intervals"):
                dim_str = key[3:-10]  # strip "dim" and "_intervals"
                dim = int(dim_str)
                per_dim[dim] = arr
        if not per_dim:
            raise FileNotFoundError(f"No persistence intervals found in {path}")
        return per_dim

    def save_persistence(self, split: Split, per_dim: dict[int, np.ndarray]) -> None:
        path = self.u.paths.persistence(split)

        arrays = {f"dim{d}_intervals": arr for d, arr in per_dim.items()}
        logger.info("Saving npz to %s with keys %s", path, list(arrays.keys()))
        np.savez(path, **arrays)

    def save_projected(self, split: Split, normalized: bool, arr: np.ndarray) -> None:
        path = self.u.paths.projected(split=split, normalized=normalized)
        logger.info("[IO] Saving projected point cloud (%s) to %s", split, path)
        np.save(path, arr)

    def load_embedding(self, split: Split, force_recompute: bool = False) -> np.ndarray:
        path = self.u.paths.embedding(split)
        if path.exists() and not force_recompute:
            logger.info(
                "[Embedding] Loading cached latent (%s) from %s for %s",
                split,
                path,
                self.u.id,
            )
            return np.load(path)
        raise FileNotFoundError(f"No embedding found at {path} for {self.u.id}")
