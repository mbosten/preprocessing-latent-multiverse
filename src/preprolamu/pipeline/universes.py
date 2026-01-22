from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
import yaml

from preprolamu.config import load_dataset_config
from preprolamu.io.io import UniverseIO
from preprolamu.io.paths import UniversePaths

logger = logging.getLogger(__name__)

BASE_DATA_DIR = Path("data")


class Scaling(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    QUANTILE = "quantile"


class LogTransform(str, Enum):
    NONE = "none"
    LOG1P = "log1p"


class FeatureSubset(str, Enum):
    ALL = "all"
    WITHOUT_CONFOUNDERS = "without_confounders"


class DuplicateHandling(str, Enum):
    DROP = "drop"
    KEEP = "keep"


class Missingness(str, Enum):
    DROP_ROWS = "drop_rows"
    IMPUTE_MEDIAN = "impute_median"


@dataclass(frozen=True)
class TdaConfig:
    homology_dimensions: Tuple[int, ...] = (0, 1, 2)
    num_landscapes: int = 5
    resolution: int = 1000
    subsample_size: int = 500_000  # points used for TDA


@dataclass(frozen=True)
class Universe:

    dataset_id: str

    # preprocessing choices
    scaling: Scaling
    log_transform: LogTransform
    feature_subset: FeatureSubset
    duplicate_handling: DuplicateHandling
    missingness: Missingness
    seed: int

    pca_dim: int = 3

    tda_config: TdaConfig = TdaConfig()

    # Parsed ID string
    id: str = field(init=False)

    base_data_dir: Path = field(
        default=Path("data"), compare=False, hash=False, repr=False
    )

    def __post_init__(self):
        # Compute once
        object.__setattr__(
            self,
            "id",
            (
                f"ds-{self.dataset_id}"
                f"_sc-{self.scaling.value}"
                f"_log-{self.log_transform.value}"
                f"_fs-{self.feature_subset.value}"
                f"_dup-{self.duplicate_handling.value}"
                f"_miss-{self.missingness.value}"
                f"_sd-{self.seed}"
            ),
        )

    # Will likely be deprecated
    def to_id_string(self) -> str:
        return self.id

    # -------------------------------------------------------------- #
    # ----------------------- PATH FUNCTIONS ----------------------- #
    # -------------------------------------------------------------- #
    @property
    def paths(self) -> UniversePaths:
        return UniversePaths(self)

    @property
    def io(self) -> UniverseIO:
        return UniverseIO(self)

    # -------------------------------------------------------------- #
    # ------------------------ IO FUNCTIONS ------------------------ #
    # -------------------------------------------------------------- #
    # def load_metrics(self, split: str = "test") -> Dict[str, Any]:
    #     path = self.metrics_path(split=split)
    #     with path.open("r", encoding="utf-8") as f:
    #         return json.load(f)

    # def save_metrics(self, split: str, metrics: Any) -> None:
    #     path = self.metrics_path(split=split)
    #     if hasattr(metrics, "__dataclass_fields__"):
    #         payload = asdict(metrics)
    #     else:
    #         payload = dict(metrics)

    #     # Save json
    #     with path.open("w", encoding="utf-8") as f:
    #         json.dump(payload, f, indent=2)

    # def load_landscapes(self, split: str = "test") -> dict[int, np.ndarray | None]:
    #     """
    #     Read individual landscape .npz files and return as a dict.
    #     """
    #     path = self.landscapes_path(split=split)

    #     with np.load(path) as data:
    #         raw = {k: data[k] for k in data.files}

    #     landscapes: dict[int, np.ndarray | None] = {}
    #     for key, arr in raw.items():
    #         if key.startswith("dim") and key.endswith("_landscapes"):
    #             dim_str = key[3:-11]  # strip "dim" and "_landscapes"
    #             dim = int(dim_str)
    #             landscapes[dim] = arr
    #     if not landscapes:
    #         raise FileNotFoundError(f"No landscapes found in {path}")
    #     return landscapes

    # def save_landscapes(self, split: str, landscapes: dict[int, np.ndarray | None]) -> None:
    #     """
    #     Write landscape dict to individual .npz files.
    #     """
    #     path = self.landscapes_path(split=split)
    #     arrays = {
    #         f"dim{d}_landscapes": arr for d, arr in landscapes.items() if arr is not None
    #     }
    #     logger.info("Saving npz to %s with keys %s", path, list(arrays.keys()))
    #     np.savez(path, **arrays)

    def to_param_dict(self) -> Dict[str, Any]:
        """
        Flatten Universe into a dict of multiverse parameters used for grouping.
        """
        out: Dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "scaling": getattr(self.scaling, "value", self.scaling),
            "log_transform": getattr(self.log_transform, "value", self.log_transform),
            "feature_subset": getattr(
                self.feature_subset, "value", self.feature_subset
            ),
            "duplicate_handling": getattr(
                self.duplicate_handling, "value", self.duplicate_handling
            ),
            "missingness": getattr(self.missingness, "value", self.missingness),
            "seed": self.seed,
        }
        return out


DATASET_IDS: List[str] = [
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
    "NF-CICIDS2018-v3",
]


def epd(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dataset_invariants_for(universe: Universe) -> dict[str, Any]:
    cfg = load_dataset_config(universe.dataset_id)
    inv = cfg.raw_cfg.get("dataset_invariants")
    if inv is None:
        raise RuntimeError(
            f"dataset_invariants missing for {universe.dataset_id}. "
            "Run `initiate` first."
        )
    return inv[universe.feature_subset.value]


def load_dataset_yaml(dataset_id: str) -> dict[str, Any]:
    path = Path("config") / "datasets" / f"{dataset_id}.yml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prune_multiverse(universes: list[Universe]) -> list[Universe]:
    kept = []
    cache: dict[str, dict[str, Any]] = {}

    for u in universes:
        if u.dataset_id not in cache:
            yml = load_dataset_yaml(u.dataset_id)
            cache[u.dataset_id] = yml["dataset_invariants"]

        inv = cache[u.dataset_id][u.feature_subset.value]

        if not inv["has_duplicates"] and u.duplicate_handling == DuplicateHandling.DROP:
            continue
        if not inv["has_missing_numeric"] and u.missingness == Missingness.DROP_ROWS:
            continue

        kept.append(u)

    logger.info("[MV] Pruned multiverse: %d -> %d", len(universes), len(kept))
    return kept


def generate_multiverse() -> List[Universe]:

    scalings = [Scaling.ZSCORE, Scaling.MINMAX, Scaling.QUANTILE]
    log_transforms = [LogTransform.NONE, LogTransform.LOG1P]
    feature_subsets = [FeatureSubset.ALL, FeatureSubset.WITHOUT_CONFOUNDERS]
    duplicate_opts = [DuplicateHandling.KEEP, DuplicateHandling.DROP]
    missingness_opts = [Missingness.DROP_ROWS, Missingness.IMPUTE_MEDIAN]
    seeds = [42, 420, 4200, 42000]

    universes: List[Universe] = []

    for ds_id in DATASET_IDS:
        for sc in scalings:
            for log_tr in log_transforms:
                for fs in feature_subsets:
                    for dup in duplicate_opts:
                        for miss in missingness_opts:
                            for sd in seeds:
                                universes.append(
                                    Universe(
                                        dataset_id=ds_id,
                                        scaling=sc,
                                        log_transform=log_tr,
                                        feature_subset=fs,
                                        duplicate_handling=dup,
                                        missingness=miss,
                                        seed=sd,
                                    )
                                )
    return prune_multiverse(universes)


def get_universe(index: int) -> Universe:

    universes = generate_multiverse()
    if index < 0 or index >= len(universes):
        raise typer.BadParameter(f"universe_index must be in [0, {len(universes)-1}]")
    universe = universes[index]
    logger.info("[EXP] Using universe: %s", universe)
    return universe
