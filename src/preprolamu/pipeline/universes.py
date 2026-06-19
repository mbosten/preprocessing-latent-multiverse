from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
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
    homology_dimensions: tuple[int, ...] = (0, 1, 2)
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

    # Path/IO functions reference to their respective files
    @property
    def paths(self) -> UniversePaths:
        return UniversePaths(self)

    @property
    def io(self) -> UniverseIO:
        return UniverseIO(self)

    def to_param_dict(self) -> dict[str, Any]:
        """
        Flatten Universe into a dict of multiverse parameters used for grouping.
        """
        out: dict[str, Any] = {
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


DATASET_IDS: list[str] = [
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
    "NF-CICIDS2018-v3",
]


@lru_cache(maxsize=None)
def load_clean_dataset(dataset_id: str) -> pd.DataFrame:
    cfg = load_dataset_config(dataset_id)
    path = cfg["clean_path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Clean datast not found: {path}. "
            f"Run `prepare-dataset {dataset_id}` first."
        )

    return pd.read_parquet(path)


def load_dataset_profile(dataset_id: str) -> dict[str, dict[str, Any]]:
    path = Path("data") / "metadata" / f"{dataset_id}_profile.yml"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset profile not found: {path}. "
            f"Run `prepare-dataset {dataset_id}` first."
        )

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_dataset_profiles(
    dataset_ids: Iterable[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        dataset_id: load_dataset_profile(dataset_id)
        for dataset_id in sorted(set(dataset_ids))
    }


def prune_multiverse(
    universes: list[Universe],
    profiles_by_dataset: dict[str, dict[str, dict[str, Any]]],
) -> list[Universe]:

    kept: list[Universe] = []

    for u in universes:
        profile = profiles_by_dataset[u.dataset_id][u.feature_subset.value]

        if (
            not profile["has_duplicates"]
            and u.duplicate_handling == DuplicateHandling.DROP
        ):
            continue

        if (
            not profile["has_missing_numeric"]
            and u.missingness == Missingness.DROP_ROWS
        ):
            continue

        kept.append(u)

    logger.info("[MV] Pruned Multiverse: %d -> %d", len(universes), len(kept))
    return kept


def generate_full_multiverse() -> list[Universe]:

    scalings = [Scaling.ZSCORE, Scaling.MINMAX, Scaling.QUANTILE]
    log_transforms = [LogTransform.NONE, LogTransform.LOG1P]
    feature_subsets = [FeatureSubset.ALL, FeatureSubset.WITHOUT_CONFOUNDERS]
    duplicate_opts = [DuplicateHandling.KEEP, DuplicateHandling.DROP]
    missingness_opts = [Missingness.DROP_ROWS, Missingness.IMPUTE_MEDIAN]
    seeds = [42, 420, 4200, 42000]

    universes: list[Universe] = []

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
    return universes


def generate_multiverse() -> list[Universe]:
    universes = generate_full_multiverse()

    profiles = build_dataset_profiles(universe.dataset_id for universe in universes)

    return prune_multiverse(universes, profiles)


def get_universe(index: int) -> Universe:
    universes = generate_multiverse()

    if index < 0 or index >= len(universes):
        raise typer.BadParameter(f"universe_index must be in [0, {len(universes)-1}]")

    universe = universes[index]
    logger.info("[EXP] Using universe: %s", universe)
    return universe
