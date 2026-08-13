from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Literal

import json
import typer

from preprolamu.io.io import UniverseIO
from preprolamu.io.paths import UniversePaths

logger = logging.getLogger(__name__)


Scaling = Literal["zscore", "minmax", "quantile"]
LogTransform = Literal["none", "log1p"]
FeatureSubset = Literal["all", "without_confounders"]
DuplicateHandling = Literal["drop", "keep"]
Missingness = Literal["drop_rows", "impute_median"]

DATASET_IDS = (
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
    "NF-CICIDS2018-v3",
)

MULTIVERSE_GRID = {
    "scaling": ("zscore", "minmax", "quantile"),
    "log_transform": ("none", "log1p"),
    "feature_subset": ("all", "without_confounders"),
    "duplicate_handling": ("keep", "drop"),
    "missingness": ("drop_rows", "impute_median"),
    "seed": (42, 420, 4200, 42000),
}

PROFILE_PATH = Path("data/interim/metadata/profiles.json")


@dataclass(frozen=True)
class TdaConfig:
    homology_dimensions: tuple[int, ...] = (0, 1, 2)
    num_landscapes: int = 5
    resolution: int = 1000
    subsample_size: int = 500_000  # points used for TDA


@dataclass(frozen=True)
class Universe:
    dataset_id: str
    scaling: Scaling
    log_transform: LogTransform
    feature_subset: FeatureSubset
    duplicate_handling: DuplicateHandling
    missingness: Missingness
    seed: int

    pca_dim: int = 3
    tda_config: TdaConfig = field(default_factory=TdaConfig)

    universe_index: int | None = field(
        default=None,
        compare=False,
        hash=False,
    )

    base_data_dir: Path = field(
        default=Path("data"),
        compare=False,
        hash=False,
    )

    id: str = field(init=False)

    def __post_init__(self):
        prefix = (
            f"u-{self.universe_index:04d}_"
            if self.universe_index is not None
            else ""
        )

        object.__setattr__(
            self,
            "id",
            (
                f"{prefix}"
                f"ds-{self.dataset_id}"
                f"_sc-{self.scaling}"
                f"_log-{self.log_transform}"
                f"_fs-{self.feature_subset}"
                f"_dup-{self.duplicate_handling}"
                f"_miss-{self.missingness}"
                f"_sd-{self.seed}"
            ),
        )

    @property
    def paths(self) -> UniversePaths:
        return UniversePaths(self)

    @property
    def io(self) -> UniverseIO:
        return UniverseIO(self)

    def to_param_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scaling": self.scaling,
            "log_transform": self.log_transform,
            "feature_subset": self.feature_subset,
            "duplicate_handling": self.duplicate_handling,
            "missingness": self.missingness,
            "seed": self.seed,
        }


def load_profiles():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Profiles file not found: {PROFILE_PATH}. "
            f"Run `prepare-dataset` first."
        )
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def generate_full_multiverse() -> list[Universe]:
    keys = tuple(MULTIVERSE_GRID)

    return [
        Universe(
            dataset_id=dataset_id,
            **dict(zip(keys, values, strict=True)),
        )
        for dataset_id in DATASET_IDS
        for values in product(*(MULTIVERSE_GRID[key] for key in keys))
    ]


def prune_multiverse(
    universes: list[Universe],
    profiles: dict[str, dict[str, dict[str, bool]]],
) -> list[Universe]:

    def redundant(universe: Universe) -> bool:
        profile = profiles[universe.dataset_id][universe.feature_subset]

        return (
            not profile["has_duplicates"]
            and universe.duplicate_handling == "drop"
        ) or (
            not profile["has_missing_numeric"]
            and universe.missingness == "drop_rows"
        )

    kept = [u for u in universes if not redundant(u)]

    logger.info(
        "[MV] Pruned multiverse: %d -> %d",
        len(universes),
        len(kept),
    )

    return kept


def generate_multiverse() -> list[Universe]:
    universes = prune_multiverse(
        generate_full_multiverse(),
        load_profiles()
    )

    return [replace(u, universe_index=i) for i, u in enumerate(universes)]


def get_universe(index: int) -> Universe:
    universes = generate_multiverse()

    if not 0 <= index < len(universes):
        raise typer.BadParameter(f"Universe index must be in [0, {len(universes) - 1}], got {index}")

    universe = universes[index]
    return universe
