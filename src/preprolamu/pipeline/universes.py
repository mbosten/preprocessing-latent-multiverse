# src/preprolamu/pipeline/universes.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import typer

logger = logging.getLogger(__name__)


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
    subsample_size: int = 10_000  # points used for TDA


@dataclass(frozen=True)
class Universe:
    """
    One 'universe' in the multiverse: preprocessing + AE + TDA params.
    Holds all parameters needed to run the full pipeline for one configuration, including fixed-value parameters.
    """

    dataset_id: str

    # Preprocessing choices
    scaling: Scaling
    log_transform: LogTransform
    feature_subset: FeatureSubset
    duplicate_handling: DuplicateHandling
    missingness: Missingness
    seed: int

    # Will be overridden by PCA dims below.
    pca_dim: int = 3

    # TDA config
    tda_config: TdaConfig = TdaConfig()

    def to_id_string(self) -> str:
        """
        Short string encoding for filenames/logs.
        """
        return (
            f"ds-{self.dataset_id}"
            f"_sc-{self.scaling.value}"
            f"_log-{self.log_transform.value}"
            f"_fs-{self.feature_subset.value}"
            f"_dup-{self.duplicate_handling.value}"
            f"_miss-{self.missingness.value}"
            f"_sd-{self.seed}"
        )


DATASET_IDS: List[str] = [
    # "Merged35",
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
    "NF-CICIDS2018-v3",
]


def generate_multiverse() -> List[Universe]:
    """
    Generate the multiverses
    """
    scalings = [Scaling.ZSCORE, Scaling.MINMAX, Scaling.QUANTILE]
    log_transforms = [LogTransform.NONE, LogTransform.LOG1P]
    feature_subsets = [FeatureSubset.ALL, FeatureSubset.WITHOUT_CONFOUNDERS]
    duplicate_opts = [DuplicateHandling.KEEP, DuplicateHandling.DROP]
    missingness_opts = [Missingness.DROP_ROWS, Missingness.IMPUTE_MEDIAN]
    seeds = [42, 420, 4200]

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
    return universes


def get_universe(index: int) -> Universe:
    """
    Get a single universe from the multiverse.
    """
    universes = generate_multiverse()
    if index < 0 or index >= len(universes):
        raise typer.BadParameter(f"universe_index must be in [0, {len(universes)-1}]")
    universe = universes[index]
    logger.info("[EXP] Using universe: %s", universe)
    return universe
