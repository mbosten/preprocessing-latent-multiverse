# src/preprolamu/pipeline/universes.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
import yaml

from preprolamu.config import load_dataset_config

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

    # Parsed ID string
    id: str = field(init=False)

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

    # Path functions
    def embedding_path(self, split: str = "test") -> Path:
        return (
            BASE_DATA_DIR / "interim" / "embeddings" / f"{self.id}_latent_{split}.npy"
        )

    def projected_path(self, split: str = "test", normalized: bool = False) -> Path:
        tag = "" if normalized else "_raw"
        return (
            BASE_DATA_DIR
            / "interim"
            / "projections"
            / f"{self.id}_projected_{split}{tag}.npy"
        )

    def persistence_path(self, split: str = "test") -> Path:
        return (
            BASE_DATA_DIR
            / "interim"
            / "persistence"
            / f"{self.id}_persistence_{split}.npz"
        )

    def landscapes_path(self, split: str = "test") -> Path:
        return (
            BASE_DATA_DIR
            / "interim"
            / "landscapes"
            / f"{self.id}_landscapes_{split}.npz"
        )

    def metrics_path(self, split: str = "test") -> Path:
        return (
            BASE_DATA_DIR / "processed" / "metrics" / f"{self.id}_metrics_{split}.json"
        )

    def ae_model_path(self) -> Path:
        return BASE_DATA_DIR / "interim" / "autoencoder" / f"{self.id}_ae.pt"

    def eval_metrics_path(self, split: str = "test") -> Path:
        return (
            BASE_DATA_DIR
            / "processed"
            / "eval_metrics"
            / f"{self.id}_eval_{split}.json"
        )

    def preprocessed_train_path(self) -> Path:
        return (
            BASE_DATA_DIR
            / "processed"
            / "train"
            / f"{self.id}_preprocessed_train.parquet"
        )

    def preprocessed_validation_path(self) -> Path:
        return (
            BASE_DATA_DIR
            / "processed"
            / "validation"
            / f"{self.id}_preprocessed_validation.parquet"
        )

    def preprocessed_test_path(self) -> Path:
        return (
            BASE_DATA_DIR
            / "processed"
            / "test"
            / f"{self.id}_preprocessed_test.parquet"
        )

    def preprocessing_status_path(self) -> Path:
        return BASE_DATA_DIR / "interim" / "preprocessing_status" / f"{self.id}.status"

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
    # "Merged35",
    "NF-ToN-IoT-v3",
    "NF-UNSW-NB15-v3",
    "NF-CICIDS2018-v3",
]


def dataset_invariants_for(universe: Universe) -> dict[str, Any]:
    cfg = load_dataset_config(universe.dataset_id)
    inv = cfg.raw_cfg.get("dataset_invariants")  # see note below
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
    """
    Generate the multiverses
    """
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
    """
    Get a single universe from the multiverse.
    """
    universes = generate_multiverse()
    if index < 0 or index >= len(universes):
        raise typer.BadParameter(f"universe_index must be in [0, {len(universes)-1}]")
    universe = universes[index]
    logger.info("[EXP] Using universe: %s", universe)
    return universe
