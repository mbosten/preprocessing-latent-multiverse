# src/alphacomplexbenchmarking/pipeline/universes.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple, Optional
import logging
import typer

logger = logging.getLogger(__name__)

class Scaling(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"

class FeatureSubset(str, Enum):
    ALL = "all"
    WITHOUT_CONFOUNDERS = "without_confounders"

class CatEncoding(str, Enum):
    ONEHOT = "one_hot"
    LABEL = "label"

class Duplicates(str, Enum):
    KEEP = "keep"
    DROP = "drop"


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
    # Preprocessing choices
    scaling: Scaling
    feature_subset: FeatureSubset
    cat_encoding: Optional[CatEncoding]
    duplicates: Duplicates
    seed: int

    # AE / embedding
    ae_latent_dim: int = 25
    ae_hidden_dims: Tuple[int, ...] = (44, 39)
    ae_epochs: int = 10
    ae_batch_size: int = 256                  # following Sičić et al. (2023)
    ae_dropout: float = 0.0377                # following Sičić et al. (2023)
    ae_regularization: float = 0.0019         # following Sičić et al. (2023)

    # Will be overridden by PCA dims below.
    pca_dim: int = 3

    # TDA config
    tda_config: TdaConfig = TdaConfig()


    dataset_id: str = "Merged35" # "NF-ToN-IoT-v3"

    def to_id_string(self) -> str:
        """
        Short string encoding for filenames/logs.
        """
        ce_val = self.cat_encoding.value if self.cat_encoding is not None else "none"
        return (
            f"ds-{self.dataset_id}"
            f"_sc-{self.scaling.value}"
            f"_fs-{self.feature_subset.value}"
            f"_ce-{ce_val}"
            f"_dup-{self.duplicates.value}"
            f"_sd-{self.seed}"
            f"_pca-{self.pca_dim}"
        )


def generate_multiverse() -> List[Universe]:
    """
    Generate the multiverses
    """
    scalings = [Scaling.ZSCORE, Scaling.MINMAX, Scaling.ROBUST, Scaling.QUANTILE]
    feature_subsets = [FeatureSubset.ALL, FeatureSubset.WITHOUT_CONFOUNDERS]
    cat_encodings = [CatEncoding.ONEHOT, CatEncoding.LABEL]
    duplicates = [Duplicates.KEEP, Duplicates.DROP]
    seeds = [42, 420, 4200]
    pca_dims = (2, 3, 4)

    universes: List[Universe] = []
    for sc in scalings:
        for fs in feature_subsets:
            # Without confounders there are no categorical features
            if fs == FeatureSubset.WITHOUT_CONFOUNDERS:
                relevant_cat_encodings = [None]
            else:
                relevant_cat_encodings = list(cat_encodings)

            for ce in relevant_cat_encodings:
                for dup in duplicates:
                    for sd in seeds:
                        for pca_dim in pca_dims:
                            universes.append(
                                Universe(
                                    scaling=sc,
                                    feature_subset=fs,
                                    cat_encoding=ce,
                                    duplicates=dup,
                                    seed=sd,
                                    pca_dim=pca_dim,
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