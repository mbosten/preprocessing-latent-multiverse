# src/alphacomplexbenchmarking/pipeline/universes.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple

class Scaling(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"

class FeatureSubset(str, Enum):
    ALL = "all"
    WITHOUT_CONFOUNDERS = "without_confounders"

class CatEncoding(str, Enum):
    ONEHOT = "one_hot"
    LABEL = "label"

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
    cat_encoding: CatEncoding
    seed: int

    # AE / embedding
    ae_latent_dim: int = 12
    ae_hidden_dims: Tuple[int, ...] = (26, )
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
        return (
            f"ds-{self.dataset_id}"
            f"_sc-{self.scaling.value}"
            f"_fs-{self.feature_subset.value}"
            f"_ce-{self.cat_encoding.value}"
            f"_sd-{self.seed}"
            f"_pca-{self.pca_dim}"
        )


def generate_multiverse() -> List[Universe]:
    """
    Generate the multiverses
    """
    scalings = [Scaling.ZSCORE, Scaling.MINMAX]
    feature_subsets = [FeatureSubset.ALL, FeatureSubset.WITHOUT_CONFOUNDERS]
    cat_encodings = [CatEncoding.ONEHOT, CatEncoding.LABEL]
    seeds = [42, 420, 4200]
    pca_dims = (2, 3, 4)

    universes: List[Universe] = []
    for sc in scalings:
        for fs in feature_subsets:
            for ce in cat_encodings:
                for sd in seeds:
                    for pca_dim in pca_dims:
                        universes.append(
                            Universe(
                                scaling=sc,
                                feature_subset=fs,
                                cat_encoding=ce,
                                seed=sd,
                                pca_dim=pca_dim,
                            )
                        )
    return universes
