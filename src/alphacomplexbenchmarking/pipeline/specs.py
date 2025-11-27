# src/alphacomplexbenchmarking/pipeline/specs.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Scaling(str, Enum):
    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"

class FeatureSubset(str, Enum):
    ALL = "all"
    WITHOUT_CONFOUNDERS = "without_confounders"

class CatEncoding(str, Enum):
    ONE_HOT = "one_hot"
    LABEL = "label"

@dataclass(frozen=True)
class RunSpec:
    # preprocessing choices
    scaling: Scaling
    feature_subset: FeatureSubset
    cat_encoding: CatEncoding
    seed: int

    # model / TDA hyperparameters
    ae_latent_dim: int
    ae_hidden_dims: tuple[int, ...]
    ae_epochs: int
    ae_batch_size: int
    pca_components: int
    homology_dimensions: tuple[int, ...]
    num_landscapes: int
    resolution: int

    # identifier for the base dataset
    dataset_id: str = "NF-ToN-IoT-v3"