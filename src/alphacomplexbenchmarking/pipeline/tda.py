# src/alphacomplexbenchmarking/pipeline/tda.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from alphacomplexbenchmarking.pipeline.specs import TdaConfig
from alphacomplexbenchmarking.pipeline.persistence import compute_alpha_complex_persistence
from alphacomplexbenchmarking.pipeline.landscapes import compute_landscapes

logger = logging.getLogger(__name__)


@dataclass
class TdaResult:
    persistence_per_dim: Dict[int, np.ndarray]
    landscapes_per_dim: Dict[int, Optional[np.ndarray]]


def run_tda_on_points(points: np.ndarray, config: TdaConfig) -> TdaResult:
    """
    Run alpha-complex persistence + landscapes on given points (N x d).
    """
    logger.info(
        f"[TDA] Running alpha complex TDA on points with shape={points.shape}, "
        f"dims={config.homology_dimensions}"
    )
    persistence = compute_alpha_complex_persistence(
        data=points,
        homology_dimensions=list(config.homology_dimensions),
    )
    landscapes = compute_landscapes(
        persistence_per_dimension=persistence,
        num_landscapes=config.num_landscapes,
        resolution=config.resolution,
        homology_dimensions=list(config.homology_dimensions),
    )
    return TdaResult(
        persistence_per_dim=persistence,
        landscapes_per_dim=landscapes,
    )