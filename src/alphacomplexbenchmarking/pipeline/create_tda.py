# src/alphacomplexbenchmarking/pipeline/create_tda.py
from __future__ import annotations

import logging

from alphacomplexbenchmarking.pipeline.universes import Universe
from alphacomplexbenchmarking.pipeline.create_tda import get_or_compute_latent
from alphacomplexbenchmarking.pipeline.embeddings import from_latent_to_point_cloud
from alphacomplexbenchmarking.pipeline.persistence import compute_alpha_complex_persistence
from alphacomplexbenchmarking.pipeline.landscapes import compute_landscapes
from alphacomplexbenchmarking.pipeline.metrics import compute_metrics_from_tda

logger = logging.getLogger(__name__)


def run_tda_for_universe(universe: Universe):
    """
    Complete TDA step for a single universe, starting from latent embeddings.

    Steps:
      - load or compute latent
      - normalize latent
      - PCA to pca_dim (fit on full normalized latent)
      - subsample to subsample_size
      - compute alpha-complex persistence
      - compute landscapes
      - compute scalar metrics
    """

    tda_cfg = universe.tda_config
    pca_dim = universe.pca_dim
    m = tda_cfg.subsample_size
    hom_dims = tda_cfg.homology_dimensions
    num_landscapes = tda_cfg.num_landscapes
    resolution = tda_cfg.resolution

    # 1. Latent
    latent = get_or_compute_latent(universe, retrain_if_missing=True, force_recompute=False)
    logger.info("[TDA] Loaded latent for %s with shape %s", universe, latent.shape)


    points_for_tda = from_latent_to_point_cloud(
        X=latent,
        pca_dim=pca_dim,
        target_size=m,
        seed=universe.seed,
        normalize=True,
    )

    # 2. Persistence
    per_dim = compute_alpha_complex_persistence(
        data=points_for_tda,
        homology_dimensions=hom_dims,
    )

    # 3. Landscapes
    landscapes = compute_landscapes(
        persistence_per_dimension=per_dim,
        num_landscapes=num_landscapes,
        resolution=resolution,
        homology_dimensions=hom_dims,
    )

    # 4. Metrics
    metrics = compute_metrics_from_tda(
        persistence_per_dimension=per_dim,
        landscapes_per_dimension=landscapes
    )

    return per_dim, landscapes, metrics
