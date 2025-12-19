# src/preprolamu/pipeline/create_tda.py
from __future__ import annotations

import logging

from preprolamu.io.storage import load_embedding, save_metrics_from_tda_output
from preprolamu.pipeline.embeddings import from_latent_to_point_cloud
from preprolamu.pipeline.landscapes import compute_landscapes
from preprolamu.pipeline.metrics import compute_metrics_from_tda
from preprolamu.pipeline.persistence import compute_alpha_complex_persistence
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def run_tda_for_universe(universe: Universe, split: str = "test"):
    """
    Complete TDA step for a single universe, starting from latent embeddings.
    """

    tda_cfg = universe.tda_config
    pca_dim = universe.pca_dim
    m = tda_cfg.subsample_size
    hom_dims = tda_cfg.homology_dimensions
    num_landscapes = tda_cfg.num_landscapes
    resolution = tda_cfg.resolution

    # 1. Read latent embedding, err if not found
    try:
        latent = load_embedding(universe, split=split, force_recompute=False)
    except FileNotFoundError as e:
        msg = (
            f"[TDA] No embedding found for universe {universe.to_id_string()} "
            f"(split='{split}'). Run the embedding step first."
        )
        logger.error(msg)
        raise RuntimeError(msg) from e

    logger.info(
        "[TDA] Loaded latent for %s with shape %s",
        universe.to_id_string(),
        latent.shape,
    )

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
        persistence_per_dimension=per_dim, landscapes_per_dimension=landscapes
    )

    # 5. Save
    save_metrics_from_tda_output(
        universe=universe,
        per_dim=per_dim,
        landscapes=landscapes,
        metrics=metrics,
    )

    return per_dim, landscapes, metrics
