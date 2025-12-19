# src/preprolamu/pipeline/create_tda.py
from __future__ import annotations

import logging

from preprolamu.io.storage import (
    load_embedding,
    load_landscapes,
    load_metrics,
    load_persistence,
    save_landscapes,
    save_metrics,
    save_persistence,
)
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
            f"[TDA] No embedding found for universe {universe.id} "
            f"(split='{split}'). Run the embedding step first."
        )
        logger.error(msg)
        raise RuntimeError(msg) from e

    logger.info(
        "[TDA] Loaded latent for %s with shape %s",
        universe.id,
        latent.shape,
    )

    persistence_path = universe.persistence_path(split=split)
    landscapes_path = universe.landscapes_path(split=split)
    metrics_path = universe.metrics_path(split=split)

    # Check if persistence already exists
    if persistence_path.exists():
        per_dim = load_persistence(universe, split)
        logger.info("[TDA] Loaded persistence from %s", persistence_path)
        return
    else:
        logger.info("[TDA] Computing persistence for %s", universe.id)
        points_for_tda = from_latent_to_point_cloud(
            X=latent,
            pca_dim=pca_dim,
            target_size=m,
            seed=universe.seed,
            normalize=True,
        )

        per_dim = compute_alpha_complex_persistence(
            data=points_for_tda,
            homology_dimensions=hom_dims,
        )

        save_persistence(universe, split, per_dim)
        logger.info("[TDA] Saved persistence to %s", persistence_path)

    # Check if landscapes already exist
    if landscapes_path.exists():
        landscapes = load_landscapes(universe, split)
        logger.info("[TDA] Landscapes already exist at %s.", landscapes_path)
    else:
        logger.info("[TDA] Computing landscapes for %s", universe.id)
        landscapes = compute_landscapes(
            persistence_per_dimension=per_dim,
            num_landscapes=num_landscapes,
            resolution=resolution,
            homology_dimensions=hom_dims,
        )

        save_landscapes(universe, split, landscapes)
        logger.info("[TDA] Saved landscapes to %s", landscapes_path)

    # Check if metrics already exist
    if metrics_path.exists():
        metrics = load_metrics(universe, split)
        logger.info("[TDA] Metrics already exist at %s.", metrics_path)
    else:
        metrics = compute_metrics_from_tda(
            persistence_per_dimension=per_dim,
            landscapes_per_dimension=landscapes,
        )

        save_metrics(universe, split, metrics)
        logger.info("[TDA] Saved metrics to %s", metrics_path)

    return per_dim, landscapes, metrics
