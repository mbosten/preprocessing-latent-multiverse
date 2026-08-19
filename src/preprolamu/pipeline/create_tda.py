from __future__ import annotations

import logging

from preprolamu.pipeline.embeddings import Embedding
from preprolamu.pipeline.persistence import Persistence
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def _load_if_exists(path, loader, overwrite):
    return None if overwrite or not path.exists() else loader()


def prepare_point_cloud(universe: Universe, latent) -> Embedding:
    point_cloud = Embedding(latent_space=latent, universe=universe)
    point_cloud.project_PCA(n_components=universe.pca_dim)
    point_cloud.normalize(method="diameter", iterations=1000)
    return point_cloud


def run_tda_for_universe(u: Universe, split="test", overwrite=False):
    # NOTE: This split parameter depends on the parameter setting during the AE step. 
    intervals = _load_if_exists(
        u.paths.persistence(split=split),
        lambda: u.io.load_persistence(split=split),
        overwrite,
    )
    landscapes = _load_if_exists(
        u.paths.landscapes(split=split),
        lambda: u.io.load_landscapes(split=split),
        overwrite,
    )
    metrics = _load_if_exists(
        u.paths.metrics(split=split),
        lambda: u.io.load_metrics(split=split),
        overwrite,
    )

    tda = Persistence(universe=u, intervals=intervals, landscapes=landscapes)

    if intervals is None:
        logger.info(f"[TDA] Computing persistence for universe {u.id} (split={split})")
        latent = u.io.load_embedding(split=split)
        point_cloud = prepare_point_cloud(u, latent)
        point_cloud.save(split=split)
        point_cloud.sample(target_size=u.tda_config.subsample_size)
        tda.points = point_cloud.latent_space
        intervals = tda.compute_intervals()
        u.io.save_persistence(split=split, per_dim=intervals)

    if landscapes is None:
        logger.info(f"[TDA] Computing landscapes for universe {u.id} (split={split})")
        landscapes = tda.compute_landscapes()
        u.io.save_landscapes(split=split, landscapes=landscapes)

    if metrics is None:
        logger.info(f"[TDA] Computing metrics for universe {u.id} (split={split})")
        metrics = tda.metrics()
        u.io.save_metrics(split=split, metrics=metrics)

    return intervals, landscapes, metrics


def compute_tda_for_test(universe: Universe, latent):
    point_cloud = prepare_point_cloud(universe, latent)
    point_cloud.sample(target_size=universe.tda_config.subsample_size)
    tda = Persistence(universe=universe, points=point_cloud.latent_space)
    tda.compute_intervals()
    tda.compute_landscapes()
    return tda.metrics()