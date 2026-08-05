from __future__ import annotations

import logging

import gudhi as gd
import numpy as np
from gudhi.representations import Landscape

from preprolamu.pipeline.embeddings import Embedding
from preprolamu.pipeline.persistence import Persistence
from preprolamu.pipeline.universes import Universe
from preprolamu.pipeline.metrics import compute_metrics_from_tda
from preprolamu.helpers import mask_infinities

logger = logging.getLogger(__name__)



def compute_alpha_complex_persistence(
    data: np.ndarray, homology_dimensions: list[int] = [0, 1, 2]
):

    logger.info(f"Computing alpha complex persistence for data of shape {data.shape}")
    ac = gd.AlphaComplex(points=data, precision="exact")
    st = ac.create_simplex_tree()
    st.compute_persistence(homology_coeff_field=2)

    logger.info(f"Computed persistence with {len(st.persistence_pairs())} intervals")

    per_dim: dict[int, np.ndarray] = {}
    for dim in homology_dimensions:
        per_dim[dim] = mask_infinities(st.persistence_intervals_in_dimension(dim))
        logger.info(f"Dim {dim}: {per_dim[dim].shape[0]} intervals after masking")
    return per_dim


def compute_landscapes(
    persistence_per_dimension: dict[int, np.ndarray],
    num_landscapes: int = 5,
    resolution: int = 1000,
    homology_dimensions: list[int] = [0, 1, 2],
) -> dict[int, np.ndarray | None]:

    LS = Landscape(
        resolution=resolution, keep_endpoints=False, num_landscapes=num_landscapes
    )

    landscapes_per_dimension: dict[int, np.ndarray | None] = {}

    for dim in homology_dimensions:
        persistence_pairs = persistence_per_dimension.get(dim, [])
        if len(persistence_pairs) == 0:
            logger.warning(
                f"No persistence pairs for dim {dim}; landscapes will be None"
            )
            landscapes_per_dimension[dim] = None
            continue

        landscapes_per_dimension[dim] = LS.fit_transform([persistence_pairs])

    return landscapes_per_dimension


def _load_if_exists(path, loader, overwrite):
    return None if overwrite or not path.exists() else loader()


def prepare_point_cloud(universe: Universe, split: str = "test"):
    logger.info("[Embedding] Loading embedding (%s, %s).", universe.id, split)

    latent = universe.io.load_embedding(split=split, force_recompute=False)
    point_cloud = Embedding(latent_space=latent, universe=universe)
    point_cloud.project_PCA(n_components=universe.pca_dim)
    scale = point_cloud.normalize(method="diameter", iterations=1000)
    logger.info("[Embedding] Normalization scale: %s", scale)
    point_cloud.save(split=split)
    point_cloud.sample(target_size=universe.tda_config.subsample_size)
    
    return point_cloud.latent_space


def run_tda_for_universe(u: Universe, split="test", overwrite=False):
    
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
        tda.points = prepare_point_cloud(u, split=split)
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


# # NOTE: This split parameter depends on the parameter setting during the AE step. 
# def run_tda_for_universe(
#     universe: Universe, split: str = "test", overwrite: bool = False
# ):

#     tc = universe.tda_config
#     pca_dim = universe.pca_dim
#     persistence_path = universe.paths.persistence(split=split)

#     # Read latent embedding, err if not found
#     logger.info("[Embedding] Loading embedding (%s, %s).", universe.id, split)
#     try:
#         latent = universe.io.load_embedding(split=split, force_recompute=False)
#     except FileNotFoundError:
#         logger.error(f"[TDA] No embedding found (u={universe.id}, split={split}).")
#         raise

#     if persistence_path.exists() and not overwrite:
#         per_dim = universe.io.load_persistence(split=split)
#         logger.info("[TDA] Loaded existing persistence from %s", persistence_path)
#         return
    
#     logger.info("[Embedding] Preparing embedding")
#     point_cloud = Embedding(latent_space=latent, universe=universe)
#     point_cloud.project_PCA(n_components=pca_dim)
#     scale = point_cloud.normalize(method="diameter", iterations=1000)
#     logger.info("[Embedding] Normalization scale: %s", scale)
#     point_cloud.save(split=split)
#     point_cloud.sample(target_size=tc.subsample_size)
    
#     logger.info("[TDA] Computing persistence (path: %s).", persistence_path)
#     per_dim = compute_alpha_complex_persistence(
#         data=point_cloud.latent_space,
#         homology_dimensions=tc.homology_dimensions,
#     )
#     universe.io.save_persistence(split=split, per_dim=per_dim)

#     # We assume that if persistence does not exist, landscapes and metrics also do not exist.
#     logger.info("[TDA] Computing landscapes (path: %s).", universe.paths.landscapes(split=split))
#     landscapes = compute_landscapes(
#         persistence_per_dimension=per_dim,
#         num_landscapes=tc.num_landscapes,
#         resolution=tc.resolution,
#         homology_dimensions=tc.homology_dimensions,
#     )
#     universe.io.save_landscapes(split=split, landscapes=landscapes)

#     logger.info("[TDA] Computing aggregate metrics (path: %s).", universe.paths.metrics(split=split))
#     metrics = compute_metrics_from_tda(
#         persistence_per_dimension=per_dim,
#         landscapes_per_dimension=landscapes,
#     )
#     universe.io.save_metrics(split=split, metrics=metrics)


#     return per_dim, landscapes, metrics
