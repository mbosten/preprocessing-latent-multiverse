from __future__ import annotations

import logging

import gudhi as gd
import numpy as np
from gudhi.representations import Landscape

from preprolamu.pipeline.embeddings import from_latent_to_point_cloud
from preprolamu.pipeline.metrics import compute_metrics_from_tda
from preprolamu.pipeline.persistence import mask_infinities
from preprolamu.pipeline.universes import Universe

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


def run_tda_for_universe(
    universe: Universe, split: str = "test", overwrite: bool = False
):

    tda_cfg = universe.tda_config
    pca_dim = universe.pca_dim
    m = tda_cfg.subsample_size
    hom_dims = tda_cfg.homology_dimensions
    num_landscapes = tda_cfg.num_landscapes
    resolution = tda_cfg.resolution

    # 1. Read latent embedding, err if not found
    try:
        latent = universe.io.load_embedding(split=split, force_recompute=False)
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

    persistence_path = universe.paths.persistence(split=split)
    landscapes_path = universe.paths.landscapes(split=split)
    metrics_path = universe.paths.metrics(split=split)

    # Check if persistence already exists
    if persistence_path.exists() and not overwrite:
        per_dim = universe.io.load_persistence(split=split)
        logger.info("[TDA] Loaded persistence from %s", persistence_path)
        return
    else:
        logger.info("[TDA] Computing persistence for %s", universe.id)
        points_for_tda, diameter = from_latent_to_point_cloud(
            X=latent,
            pca_dim=pca_dim,
            target_size=m,
            seed=universe.seed,
            normalize=True,
            save_projected_to=(universe, split),
            save_projected_raw_to=(universe, split),
        )
        logger.info("[TDA] normalization diameter: %s", diameter)

        per_dim = compute_alpha_complex_persistence(
            data=points_for_tda,
            homology_dimensions=hom_dims,
        )

        universe.io.save_persistence(split=split, per_dim=per_dim)
        logger.info("[TDA] Saved persistence to %s", persistence_path)

    # Check if landscapes already exist
    if landscapes_path.exists() and not overwrite:
        landscapes = universe.io.load_landscapes(split=split)
        logger.info("[TDA] Landscapes already exist at %s.", landscapes_path)
    else:
        logger.info("[TDA] Computing landscapes for %s", universe.id)
        landscapes = compute_landscapes(
            persistence_per_dimension=per_dim,
            num_landscapes=num_landscapes,
            resolution=resolution,
            homology_dimensions=hom_dims,
        )

        universe.io.save_landscapes(split=split, landscapes=landscapes)
        logger.info("[TDA] Saved landscapes to %s", landscapes_path)

    # Check if metrics already exist
    if metrics_path.exists() and not overwrite:
        metrics = universe.io.load_metrics(split=split)
        logger.info("[TDA] Metrics already exist at %s.", metrics_path)
    else:
        metrics = compute_metrics_from_tda(
            persistence_per_dimension=per_dim,
            landscapes_per_dimension=landscapes,
        )

        universe.io.save_metrics(split, metrics)
        logger.info("[TDA] Saved metrics to %s", metrics_path)

    return per_dim, landscapes, metrics
