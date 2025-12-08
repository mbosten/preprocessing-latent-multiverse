# src/preprolamu/pipeline/parallel.py
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List

from preprolamu.io.storage import (
    get_metrics_path,
    get_tda_result_path,
    load_numpy_array,
    save_json,
    save_tda_npz,
)
from preprolamu.pipeline.autoencoder import train_autoencoder_for_universe
from preprolamu.pipeline.embeddings import compute_embeddings_and_subsample_for_tda
from preprolamu.pipeline.metrics import compute_metrics_from_tda
from preprolamu.pipeline.preprocessing import preprocess_variant
from preprolamu.pipeline.tda import run_tda_on_points
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def run_full_pipeline_for_universe(universe: Universe) -> str:
    """
    Full multiverse pipeline for a single Universe:
      1. Preprocess
      2. Train AE
      3. Compute embeddings + PCA + subsample
      4. Run TDA (alpha + landscapes)
      5. Compute summary metrics
      6. Save results

    Returns the universe id string for logging.
    """
    universe_id = universe.to_id_string()
    logger.info(f"[PIPE] Starting full pipeline for universe={universe_id}")

    # 1. Preprocess
    preprocess_variant(universe)

    # 2. Train AE
    train_autoencoder_for_universe(universe)

    # 3. Embeddings + PCA + subsample
    emb_path = compute_embeddings_and_subsample_for_tda(universe)

    # 4. TDA
    points = load_numpy_array(emb_path)
    tda_result = run_tda_on_points(points, universe.tda_config)

    # Save TDA arrays
    tda_path = get_tda_result_path(universe)

    # Flatten dicts for npz
    tda_arrays = {}
    for dim, arr in tda_result.persistence_per_dim.items():
        tda_arrays[f"pers_dim_{dim}"] = arr
    for dim, arr in tda_result.landscapes_per_dim.items():
        if arr is not None:
            tda_arrays[f"land_dim_{dim}"] = arr
    save_tda_npz(tda_path, **tda_arrays)
    logger.info(f"[PIPE] Saved TDA results to {tda_path}")

    # 5. Metrics
    metrics = compute_metrics_from_tda(tda_result)

    metrics_payload = {
        "universe_id": universe_id,
        "total_persistence_per_dim": metrics.total_persistence_per_dim,
        "landscape_l2_per_dim": metrics.landscape_l2_per_dim,
    }
    metrics_path = get_metrics_path(universe)
    save_json(metrics_path, metrics_payload)
    logger.info(f"[PIPE] Saved metrics to {metrics_path}")

    logger.info(f"[PIPE] Finished full pipeline for universe={universe_id}")
    return universe_id


def _worker_run_full_pipeline(universe: Universe) -> str:
    """
    Worker function for ProcessPoolExecutor.
    """
    return run_full_pipeline_for_universe(universe)


def run_many_universes(
    universes: Iterable[Universe], max_workers: int | None = None
) -> List[str]:
    """
    Run many Universes in parallel using process-based parallelism.
    Returns the list of universe ids in order of completion.
    """
    universes = list(universes)
    logger.info(
        f"[PIPE] Submitting {len(universes)} universes to ProcessPoolExecutor "
        f"(max_workers={max_workers})"
    )

    completed: List[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_universe = {
            executor.submit(_worker_run_full_pipeline, universe): universe
            for universe in universes
        }

        for future in as_completed(future_to_universe):
            universe = future_to_universe[future]
            try:
                universe_id = future.result()
            except Exception:
                logger.exception(
                    "[PIPE] Run failed for universe=%s", universe.to_id_string()
                )
            else:
                logger.info(
                    "[PIPE] Run finished successfully for universe=%s", universe_id
                )
                completed.append(universe_id)

    return completed
