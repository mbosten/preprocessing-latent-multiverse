# src/alphacomplexbenchmarking/pipeline/parallel.py
from __future__ import annotations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List
import numpy as np

from alphacomplexbenchmarking.pipeline.specs import RunSpec
from alphacomplexbenchmarking.pipeline.preprocessing import preprocess_variant
from alphacomplexbenchmarking.pipeline.autoencoder import train_autoencoder_for_variant
from alphacomplexbenchmarking.pipeline.embeddings import compute_embeddings_and_subsample_for_tda
from alphacomplexbenchmarking.pipeline.tda import run_tda_on_points
from alphacomplexbenchmarking.pipeline.metrics import compute_metrics_from_tda
from alphacomplexbenchmarking.io.storage import (
    get_embedding_path,
    load_numpy_array,
    get_tda_result_path,
    get_metrics_path,
    save_tda_npz,
    save_json,
)

logger = logging.getLogger(__name__)


def run_full_pipeline_for_spec(spec: RunSpec) -> str:
    """
    Full multiverse pipeline for a single RunSpec (universe):
      1. Preprocess
      2. Train AE (placeholder for now)
      3. Compute embeddings + PCA + subsample
      4. Run TDA (alpha + landscapes)
      5. Compute summary metrics
      6. Save results

    Returns the spec id string for logging.
    """
    spec_id = spec.to_id_string()
    logger.info(f"[PIPE] Starting full pipeline for spec={spec_id}")

    # 1. Preprocess
    preprocess_variant(spec)

    # 2. Train AE (placeholder)
    train_autoencoder_for_variant(spec)

    # 3. Embeddings + PCA + subsample
    emb_path = compute_embeddings_and_subsample_for_tda(spec)

    # 4. TDA
    points = load_numpy_array(emb_path)
    tda_result = run_tda_on_points(points, spec.tda_config)

    # Save TDA arrays
    tda_path = get_tda_result_path(spec)
    
    # Flatten dicts for npz; we can encode keys like "pers_dim_0", "land_dim_0", etc.
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
        "spec_id": spec_id,
        "total_persistence_per_dim": metrics.total_persistence_per_dim,
        "landscape_l2_per_dim": metrics.landscape_l2_per_dim,
    }
    metrics_path = get_metrics_path(spec)
    save_json(metrics_path, metrics_payload)
    logger.info(f"[PIPE] Saved metrics to {metrics_path}")

    logger.info(f"[PIPE] Finished full pipeline for spec={spec_id}")
    return spec_id


def _worker_run_full_pipeline(spec: RunSpec) -> str:
    """
    Worker function for ProcessPoolExecutor.
    """
    return run_full_pipeline_for_spec(spec)


def run_many_specs(specs: Iterable[RunSpec], max_workers: int | None = None) -> List[str]:
    """
    Run many RunSpecs in parallel using process-based parallelism.
    Returns the list of spec ids in order of completion.
    """
    specs = list(specs)
    logger.info(
        f"[PIPE] Submitting {len(specs)} specs to ProcessPoolExecutor "
        f"(max_workers={max_workers})"
    )

    completed: List[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {
            executor.submit(_worker_run_full_pipeline, spec): spec
            for spec in specs
        }

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                spec_id = future.result()
            except Exception:
                logger.exception("[PIPE] Run failed for spec=%s", spec.to_id_string())
            else:
                logger.info("[PIPE] Run finished successfully for spec=%s", spec_id)
                completed.append(spec_id)

    return completed