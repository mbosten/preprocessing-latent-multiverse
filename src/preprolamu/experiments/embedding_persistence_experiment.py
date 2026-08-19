import argparse

import logging
from pathlib import Path

import numpy as np
from project_utils import setup_logging

from preprolamu.pipeline.embeddings import Embedding
from preprolamu.experiments.experiment import Experiment
from preprolamu.pipeline.persistence import Persistence
from preprolamu.pipeline.universes import get_universe



SAMPLE_SIZES = [100, 250, 500, 750, 1000]
SEEDS = [1, 11, 111, 1111, 11111]
NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI


logger = logging.getLogger(__name__)


def main():
    """
    This is likely a redundant experiment since such low sample sizes aren't informative.
    Moreover, full dimensionality is likely still too much to compute.
    """

    parser = argparse.ArgumentParser(description="Landscape norms on full embedding dimensionality.")
    parser.add_argument("--universe-index", dest="uid", default=0, type=int)
    args = parser.parse_args()

    u = get_universe(args.uid)
    logger.info(f"Processing universe: {u.id}")

    stem = (
        f"full_dim_embedding_universe_{u.id}_"
        f"seeds_{SEEDS}_sizes{max(SAMPLE_SIZES)}k"
    )

    exp = Experiment(
        name = "embedding_persistence_experiment",
        stem = stem,
        parameter_name = "sample_size",
    )
    
    persistence_out = exp.figure_path("persistence_time")
    landscape_out = exp.figure_path("landscape_time")
    norm_out = exp.figure_path("landscape_norm")
    total_out = exp.figure_path("total_persistence")
    results_out = exp.results_path("tda_metrics")

    if exp.outputs_exist(
        persistence_out,
        landscape_out,
        norm_out,
        total_out,
        results_out,
    ):
        logger.info("All output files already exist. Exiting.")
        return

    latent = u.io.load_embedding(split="test")
    latent_N, latent_dim = latent.shape
    logger.info("Loaded embedding with shape %s.", latent.shape)

    emb = Embedding(latent_space=latent, universe=u)

    del latent

    exp.timings = {"persistence": {}, "landscapes": {}, "metrics": {}}

    for seed in SEEDS:
        for k in SAMPLE_SIZES:
            logger.info("Processing seed=%d, sample size=%d.", seed, k)

            key = (seed, k)
            df = emb.sample(target_size=k, seed=seed, inplace=False)
            tda = Persistence(universe=u, points=df)

            with exp.timer("persistence", key):
                tda.compute_intervals(precision="exact")

            with exp.timer("landscapes", key):
                tda.compute_landscapes()

            with exp.timer("metrics", key):
                exp.results[key] = {
                    "norms": tda.landscape_norms(),
                    "persistence": tda.total_persistence(),
                }

            del tda, df

    exp.plot_timings("persistence")
    exp.plot_timings("landscapes")

    exp.plot_metric("norms", "Landscape vector norm", norm_out)
    exp.plot_metric("persistence", "Total persistence", total_out)

    fields = [
        "universe_id",
        "n_points",
        "n_latent_dims",
        "seed",
        "sample_size",
        "norm_H0",
        "norm_H1",
        "norm_H2",
        "sum_H0",
        "sum_H1",
        "sum_H2",
    ]

    rows = []

    for (seed, k), results in sorted(exp.results.items()):
        norms = results["norms"]
        sums = results["persistence"]
        rows.append([
            u.id,
            latent_N,
            latent_dim,
            seed,
            k,
            *(norms.get(dim, np.nan) for dim in range(3)),
            *(sums.get(dim, np.nan) for dim in range(3)),
        ])

    exp.save_results(results_out, fields, rows)


if __name__ == "__main__":
    setup_logging(
        log_dir=Path("logs"),
        suppress_loggers=[
            "PIL",
            "matplotlib.font_manager",
            "matplotlib.texmanager",
            "matplotlib.dviread",
        ],
    )
    main()
