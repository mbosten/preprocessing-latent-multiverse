# import libraries
import argparse
import logging

from pathlib import Path

import numpy as np

from preprolamu.experiments.experiment import Experiment
from preprolamu.helpers import setup_logging
from preprolamu.pipeline.embeddings import Embedding
from preprolamu.pipeline.persistence import Persistence
from preprolamu.pipeline.universes import get_universe


SAMPLE_SIZES = range(20000, 510000, 20000)
SEED = 42
N_COMPONENTS = 3
NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI


logger = logging.getLogger(__name__)


def main():
    """
    How do landscape norms and computation times change as subsample size increases?

    Preliminary results: Many universes show a kind of convergence of the norms.
    See the combined results for an overview.
    """
    parser = argparse.ArgumentParser(description="sample size effects on landscape norms")
    parser.add_argument("--universe-index", dest="uid", default=0, type=int)
    args = parser.parse_args()

    u = get_universe(args.uid)
    logger.info(f"Processing universe: {u.id}")

    stem = (
        f"sample_size_universe_{u.id}_"
        f"{max(SAMPLE_SIZES)}k"
    )

    exp = Experiment(
        name = "sample_size_experiment",
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

    # Do not set universe parameter in Embedding class to use the manual seed
    point_cloud = Embedding(latent_space=latent)
    point_cloud.project_PCA(n_components=N_COMPONENTS, seed=SEED)
    point_cloud.normalize(method="diameter", seed=SEED, iterations=1000, inplace=True)

    del latent

    exp.timings = {"persistence": {}, "landscapes": {}, "metrics": {}}

    for size in SAMPLE_SIZES:
        logger.info("Processing sample size: %d", size)

        size = min(size, latent_N)  # Ensure we don't sample more than available points
        hom_dims = tuple(range(min(N_COMPONENTS, 3)))
        
        if size == latent_N:
            logger.info("Sample size equals total number of points. Using full dataset.")
            X = point_cloud.latent_space

        X = point_cloud.sample(target_size=size, seed=SEED, inplace=False)
        tda = Persistence(universe=u, latent_space=X)

        with exp.timer("persistence", size):
            tda.compute_intervals(precision="exact", hom_dims=hom_dims)
        
        with exp.timer("landscapes", size):
            tda.compute_landscapes(hom_dims=hom_dims)

        with exp.timer("metrics", size):
            exp.results[size] = {
                "norms": tda.landscape_norms(),
                "persistence": tda.total_persistence(),
            }

        del tda, X

        
    exp.plot_timings("persistence")
    exp.plot_timings("landscapes")

    exp.plot_metric("norms", "landscape vector norm", norm_out)
    exp.plot_metric("persistence", "total persistence", total_out)

    fields = [
        "universe_id",
        "seed",
        "n_points",
        "n_latent_dim",
        "sample_size",
        "norm_H0",
        "norm_H1",
        "norm_H2",
        "sum_H0",
        "sum_H1",
        "sum_H2",
    ]

    rows = []

    for size, result in sorted(exp.results.items()):
        norms = result["norms"]
        sums = result["persistence"]

        rows.append([
            u.id,
            SEED,
            latent_N,
            latent_dim,
            size,
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