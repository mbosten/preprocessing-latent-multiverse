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


PCA_DIMS = range(1, 6, 1)
SEED = 42
SUBSAMPLE_SIZE = 100_000
NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI


logger = logging.getLogger(__name__)


def main():
    """
    How do landscape norms and computation times change as the number of projected PCA components increases?

    Preliminary results: When max PCA components = 5 and 100k sample, in many cases norms increase as components increase.
    This does not support suitability of 3 PCA components, for example.
    """
    parser = argparse.ArgumentParser(description="PCA effects on landscape norms")
    parser.add_argument("--universe-index", dest="uid", default=0, type=int)
    args = parser.parse_args()

    u = get_universe(args.uid)
    logger.info(f"Processing universe: {u.id}")

    stem = (
        f"pca_dims_universe_{u.id}_"
        f"{max(PCA_DIMS)}dims_{SUBSAMPLE_SIZE}"
    )

    exp = Experiment(
        name = "pca_dim_experiment",
        stem = stem,
        parameter_name = "pca_components",
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
    point_cloud.sample(target_size=SUBSAMPLE_SIZE, seed=SEED, inplace=True)
    point_cloud.normalize(method="diameter", seed=SEED, iterations=1000, inplace=True)

    del latent

    exp.timings = {"persistence": {}, "landscapes": {}, "metrics": {}}

    for components in PCA_DIMS:
        logger.info("Processing %d PCA components.", components)

        projected = point_cloud.project_PCA(n_components=components, seed=SEED, inplace=False)
        hom_dims = tuple(range(min(components, 3)))
        tda = Persistence(universe=u, points=projected)

        # NOTE: I have used Delauney in prior runs. What is the difference again? Integrate where necessary.
        with exp.timer("persistence", components):
            tda.compute_intervals(precision="exact", hom_dims=hom_dims)

        with exp.timer("landscapes", components):
            tda.compute_landscapes(hom_dims=hom_dims)

        with exp.timer("metrics", components):
            exp.results[components] = {
                "norms": tda.landscape_norms(),
                "persistence": tda.total_persistence(),
            }

        del tda, projected

    exp.plot_timings("persistence")
    exp.plot_timings("landscapes")

    exp.plot_metric("norms", "Landscape vector norm", norm_out)
    exp.plot_metric("persistence", "Total persistence", total_out)

    fields = [
        "universe_id",
        "seed",
        "n_points",
        "n_latent_dim",
        "pca_components",
        "norm_H0",
        "norm_H1",
        "norm_H2",
        "sum_H0",
        "sum_H1",
        "sum_H2",
    ]

    rows = []

    for components, result in sorted(exp.results.items()):
        norms = result["norms"]
        persistence = result["persistence"]

        rows.append([
            u.id,
            SEED,
            latent_N,
            latent_dim,
            components,
            *(norms.get(dim, np.nan) for dim in range(3)),
            *(persistence.get(dim, np.nan) for dim in range(3)),
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

