# import libraries
import argparse
import csv

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from project_utils import setup_logging

from preprolamu.pipeline.embeddings import Embedding
from preprolamu.pipeline.persistence import Persistence
from preprolamu.pipeline.universes import get_universe



NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI
SUBSAMPLE_SIZE = 100_000
PCA_DIMS = range(1, 6, 1)
SEED = 42

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Functions                                                
# ──────────────────────────────────────────────────────────
def plot_timings(values, out_path):
    logger.info("%12s | %8s", "PCA components", "Time (s)")
    logger.info("-" * 25)

    for components, elapsed in values.items():
        logger.info("%12d | %8.3f", components, elapsed)

    fig, ax = plt.subplots(figsize=TIMEFIGSIZE, dpi=DPI)
    ax.plot(values.keys(), values.values())
    ax.set(
        xlabel="PCA components",
        ylabel="Computation time (s)",
    )
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric(results, key, ylabel, out_path):
    components = sorted(results)
    hom_dims = sorted({
        dim
        for result in results.values()
        for dim in result[key]
    })

    fig, ax = plt.subplots(figsize=NORMFIGSIZE, dpi=DPI)

    for dim in hom_dims:
        ax.plot(
            components,
            [results[c][key].get(dim, np.nan) for c in components],
            label=f"H{dim}",
        )

    ax.set_xlabel("PCA components", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
# ──────────────────────────────────────────────────────────
# Pipeline                                                
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA effects on landscape norms")
    parser.add_argument("--universe-index", dest="uid", default=0, type=int)
    args = parser.parse_args()

    u = get_universe(args.uid)
    logger.info(f"Processing universe: {u.id}")

    stem = (
        f"pca_dims_universe_{u.id}_"
        f"{max(PCA_DIMS)}dims_{SUBSAMPLE_SIZE}"
    )

    out_dir = Path("data/figures/pca_dim_experiment")
    results_dir = Path("data/experiments/pca_dim_experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    persistence_out = out_dir / f"persistence_time_{stem}.png"
    landscape_out = out_dir / f"landscape_time_{stem}.png"
    norm_out = out_dir / f"landscape_norm_{stem}.png"
    total_out = out_dir / f"total_persistence_{stem}.png"
    results_out = results_dir / f"tda_metrics_{stem}.csv"

    outputs = [
        persistence_out,
        landscape_out,
        norm_out,
        total_out,
        results_out,
    ]

    if all(path.exists() for path in outputs):
        logger.info("All output files already exist. Exiting.")
        return

    latent = u.io.load_embedding(split="test", force_recompute=False)
    _, latent_dim = latent.shape
    logger.info("Loaded embedding with shape %s.", latent.shape)

    # Do not set universe parameter in Embedding class to use the manual seed
    point_cloud = Embedding(latent_space=latent)
    point_cloud.sample(target_size=SUBSAMPLE_SIZE, seed=SEED, inplace=True)
    point_cloud.normalize(method="diameter", seed=SEED, iterations=1000)

    del latent

    results = {}
    timings = {"persistence": {}, "landscapes": {}, "metrics": {}}



    for components in PCA_DIMS:
        logger.info("Processing %d PCA components.", components)

        projected = point_cloud.project_PCA(n_components=components, seed=SEED, inplace=False)
        hom_dims = tuple(range(min(components, 3)))
        tda = Persistence(universe=u, points=projected)

        # NOTE: I have used Delauney in prior runs. What is the difference again? Integrate where necessary.
        start = time.perf_counter()
        tda.compute_intervals(precision="exact", hom_dims=hom_dims)
        timings["persistence"][components] = time.perf_counter() - start

        start = time.perf_counter()
        tda.compute_landscapes(hom_dims=hom_dims)
        timings["landscapes"][components] = time.perf_counter() - start

        start = time.perf_counter()
        results[components] = {
            "norms": tda.landscape_norms(),
            "persistence": tda.total_persistence(),
        }
        timings["metrics"][components] = time.perf_counter() - start

        del tda, projected

    plot_timings(timings["persistence"], persistence_out)
    plot_timings(timings["landscapes"], landscape_out)


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

    with results_out.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)

        for components, result in sorted(results.items()):
            norms = result["norms"]
            persistence = result["persistence"]

            writer.writerow([
                u.id,
                SEED,
                len(point_cloud.latent_space),
                latent_dim,
                components,
                *(norms.get(dim, np.nan) for dim in range(3)),
                *(persistence.get(dim, np.nan) for dim in range(3)),
            ])

    logger.info("Wrote metric table to %s.", results_out)

    plot_metric(results, "norms", "Landscape vector norm", norm_out)
    plot_metric(results, "persistence", "Total persistence", total_out)


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

