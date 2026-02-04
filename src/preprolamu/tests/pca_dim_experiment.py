# import libraries
import argparse
import csv
import gc
import logging
import time
from pathlib import Path

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np

from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.embeddings import normalize_space, project_PCA
from preprolamu.pipeline.landscapes import compute_landscapes
from preprolamu.pipeline.metrics import compute_landscape_norm
from preprolamu.pipeline.persistence import mask_infinities
from preprolamu.pipeline.universes import get_universe

setup_logging(log_dir=Path("logs"))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="sample size effects on landscape norms")

NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI
SUBSAMPLE_SIZE = 100_000

parser.add_argument(
    "--universe-index",
    dest="uid",
    default=0,
    type=int,
)

args = parser.parse_args()

u = get_universe(args.uid)
logger.info(f"Processing universe: {u.id}")
seed = 42
out_dir = Path("data/figures/pca_dim_experiment")
out_dir.mkdir(parents=True, exist_ok=True)

latent = u.io.load_embedding(split="test", force_recompute=False)
logger.info(f"Loaded projection with shape: {latent.shape}")
N, D = latent.shape


def random_sample_indices(n_points: int, k: int, seed=42) -> np.ndarray:
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")
    else:
        rng = np.random.default_rng(seed)
    if k > n_points:
        raise ValueError("k cannot exceed n_points when sampling without replacement.")
    return rng.choice(n_points, size=k, replace=False)


pca_persistence_results = {}
persistence_timings = []

persistence_start = time.perf_counter()

# Sample embedding space to ensure reasonable computation times when increasing pca components
indices = random_sample_indices(N, SUBSAMPLE_SIZE, seed=seed)
X = latent[indices]
logger.info(X.shape)

# Active memory management
del latent
gc.collect()

# Diameter division to normalize the data
Xnorm, diameter = normalize_space(X, seed=seed, diameter_iterations=1000)
pca_dims = list(range(1, 6, 1))

# Active memory management
del X
gc.collect()

for components in pca_dims:
    logger.info(components)
    Xproj = project_PCA(Xnorm, n_components=components, seed=seed)

    if components < 3:
        hom_range = range(components)
    else:
        hom_range = range(3)

    logger.info("Complex...")
    ac = gd.DelaunayCechComplex(points=Xproj, precision="safe")
    # ac = gd.AlphaComplex(points=Xproj, precision="exact")
    logger.info("Simplex tree...")
    simplex_tree = ac.create_simplex_tree()
    logger.info("Persistence...")
    simplex_tree.compute_persistence(homology_coeff_field=2)

    per_dim: dict[int, np.ndarray] = {}

    logger.info("Intervals...")
    for hom_dim in hom_range:
        per_dim[hom_dim] = mask_infinities(
            simplex_tree.persistence_intervals_in_dimension(hom_dim)
        )

    pca_persistence_results[components] = per_dim

    persistence_elapsed = time.perf_counter() - persistence_start
    logger.info(f"Elapsed time: {persistence_elapsed:.3f} seconds")
    persistence_timings.append((components, persistence_elapsed))

logger.info(f"{'PCA components':>12} | {'Time (s)':>8}")
logger.info("-" * 25)
for components, t in persistence_timings:
    logger.info(f"{components:12d} | {t:8.3f}")

# Active memory management
del Xnorm
gc.collect()

pca_components = [s for s, _ in persistence_timings]
persistence_times = [t for _, t in persistence_timings]

fig, ax = plt.subplots(figsize=TIMEFIGSIZE, dpi=DPI)
ax.plot(pca_components, persistence_times)
ax.set_xlabel("PCA components", fontsize=20, labelpad=12)
ax.set_ylabel("Computation time (s)", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=16)
fig.tight_layout(pad=1.5)
persistence_out_path = (
    out_dir
    / f"persistence_time_pca_dims_universe_{u.id}_{max(pca_dims)}dims_{SUBSAMPLE_SIZE}.png"
)
fig.savefig(persistence_out_path, dpi=DPI)
plt.close()


pca_landscape_results = {}
landscape_timings = []
for components, results in pca_persistence_results.items():
    landscape_start = time.perf_counter()
    logger.info(components)

    if components < 3:
        hom_list = list(range(components))
    else:
        hom_list = list(range(3))

    if max(hom_list) > 2:
        raise ValueError(
            "Landscapes can only be computed for homology dimensions 0, 1, and 2."
        )

    landscapes = compute_landscapes(
        persistence_per_dimension=results,
        num_landscapes=5,
        resolution=1000,
        homology_dimensions=hom_list,
    )

    pca_landscape_results[components] = landscapes

    landscape_elapsed = time.perf_counter() - landscape_start
    landscape_timings.append((components, landscape_elapsed))

logger.info(f"{'PCA components':>12} | {'Time (s)':>8}")
logger.info("-" * 25)
for components, t in landscape_timings:
    logger.info(f"{components:12d} | {t:8.3f}")


components = [s for s, _ in landscape_timings]
landscape_times = [t for _, t in landscape_timings]

fig, ax = plt.subplots(figsize=TIMEFIGSIZE, dpi=DPI)
ax.plot(components, landscape_times)
ax.set_xlabel("Number of PCA components", fontsize=20, labelpad=12)
ax.set_ylabel("Computation time (s)", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=16)
fig.tight_layout(pad=1.5)
landscape_out_path = (
    out_dir
    / f"landscape_time_pca_dims_universe_{u.id}_{max(pca_dims)}dims_{SUBSAMPLE_SIZE}.png"
)
fig.savefig(landscape_out_path, dpi=DPI)
plt.close()

pca_norm_results = {}
for size, landscapes in pca_landscape_results.items():
    dim_norms = compute_landscape_norm(landscapes, score_type="separate")
    pca_norm_results[size] = dim_norms

# Store universe-level data to disk
results_dir = Path("data/experiments/pca_dim_experiment")
results_dir.mkdir(parents=True, exist_ok=True)

results_path = (
    results_dir
    / f"landscape_norm_pca_dims_universe_{u.id}_{max(pca_dims)}dims_{SUBSAMPLE_SIZE}.csv"
)

with results_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "universe_id",
            "seed",
            "n_points",
            "n_latent_dim",
            "pca_components",
            "H0",
            "H1",
            "H2",
        ]
    )

    for comps in sorted(pca_norm_results.keys()):
        norms = pca_norm_results[comps]
        h0 = float(norms.get(0, np.nan))
        h1 = float(norms.get(1, np.nan))
        h2 = float(norms.get(2, np.nan))

        writer.writerow([u.id, seed, SUBSAMPLE_SIZE, int(D), int(comps), h0, h1, h2])

logger.info(f"Wrote norm table to {results_path}")

x = sorted(pca_norm_results.keys())

y0 = [pca_norm_results[k].get(0, np.nan) for k in x]
y1 = [pca_norm_results[k].get(1, np.nan) for k in x]
y2 = [pca_norm_results[k].get(2, np.nan) for k in x]

fig, ax = plt.subplots(figsize=NORMFIGSIZE, dpi=DPI)
ax.plot(x, y0, label="H0")
ax.plot(x, y1, label="H1")
ax.plot(x, y2, label="H2")

ax.set_xlabel("PCA Component", fontsize=20, labelpad=12)
ax.set_ylabel("Landscape L2 Norm", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.legend(fontsize=18)
fig.tight_layout(pad=1.5)
norm_out_path = (
    out_dir
    / f"landscape_norm_pca_dims_universe_{u.id}_{max(pca_dims)}dims_{SUBSAMPLE_SIZE}.png"
)
fig.savefig(norm_out_path, dpi=DPI)
plt.close()
