# import libraries
import argparse
import csv
import gc
import logging

# For timing
import sys
import time
from pathlib import Path

# For alpha complexes
import gudhi as gd

# Regular plots
import matplotlib.pyplot as plt
import numpy as np

from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.landscapes import compute_landscapes
from preprolamu.pipeline.metrics import compute_landscape_norm
from preprolamu.pipeline.persistence import mask_infinities
from preprolamu.pipeline.universes import get_universe

setup_logging(log_dir=Path("logs"))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="sample size effect on landscape norm under multiple seeds"
)

NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI

parser.add_argument(
    "--universe-index",
    dest="uid",
    default=0,
    type=int,
)


args = parser.parse_args()

# PRIORS
u = get_universe(383)
logger.info(f"Processing universe: {u.id}")
seeds = [1, 11, 111, 1111, 11111]
sample_sizes = list(range(20000, 510000, 20000))

# PATHS
out_dir = Path("data/figures/u383_seed_sample_size_experiment")
out_dir.mkdir(parents=True, exist_ok=True)

projection_path = u.paths.projected(split="test", normalized=True)

# persistence_out_path = (
#     out_dir / f"persistence_time_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
# )
# landscape_out_path = (
#     out_dir / f"landscape_time_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
# )
results_dir = Path("data/experiments/u383_seed_sample_size_experiment")
results_dir.mkdir(parents=True, exist_ok=True)

norm_csv_path = results_dir / f"landscape_norm_sample_size_universe_{u.id}.csv"
norm_out_path = (
    out_dir / f"landscape_norm_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)


# FUNCTIONS
def random_sample_indices(n_points: int, k: int, seed=42) -> np.ndarray:
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")
    else:
        rng = np.random.default_rng(seed)
    if k > n_points:
        raise ValueError("k cannot exceed n_points when sampling without replacement.")
    return rng.choice(n_points, size=k, replace=False)


# Abort script if all output files already exist
paths = [
    norm_csv_path,
    norm_out_path,
]

if all(p.exists() for p in paths):
    logger.info("All output files already exist. Exiting.")
    sys.exit(0)

# load data
projection = np.load(projection_path)
logger.info(f"Loaded projection with shape: {projection.shape}")

N = projection.shape[0]

sample_size_persistence_results = {}
persistence_timings = []

# Compute persistence per sample size and for random sampling initially.
for seed in seeds:
    rng = np.random.default_rng(seed)
    for size in sample_sizes:
        persistence_start = time.perf_counter()
        logger.info(f"Processing sample size: {size} with seed: {seed}")

        size = min(size, N)  # Ensure we don't sample more than available points

        if size == N:
            logger.info(
                "Sample size equals total number of points. Using full dataset."
            )
            Xrng = projection
        else:
            indices = random_sample_indices(N, k=size, seed=seed)
            Xrng = projection[indices]

        ac = gd.AlphaComplex(points=Xrng, precision="exact")
        simplex_tree = ac.create_simplex_tree()
        simplex_tree.compute_persistence()

        per_dim: dict[int, np.ndarray] = {}
        for dim in [0, 1, 2]:
            per_dim[dim] = mask_infinities(
                simplex_tree.persistence_intervals_in_dimension(dim)
            )

        sample_size_persistence_results[(seed, size)] = per_dim

        persistence_elapsed = time.perf_counter() - persistence_start
        persistence_timings.append((seed, size, persistence_elapsed))

        if size == N:
            logger.info("Reached full dataset size. Stopping further computations.")
            break

logger.info(f"{'Seed':>6} | {'Sample size':>12} | {'Time (s)':>8}")
logger.info("-" * 45)
for seed, size, t in persistence_timings:
    logger.info(f"{seed:6d} | {size:12d} | {t:8.3f}")

# Active memory management
del projection, Xrng
gc.collect()


sample_size_landscape_results = {}
landscape_timings = []
for (seed, size), results in sample_size_persistence_results.items():
    landscape_start = time.perf_counter()
    logger.info(f"Processing sample size: {size} with seed: {seed} for landscapes")

    landscapes = compute_landscapes(
        persistence_per_dimension=results,
        num_landscapes=5,
        resolution=1000,
        homology_dimensions=[0, 1, 2],
    )

    sample_size_landscape_results[(seed, size)] = landscapes

    landscape_elapsed = time.perf_counter() - landscape_start
    landscape_timings.append((seed, size, landscape_elapsed))

logger.info(f"{'Seed':>6} | {'Sample size':>12} | {'Time (s)':>8}")
logger.info("-" * 45)
for seed, size, t in landscape_timings:
    logger.info(f"{seed:6d} | {size:12d} | {t:8.3f}")


sample_size_norm_results = {}

for (seed, size), landscapes in sample_size_landscape_results.items():
    dim_norms = compute_landscape_norm(landscapes, score_type="separate")
    sample_size_norm_results[(seed, size)] = dim_norms


with norm_csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["universe_id", "seed", "sample_size", "H0", "H1", "H2"])
    for seed, size in sorted(sample_size_norm_results.keys()):
        h0 = float(sample_size_norm_results[(seed, size)][0])
        h1 = float(sample_size_norm_results[(seed, size)][1])
        h2 = float(sample_size_norm_results[(seed, size)][2])
        writer.writerow([u.id, seed, size, h0, h1, h2])

logger.info(f"Wrote norms to {norm_csv_path}")


# Sort x-axis
x = sorted(sample_size_norm_results.keys())

# Extract y-values for each key (0, 1, 2)
y0 = [sample_size_norm_results[(seed, k)][0] for (seed, k) in x]
y1 = [sample_size_norm_results[(seed, k)][1] for (seed, k) in x]
y2 = [sample_size_norm_results[(seed, k)][2] for (seed, k) in x]

# Plot
fig, ax = plt.subplots(figsize=NORMFIGSIZE, dpi=DPI)
ax.plot(x, y0, label="H0")
ax.plot(x, y1, label="H1")
ax.plot(x, y2, label="H2")

ax.set_xlabel("Sample Size", fontsize=20, labelpad=12)
ax.set_ylabel("Landscape L2 Norm", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.legend(fontsize=18)
fig.tight_layout(pad=1.5)

fig.savefig(norm_out_path, dpi=DPI)
plt.close()
