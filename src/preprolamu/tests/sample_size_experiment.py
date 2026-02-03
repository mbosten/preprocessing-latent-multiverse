# import libraries
import argparse
import csv
import logging

# For timing
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

parser = argparse.ArgumentParser(description="sample size effects on landscape norms")

NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI

parser.add_argument(
    "--universe-index",
    dest="uid",
    default=0,
    type=int,
)

parser.add_argument(
    "--sampling-method",
    dest="sampler",
    default="random",
    type=str,
    choices=["random", "fps"],
)

args = parser.parse_args()

u = get_universe(args.uid)
logger.info(f"Processing universe: {u.id}")
seed = 42
out_dir = Path("data/figures")
out_dir.mkdir(parents=True, exist_ok=True)

projection_path = u.paths.projected(split="test", normalized=True)
projection = np.load(projection_path)
logger.info(f"Loaded projection with shape: {projection.shape}")


def random_sample_indices(n_points: int, k: int, seed=42) -> np.ndarray:
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")
    else:
        rng = np.random.default_rng(seed)
    if k > n_points:
        raise ValueError("k cannot exceed n_points when sampling without replacement.")
    return rng.choice(n_points, size=k, replace=False)


def fps_indices(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    X = np.asarray(X)
    N = X.shape[0]

    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= N:
        return np.arange(N, dtype=np.int64)

    rng = np.random.default_rng(seed)
    seed_idx = int(rng.integers(0, N))  # starting point
    selected = np.empty(k, dtype=np.int64)
    selected[0] = seed_idx
    diff = X - X[seed_idx]
    nearest_d2 = np.einsum("ij,ij->i", diff, diff)

    for i in range(1, k):
        idx = int(np.argmax(nearest_d2))
        selected[i] = idx

        diff = X - X[idx]
        d2 = np.einsum("ij,ij->i", diff, diff)
        nearest_d2 = np.minimum(nearest_d2, d2)

    return selected


sample_sizes = list(range(10000, 410000, 20000))
N = projection.shape[0]

sample_size_persistence_results = {}
persistence_timings = []

# Compute persistence per sample size and for random sampling initially.
for size in sample_sizes:
    persistence_start = time.perf_counter()
    logger.info(f"Processing sample size: {size}")
    if args.sampler == "fps":
        indices = fps_indices(projection, k=size, seed=seed)
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

    sample_size_persistence_results[size] = per_dim

    persistence_elapsed = time.perf_counter() - persistence_start
    persistence_timings.append((size, persistence_elapsed))

logger.info(f"{'Sample size':>12} | {'Time (s)':>8}")
logger.info("-" * 25)
for size, t in persistence_timings:
    logger.info(f"{size:12d} | {t:8.3f}")

persistence_sizes = [s for s, _ in persistence_timings]
persistence_times = [t for _, t in persistence_timings]

fig, ax = plt.subplots(figsize=TIMEFIGSIZE, dpi=DPI)
ax.plot(persistence_sizes, persistence_times)
ax.set_xlabel("Sample size", fontsize=20, labelpad=12)
ax.set_ylabel("Computation time (s)", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=16)
fig.tight_layout(pad=1.5)
persistence_out_path = (
    out_dir / f"persistence_time_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)
fig.savefig(persistence_out_path, dpi=DPI)
plt.close()

sample_size_landscape_results = {}
landscape_timings = []
for size, results in sample_size_persistence_results.items():
    landscape_start = time.perf_counter()
    logger.info(f"Processing sample size: {size}")

    landscapes = compute_landscapes(
        persistence_per_dimension=results,
        num_landscapes=5,
        resolution=1000,
        homology_dimensions=[0, 1, 2],
    )

    sample_size_landscape_results[size] = landscapes

    landscape_elapsed = time.perf_counter() - landscape_start
    landscape_timings.append((size, landscape_elapsed))

logger.info(f"{'Sample size':>12} | {'Time (s)':>8}")
logger.info("-" * 25)
for size, t in landscape_timings:
    logger.info(f"{size:12d} | {t:8.3f}")

landscape_sizes = [s for s, _ in landscape_timings]
landscape_times = [t for _, t in landscape_timings]

fig, ax = plt.subplots(figsize=TIMEFIGSIZE, dpi=DPI)
ax.plot(landscape_sizes, landscape_times)
ax.set_xlabel("Sample size", fontsize=20, labelpad=12)
ax.set_ylabel("Computation time (s)", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=16)
fig.tight_layout(pad=1.5)
landscape_out_path = (
    out_dir / f"landscape_time_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)
fig.savefig(landscape_out_path, dpi=DPI)
plt.close()

sample_size_norm_results = {}

for size, landscapes in sample_size_landscape_results.items():
    dim_norms = compute_landscape_norm(landscapes, score_type="separate")
    sample_size_norm_results[size] = dim_norms

# Store norms in universe-csv on disk to aggregate later
results_dir = Path("data/experiments/sample_size_experiment")
results_dir.mkdir(parents=True, exist_ok=True)

norm_csv_path = (
    results_dir / f"landscape_norm_sample_size_universe_{u.id}_{args.sampler}.csv"
)

with norm_csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["universe_id", "sampler", "seed", "sample_size", "H0", "H1", "H2"])
    for size in sorted(sample_size_norm_results.keys()):
        h0 = float(sample_size_norm_results[size][0])
        h1 = float(sample_size_norm_results[size][1])
        h2 = float(sample_size_norm_results[size][2])
        writer.writerow([u.id, args.sampler, seed, size, h0, h1, h2])

logger.info(f"Wrote norms to {norm_csv_path}")


# Sort x-axis
x = sorted(sample_size_norm_results.keys())

# Extract y-values for each key (0, 1, 2)
y0 = [sample_size_norm_results[k][0] for k in x]
y1 = [sample_size_norm_results[k][1] for k in x]
y2 = [sample_size_norm_results[k][2] for k in x]

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

norm_out_path = (
    out_dir / f"landscape_norm_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)
fig.savefig(norm_out_path, dpi=DPI)
plt.close()
