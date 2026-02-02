# import libraries
import argparse

# For timing
import time
from pathlib import Path

# For alpha complexes
import gudhi as gd

# Regular plots
import matplotlib.pyplot as plt
import numpy as np

from preprolamu.pipeline.landscapes import compute_landscapes
from preprolamu.pipeline.metrics import compute_landscape_norm
from preprolamu.pipeline.persistence import mask_infinities
from preprolamu.pipeline.universes import get_universe

parser = argparse.ArgumentParser(description="sample size effects on landscape norms")

parser.add_argument(
    "--universe-index",
    dest="uid",
    default=0,
    type=int,
)

args = parser.parse_args()

u = get_universe(args.uid)
print(f"Processing universe: {u.id}", flush=True)
seed = 42
out_dir = Path("data/figures")
out_dir.mkdir(parents=True, exist_ok=True)

projection_path = u.paths.projected(split="test", normalized=True)
projection = np.load(projection_path)
print(f"Loaded projection with shape: {projection.shape}", flush=True)


def random_sample_indices(n_points: int, k: int, seed=42) -> np.ndarray:
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")
    else:
        rng = np.random.default_rng(seed)
    if k > n_points:
        raise ValueError("k cannot exceed n_points when sampling without replacement.")
    return rng.choice(n_points, size=k, replace=False)


sample_sizes = list(range(10000, 360000, 10000))
N = projection.shape[0]

sample_size_persistence_results = {}
persistence_timings = []

# Compute persistence per sample size and for random sampling initially.
for size in sample_sizes:
    persistence_start = time.perf_counter()
    print(f"Processing sample size: {size}", flush=True)
    random_indices = random_sample_indices(N, size, seed)
    Xrng = projection[random_indices]

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

print(f"{'Sample size':>12} | {'Time (s)':>8}", flush=True)
print("-" * 25, flush=True)
for size, t in persistence_timings:
    print(f"{size:12d} | {t:8.3f}", flush=True)

persistence_sizes = [s for s, _ in persistence_timings]
persistence_times = [t for _, t in persistence_timings]

plt.plot(persistence_sizes, persistence_times)
plt.xlabel("Sample size", fontsize=20, labelpad=12)
plt.ylabel("Computation time (s)", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=16)
persistence_out_path = (
    out_dir / f"persistence_time_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)
plt.savefig(persistence_out_path, dpi=300, bbox_inches="tight")
plt.close()

sample_size_landscape_results = {}
landscape_timings = []
for size, results in sample_size_persistence_results.items():
    landscape_start = time.perf_counter()
    print(f"Processing sample size: {size}", flush=True)

    landscapes = compute_landscapes(
        persistence_per_dimension=results,
        num_landscapes=5,
        resolution=1000,
        homology_dimensions=[0, 1, 2],
    )

    sample_size_landscape_results[size] = landscapes

    landscape_elapsed = time.perf_counter() - landscape_start
    landscape_timings.append((size, landscape_elapsed))

print(f"{'Sample size':>12} | {'Time (s)':>8}", flush=True)
print("-" * 25, flush=True)
for size, t in landscape_timings:
    print(f"{size:12d} | {t:8.3f}", flush=True)

landscape_sizes = [s for s, _ in landscape_timings]
landscape_times = [t for _, t in landscape_timings]

plt.plot(landscape_sizes, landscape_times)
plt.xlabel("Sample size", fontsize=20, labelpad=12)
plt.ylabel("Computation time (s)", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=16)
landscape_out_path = (
    out_dir / f"landscape_time_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)
plt.savefig(landscape_out_path, dpi=300, bbox_inches="tight")
plt.close()

sample_size_norm_results = {}

for size, landscapes in sample_size_landscape_results.items():
    dim_norms = compute_landscape_norm(landscapes, score_type="separate")
    sample_size_norm_results[size] = dim_norms


# Sort x-axis
x = sorted(sample_size_norm_results.keys())

# Extract y-values for each key (0, 1, 2)
y0 = [sample_size_norm_results[k][0] for k in x]
y1 = [sample_size_norm_results[k][1] for k in x]
y2 = [sample_size_norm_results[k][2] for k in x]

# Plot
plt.figure(figsize=(12, 8))
plt.plot(x, y0, label="H0")
plt.plot(x, y1, label="H1")
plt.plot(x, y2, label="H2")

plt.xlabel("Sample Size", fontsize=20, labelpad=12)
plt.ylabel("Landscape L2 Norm", fontsize=20)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.legend(fontsize=18)
norm_out_path = (
    out_dir / f"landscape_norm_sample_size_universe_{u.id}_{max(sample_sizes)}k.png"
)
plt.savefig(norm_out_path, dpi=300, bbox_inches="tight")
plt.close()
