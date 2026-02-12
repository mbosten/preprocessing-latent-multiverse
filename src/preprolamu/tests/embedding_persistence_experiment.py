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
from preprolamu.pipeline.landscapes import compute_landscapes
from preprolamu.pipeline.metrics import compute_landscape_norm
from preprolamu.pipeline.persistence import mask_infinities
from preprolamu.pipeline.universes import get_universe

# --------- Logistics ---------
setup_logging(log_dir=Path("logs"))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Landscape norms on full embedding dimensionality."
)

parser.add_argument(
    "--universe-index",
    dest="uid",
    default=99,
    type=int,
)

args = parser.parse_args()


# --------- Globals ---------
SAMPLE_SIZES = [100, 250, 500, 750, 1000]
SEEDS = [1, 11, 111, 1111, 11111]
NORMFIGSIZE = (12, 8)  # inches
TIMEFIGSIZE = (8, 6)  # inches
DPI = 300  # fixed DPI

# ---------- Paths ---------
data_dir = Path("data/experiments/embedding_persistence_experiment")
data_dir.mkdir(parents=True, exist_ok=True)

fig_dir = Path("data/figures/embedding_persistence_experiment")
fig_dir.mkdir(parents=True, exist_ok=True)

persistence_path = data_dir / "persistence_results"
persistence_path.mkdir(parents=True, exist_ok=True)

landscape_path = data_dir / "landscape_results"
landscape_path.mkdir(parents=True, exist_ok=True)

norm_csv_path = data_dir / f"landscape_norms_{args.uid}.csv"

norm_fig_path = fig_dir / f"landscape_norm_fig_{args.uid}.png"
# --------- Script ---------

# - Loading data
u = get_universe(args.uid)
embed = u.io.load_embedding(split="test", force_recompute=False)
N, D = embed.shape
logger.info(f"Loaded universe {args.uid} with embedding shape {embed.shape}.")

# - Output data objects
persistence_results = {}
persistence_timings = []


for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for k in SAMPLE_SIZES:

        persistence_time = time.perf_counter()
        indices = rng.choice(N, size=k, replace=False)
        df = embed[indices]
        logger.info(f"Processing seed={seed}, embedding shape={df.shape}...")
        ac = gd.AlphaComplex(points=df, precision="safe")
        logger.info("Creating simplex tree...")
        st = ac.create_simplex_tree()
        logger.info("Computing persistence...")
        st.compute_persistence()
        logger.info("Combining results.")

        per_dim: dict[int, np.ndarray] = {}
        for dim in [0, 1, 2]:
            per_dim[dim] = mask_infinities(st.persistence_intervals_in_dimension(dim))

        persistence_results[(seed, k)] = per_dim
        persistence_elapsed = time.perf_counter() - persistence_time
        persistence_timings.append((seed, k, persistence_elapsed))

        # store results
        logger.info("Saving persistence object")
        path = persistence_path / f"persistence_seed{seed}_k{k}.npz"
        arrays = {f"dim{d}_intervals": arr for d, arr in per_dim.items()}
        np.savez(path, **arrays)


logger.info(f"{'Sample size':>12} | {'Seed':>6} | {'Persistence Time (s)':>20}")
logger.info("-" * 45)
for seed, k, t in persistence_timings:
    logger.info(f"{k:12d} | {seed:6d} | {t:20.3f}")


# Active memory management (for what it is worth)
del embed
gc.collect()


landscape_results = {}
landscape_timings = []

for (seed, k), results in persistence_results.items():
    landscape_start = time.perf_counter()
    logger.info(f"Computing landscapes for seed={seed}, sample size={k}")

    landscapes = compute_landscapes(
        persistence_per_dimension=results,
        num_landscapes=5,
        resolution=1000,
        homology_dimensions=[0, 1, 2],
    )

    landscape_results[(seed, k)] = landscapes

    landscape_elapsed = time.perf_counter() - landscape_start
    landscape_timings.append(((seed, k), landscape_elapsed))

    logger.info("Saving landscape object")
    # store results
    path = landscape_path / f"landscape_seed{seed}_k{k}.npz"
    arrays = {
        f"dim{d}_landscapes": arr for d, arr in landscapes.items() if arr is not None
    }
    np.savez(path, **arrays)

logger.info(f"{'Seed':>6} | {'Sample size':>12} | {'Time (s)':>8}")
logger.info("-" * 45)
for (seed, k), t in landscape_timings:
    logger.info(f"{seed:6d} | {k:12d} | {t:8.3f}")


norm_results = {}

for (seed, k), landscapes in landscape_results.items():
    dim_norms = compute_landscape_norm(landscapes, score_type="separate")
    norm_results[(seed, k)] = dim_norms


with norm_csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["universe_id", "sampler", "seed", "sample_size", "H0", "H1", "H2"])
    for seed, k in sorted(norm_results.keys()):
        h0 = float(norm_results[(seed, k)][0])
        h1 = float(norm_results[(seed, k)][1])
        h2 = float(norm_results[(seed, k)][2])
        writer.writerow([u.id, args.sampler, seed, k, h0, h1, h2])

logger.info(f"Wrote norms to {norm_csv_path}")


# Sort x-axis
x = sorted(norm_results.keys())

# Extract y-values for each key (0, 1, 2)
y0 = [norm_results[(seed, k)][0] for (seed, k) in x]
y1 = [norm_results[(seed, k)][1] for (seed, k) in x]
y2 = [norm_results[(seed, k)][2] for (seed, k) in x]

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

fig.savefig(norm_fig_path, dpi=DPI)
plt.close()
