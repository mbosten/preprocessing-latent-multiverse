# src/alphacomplexbenchmarking/experiments/parameter_sensitivity.py
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
import torch

from alphacomplexbenchmarking.logging_config import setup_logging

# Adjust these imports to your actual module names:
from alphacomplexbenchmarking.pipeline.universes import Universe, generate_multiverse
from alphacomplexbenchmarking.config import load_dataset_config, DatasetConfig
from alphacomplexbenchmarking.io.storage import (
    get_preprocessed_path,
    ensure_parent_dir,
)
from alphacomplexbenchmarking.pipeline.autoencoder import (
    train_autoencoder_for_universe,
    load_autoencoder_for_universe,
    _get_feature_matrix_for_ae,
)
from alphacomplexbenchmarking.pipeline.embeddings import normalize_space
from alphacomplexbenchmarking.pipeline.persistence import (
    compute_alpha_complex_persistence,
)
from alphacomplexbenchmarking.pipeline.landscapes import compute_landscapes
from alphacomplexbenchmarking.visualization import (
    _plot_persistence_diagram,
    _plot_multiple_persistence_diagrams,
    _plot_multiple_barcodes,
)
import typer

logger = logging.getLogger(__name__)
app = typer.Typer(help="Parameter experiments to justify metholodical choices in the multiverse analysis.")


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging",
    ),
):
    """
    Global CLI options, executed before any subcommand.
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_dir=Path("logs"), level=level)
    logger = logging.getLogger(__name__)
    logger.debug("CLI started with verbose=%s", verbose)

# ----------------- Helpers: embeddings + TDA ----------------- #

def _get_latent_embeddings_for_universe(universe: Universe) -> np.ndarray:
    """
    Load preprocessed data for this universe, ensure AE is trained, then
    compute latent embeddings (no PCA yet).
    """
    # Ensure AE is trained (no-op if already done, depending on your implementation)
    train_autoencoder_for_universe(universe)

    # Load preprocessed data (same as in embeddings.py)
    ds_cfg: DatasetConfig = load_dataset_config(universe.dataset_id)
    preprocessed_path = get_preprocessed_path(universe)

    logger.info("[EXP] Loading preprocessed data from %s", preprocessed_path)
    df = pd.read_parquet(preprocessed_path)

    # Same feature selection as in AE training
    X = _get_feature_matrix_for_ae(df, ds_cfg)

    # Load AE
    model = load_autoencoder_for_universe(universe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("[EXP] Computing latent embeddings for shape %s", X.shape)
    with torch.no_grad():
        tensor_X = torch.from_numpy(X).to(device)
        latent_tensor = model.encode(tensor_X)
        latent = latent_tensor.cpu().numpy()

    logger.info("[EXP] Latent representation shape: %s", latent.shape)
    return latent


def _pca_project(latent: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    """
    Normalize latent and project to pca_dim using PCA.
    """
    # diameter normalization
    latent_norm = normalize_space(latent, diameter_iterations=1000, seed=42)

    rng = np.random.default_rng(seed)
    # PCA itself is deterministic, rng only for potential later use.
    pca = PCA(n_components=pca_dim, random_state=seed)
    projected = pca.fit_transform(latent_norm)
    logger.info("[EXP] PCA projection to dim=%d → shape=%s", pca_dim, projected.shape)
    return projected


def _subsample_points(points: np.ndarray, m: int, seed: int) -> np.ndarray:
    """
    Uniformly subsample m points without replacement.
    """
    n = points.shape[0]
    m_eff = min(m, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m_eff, replace=False)
    return points[idx]


def _compute_persistence_and_landscapes_for_points(
    points: np.ndarray,
    homology_dims: List[int],
    num_landscapes: int,
    resolution: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray | None]]:
    """
    Compute persistence intervals and landscapes for a point cloud.
    Returns:
      per_dim:    dim -> intervals array
      landscapes: dim -> landscapes array (or None if no intervals)
    """
    logger.info(
        "[EXP] Computing persistence & landscapes for %d points, dims=%s, num_landscapes=%d, resolution=%d",
        points.shape[0],
        homology_dims,
        num_landscapes,
        resolution,
    )

    per_dim = compute_alpha_complex_persistence(
        data=points,
        homology_dimensions=homology_dims,
    )
    landscapes = compute_landscapes(
        persistence_per_dimension=per_dim,
        num_landscapes=num_landscapes,
        resolution=resolution,
        homology_dimensions=homology_dims,
    )
    return per_dim, landscapes


def _flatten_landscape(ls: np.ndarray | None) -> np.ndarray | None:
    """
    Flatten a landscape array LS.fit_transform(...) into a 1D vector.
    If None, returns None.
    """
    if ls is None:
        return None
    return ls.reshape(-1)


def _l2_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """
    L2 distance between two flattened landscapes; if either is None, return NaN.
    """
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))

# ----------------- Experiment: Subsample size sweep ----------------- #
# example: uv run exp subsample-sweep 0 --max-rows 50000 --subsample-sizes "500,1000,2000,5000,10000" --homology-dim 1 --pca-dim 3 --num-landscapes 5 --resolution 500 --output-dir data/experiments/subsampling
@app.command("subsample-sweep")
def subsample_sweep(
    universe_index: int = typer.Argument(..., help="Index into generate_multiverse() to pick a universe."),
    max_rows: int = typer.Option(50_000, help="Max rows to consider from preprocessed dataset."),
    subsample_sizes: str = typer.Option(
        "500,1000,2000,5000,10000",
        help="Comma-separated list of subsample sizes to test.",
    ),
    homology_dim: int = typer.Option(1, help="Homology dimension to analyze (e.g. 0,1,2)."),
    pca_dim: int = typer.Option(3, help="PCA dimension to use for this sweep."),
    num_landscapes: int = typer.Option(5, help="Number of landscapes."),
    resolution: int = typer.Option(500, help="Landscape resolution."),
    output_dir: Path = typer.Option(
        Path("data/experiments/subsampling"),
        help="Directory to save figures and intermediate data.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    For a single universe, fix PCA dim and sweep subsample sizes.
    We:
      - compute latent embeddings,
      - project to PCA dim,
      - for each subsample size:
          - subsample,
          - compute landscapes,
          - measure L2 distance vs the largest subsample,
          - plot landscapes and distance curves.
    """

    logger.info("[EXP] Running subsample_sweep for universe_index=%d", universe_index)

    # 1. Pick universe
    universes = generate_multiverse()
    if universe_index < 0 or universe_index >= len(universes):
        raise typer.BadParameter(f"universe_index must be in [0, {len(universes)-1}]")
    universe = universes[universe_index]
    logger.info("[EXP] Using universe: %s", universe)

    # 2. Get latent embeddings from AE
    latent = _get_latent_embeddings_for_universe(universe)

    # 3. Optionally limit number of rows (for toy experiments)
    if latent.shape[0] > max_rows:
        latent = latent[:max_rows]
        logger.info("[EXP] Truncated latent to first %d rows for experiment.", max_rows)

    # 4. PCA to fixed dim
    latent_pca = _pca_project(latent, pca_dim=pca_dim, seed=universe.seed)

    # 5. Parse subsample sizes
    subsample_list = [int(x) for x in subsample_sizes.split(",") if x.strip()]
    subsample_list = sorted(subsample_list)
    logger.info("[EXP] Subsample sizes: %s", subsample_list)

    # 6. Compute landscapes for each subsample
    hom_dims = [homology_dim]
    landscapes_per_m: Dict[int, Dict[int, np.ndarray | None]] = {}
    diagrams_per_m: Dict[int, Dict[int, np.ndarray]] = {}

    for m in subsample_list:
        pts = _subsample_points(latent_pca, m, seed=universe.seed + m)
        logger.info("[EXP] Computing landscapes for subsample size m=%d", m)

        per_dim, lsc = _compute_persistence_and_landscapes_for_points(
            points=pts,
            homology_dims=hom_dims,
            num_landscapes=num_landscapes,
            resolution=resolution,
        )
        diagrams_per_m[m] = per_dim
        landscapes_per_m[m] = lsc

    # 7. Compute distances vs largest subsample
    output_dir.mkdir(parents=True, exist_ok=True)
    base_m = max(subsample_list)
    base_ls = _flatten_landscape(landscapes_per_m[base_m].get(homology_dim))

    distances: List[float] = []
    for m in subsample_list:
        vec = _flatten_landscape(landscapes_per_m[m].get(homology_dim))
        d = _l2_distance(vec, base_ls)
        distances.append(d)
        logger.info("[EXP] m=%d, distance vs m=%d: %s", m, base_m, d)

    # 8. Save distances and a few example landscapes
    np.save(output_dir / f"distances_univ{universe_index}_pca{pca_dim}_H{homology_dim}.npy", np.array(distances))

    # 9. Plot distance curve
    plt.figure()
    plt.plot(subsample_list, distances, marker="o")
    plt.xlabel("Subsample size m")
    plt.ylabel(f"L2 distance vs m={base_m}")
    plt.title(f"Landscape stability vs subsample size\nUniverse {universe_index}, PCA={pca_dim}, H={homology_dim}")
    plt.grid(True)
    fig_path = output_dir / f"distance_curve_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved distance curve to %s", fig_path)

    # 10. Plot a few landscapes overlaid (for visual intuition)
    #     We'll pick the smallest, a mid, and the largest subsample
    chosen_ms = sorted({subsample_list[0], subsample_list[len(subsample_list)//2], subsample_list[-1]})
    # Landscapes overlay
    plt.figure()
    for m in chosen_ms:
        ls = landscapes_per_m[m].get(homology_dim)
        if ls is None:
            continue
        y = ls.reshape(-1)
        x = np.linspace(0, 1, y.shape[0])
        plt.plot(x, y, label=f"m={m}")
    plt.xlabel("Landscape sample index (normalized)")
    plt.ylabel("Landscape value")
    plt.title(f"Selected landscapes vs subsample size\nUniverse {universe_index}, PCA={pca_dim}, H={homology_dim}")
    plt.legend()
    fig_path = output_dir / f"landscapes_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved landscape overlay to %s", fig_path)

    # 11. Plot persistence diagrams for the same chosen subsample sizes
    diagrams_for_plot: Dict[int, np.ndarray | None] = {}
    for m in chosen_ms:
        intervals = diagrams_per_m[m].get(homology_dim)
        diagrams_for_plot[m] = intervals

    _plot_multiple_persistence_diagrams(
        diagrams=diagrams_for_plot,
        homology_dim=homology_dim,
        title=f"Persistence diagrams (H={homology_dim})\nUniverse {universe_index}, PCA={pca_dim}, varying m",
        save_path=output_dir / f"pd_combined_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png",
        label_prefix="m=",
    )
    _plot_multiple_barcodes(
        diagrams=diagrams_for_plot,
        title=f"Barcodes (H={homology_dim})\nUniverse {universe_index}, PCA={pca_dim}, varying m",
        save_path=output_dir / f"barcodes_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png",
        label_prefix="m=",
        max_bars_per_group=100,  # adjust if you want more/less
    )


# ----------------- Experiments: PCA dim sweep ----------------- #
# example: uv run exp pca-sweep 0 --max-rows 50000 --pca-dims "2,3,4" --homology-dim 1 --subsample-size 10000 --num-landscapes 5 --resolution 500 --output-dir data/experiments/pca
@app.command("pca-sweep")
def pca_sweep(
    universe_index: int = typer.Argument(..., help="Index into generate_multiverse() to pick a universe."),
    max_rows: int = typer.Option(50_000, help="Max rows to consider from preprocessed dataset."),
    pca_dims: str = typer.Option(
        "2,3,4,6,8",
        help="Comma-separated list of PCA dims to test.",
    ),
    homology_dim: int = typer.Option(1, help="Homology dimension to analyze (e.g. 0,1,2)."),
    subsample_size: int = typer.Option(10_000, help="Subsample size m for all PCA dims."),
    num_landscapes: int = typer.Option(5, help="Number of landscapes."),
    resolution: int = typer.Option(500, help="Landscape resolution."),
    output_dir: Path = typer.Option(
        Path("data/experiments/pca"),
        help="Directory to save figures and intermediate data.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    For a single universe, fix subsample size m and sweep PCA dimensions.
    For each PCA dim:
      - PCA project latent to that dim,
      - subsample m points,
      - compute landscapes,
      - measure L2 distance vs a reference PCA dimension (largest),
      - plot landscapes and distance curves.
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_dir=Path("logs"), level=level)
    logger.info("[EXP] Running pca_sweep for universe_index=%d", universe_index)

    universes = generate_multiverse()
    if universe_index < 0 or universe_index >= len(universes):
        raise typer.BadParameter(f"universe_index must be in [0, {len(universes)-1}]")
    universe = universes[universe_index]
    logger.info("[EXP] Using universe: %s", universe)

    # 1. Get latent embeddings
    latent = _get_latent_embeddings_for_universe(universe)
    if latent.shape[0] > max_rows:
        latent = latent[:max_rows]
        logger.info("[EXP] Truncated latent to first %d rows for experiment.", max_rows)

    # 2. Parse PCA dims
    pca_dim_list = [int(x) for x in pca_dims.split(",") if x.strip()]
    pca_dim_list = sorted(pca_dim_list)
    logger.info("[EXP] PCA dims: %s", pca_dim_list)

    hom_dims = [homology_dim]
    landscapes_per_d: Dict[int, Dict[int, np.ndarray | None]] = {}
    diagrams_per_d: Dict[int, Dict[int, np.ndarray]] = {}

    # 3. For each PCA dim, project, subsample, compute persistence & landscapes
    for d in pca_dim_list:
        pts_full = _pca_project(latent, pca_dim=d, seed=universe.seed + d)
        pts = _subsample_points(pts_full, subsample_size, seed=universe.seed + d + subsample_size)
        logger.info("[EXP] Computing persistence & landscapes for PCA dim d=%d", d)

        per_dim, lsc = _compute_persistence_and_landscapes_for_points(
            points=pts,
            homology_dims=hom_dims,
            num_landscapes=num_landscapes,
            resolution=resolution,
        )
        diagrams_per_d[d] = per_dim
        landscapes_per_d[d] = lsc

    # 4. Distances vs largest PCA dim
    output_dir.mkdir(parents=True, exist_ok=True)
    base_d = max(pca_dim_list)
    base_ls = _flatten_landscape(landscapes_per_d[base_d].get(homology_dim))

    distances: List[float] = []
    for d in pca_dim_list:
        vec = _flatten_landscape(landscapes_per_d[d].get(homology_dim))
        dval = _l2_distance(vec, base_ls)
        distances.append(dval)
        logger.info("[EXP] d=%d, distance vs d=%d: %s", d, base_d, dval)

    np.save(output_dir / f"distances_univ{universe_index}_m{subsample_size}_H{homology_dim}.npy", np.array(distances))

    # 5. Plot distance curve
    plt.figure()
    plt.plot(pca_dim_list, distances, marker="o")
    plt.xlabel("PCA dimension d")
    plt.ylabel(f"L2 distance vs d={base_d}")
    plt.title(f"Landscape stability vs PCA dim\nUniverse {universe_index}, m={subsample_size}, H={homology_dim}")
    plt.grid(True)
    fig_path = output_dir / f"distance_curve_univ{universe_index}_m{subsample_size}_H{homology_dim}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved PCA distance curve to %s", fig_path)

    # 6. Plot selected landscapes overlaid
    chosen_ds = sorted({pca_dim_list[0], pca_dim_list[len(pca_dim_list)//2], pca_dim_list[-1]})
    plt.figure()
    for d in chosen_ds:
        ls = landscapes_per_d[d].get(homology_dim)
        if ls is None:
            continue
        y = ls.reshape(-1)
        x = np.linspace(0, 1, y.shape[0])
        plt.plot(x, y, label=f"d={d}")
    plt.xlabel("Landscape sample index (normalized)")
    plt.ylabel("Landscape value")
    plt.title(f"Selected landscapes vs PCA dim\nUniverse {universe_index}, m={subsample_size}, H={homology_dim}")
    plt.legend()
    fig_path = output_dir / f"landscapes_univ{universe_index}_m{subsample_size}_H{homology_dim}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved PCA landscape overlay to %s", fig_path)

    # 7. Plot persistence diagrams for the same chosen PCA dims
    #    as a single combined figure, color-coded by d
    diagrams_for_plot_d: Dict[int, np.ndarray | None] = {}
    for d in chosen_ds:
        intervals = diagrams_per_d[d].get(homology_dim)
        diagrams_for_plot_d[d] = intervals

    _plot_multiple_persistence_diagrams(
        diagrams=diagrams_for_plot_d,
        homology_dim=homology_dim,
        title=f"Persistence diagrams (H={homology_dim})\nUniverse {universe_index}, m={subsample_size}, varying PCA dim",
        save_path=output_dir / f"pd_combined_univ{universe_index}_H{homology_dim}_m{subsample_size}.png",
        label_prefix="d=",
    )

    # 8. Combined barcode plot for chosen PCA dims
    barcodes_for_plot_d: Dict[int, np.ndarray | None] = {}
    for d in chosen_ds:
        intervals = diagrams_per_d[d].get(homology_dim)
        barcodes_for_plot_d[d] = intervals

    _plot_multiple_barcodes(
        diagrams=barcodes_for_plot_d,
        title=f"Barcodes (H={homology_dim})\nUniverse {universe_index}, m={subsample_size}, varying PCA dim",
        save_path=output_dir / f"barcodes_univ{universe_index}_H{homology_dim}_m{subsample_size}.png",
        label_prefix="d=",
        max_bars_per_group=100,
    )

if __name__ == "__main__":
    app()