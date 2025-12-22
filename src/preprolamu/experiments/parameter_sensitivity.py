# src/preprolamu/experiments/parameter_sensitivity.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.decomposition import PCA
from typing_extensions import Annotated

from preprolamu.io.storage import get_latent_cache_path
from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.autoencoder import train_autoencoder_for_universe
from preprolamu.pipeline.embeddings import (
    compute_embeddings_for_universe,
    from_latent_to_point_cloud,
    normalize_space,
)
from preprolamu.pipeline.landscapes import compute_landscapes
from preprolamu.pipeline.persistence import (
    build_alpha_complex_simplex_tree,
    compute_alpha_complex_persistence,
)

# Adjust these imports to your actual module names:
from preprolamu.pipeline.universes import Universe, get_universe
from preprolamu.visualization import (
    _plot_distance_curve,
    _plot_landscape_overlay,
    _plot_multiple_barcodes,
    _plot_multiple_persistence_diagrams,
    _plot_persistence_diagram,
)

logger = logging.getLogger(__name__)
app = typer.Typer(
    help="Parameter experiments to justify metholodical choices in the multiverse analysis."
)


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
    logger.info("CLI started with verbose=%s", verbose)


# ----------------- Helpers: embeddings + TDA ----------------- #


def _get_latent_embeddings_for_universe(
    universe: Universe, retrain_if_missing: bool = True
):
    """
    Load latent embeddings for this universe if cached, otherwise retrain AE (if allowed) and compute latent embeddings.
    """

    cache_path = get_latent_cache_path(universe)

    if cache_path.exists():
        logger.info("[EXP] Loading cached latent embeddings from %s", cache_path)
        return np.load(cache_path)

    logger.info("[EXP] No cached latent found for universe %s.", universe)

    # Optionally ensure AE is trained
    if retrain_if_missing:
        logger.info(
            "[EXP] Training AE for universe %s (retrain_if_missing=True).", universe
        )
        train_autoencoder_for_universe(universe)
    else:
        logger.info(
            "[EXP] Assuming AE for universe %s is already trained; not retraining.",
            universe,
        )

    latent = compute_embeddings_for_universe(universe)

    logger.info("[EXP] Latent representation shape: %s", latent.shape)
    np.save(cache_path, latent)
    logger.info("[EXP] Saved latent embeddings cache to %s", cache_path)
    return latent


def _pca_project(latent: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    """
    Normalize latent and project to pca_dim using PCA.
    """
    # diameter normalization
    latent_norm = normalize_space(latent, diameter_iterations=1000, seed=seed)

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
    gd.plot_persistence_diagram(per_dim[2])
    plt.show()
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


# ----------------- Experiment: Simple PD Visualization ----------------- #
# example:
@app.command("simple-pd")
def simple_pd(
    universe_index: Annotated[int, typer.Argument()] = 1,
    pca_dim: Annotated[int, typer.Option()] = 3,
    tda_sample_size: Annotated[int, typer.Option()] = 1000,
    output_dir: Annotated[Path, typer.Option()] = Path("data/experiments/simple_pd"),
):
    logger.info("[TEST] Running PD test for universe_index=%d", universe_index)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get universe
    universe = get_universe(universe_index)

    # 2. Get embedding space for universe
    latent = _get_latent_embeddings_for_universe(universe, retrain_if_missing=True)

    # 3. Prepare point cloud for TDA (subsample, normalize, PCA)
    points_for_tda = from_latent_to_point_cloud(
        latent,
        pca_dim=pca_dim,
        target_size=tda_sample_size,
        seed=universe.seed,
        normalize=True,
    )

    # 4. Build alpha simplex tree
    st = build_alpha_complex_simplex_tree(points_for_tda)
    result_str = (
        "Alpha complex is of dimension "
        + repr(st.dimension())
        + " - "
        + repr(st.num_simplices())
        + " simplices - "
        + repr(st.num_vertices())
        + " vertices."
    )
    logger.info("[PD] " + result_str)

    logger.info(
        f"[PD] Computed persistence with {len(st.persistence_pairs())} intervals"
    )
    logger.info(f"[PD] Betti numbers: {st.betti_numbers()}")

    # 5. Compute persistence
    diag = st.persistence(min_persistence=-1.0)

    # 6. Plot persistence diagram
    ax = gd.plot_persistence_diagram(diag)
    fig = ax.figure
    fig_path = (
        output_dir
        / f"persistence_diagram_univ{universe_index}_pca{pca_dim}_m{tda_sample_size}.png"
    )
    fig.savefig(fig_path, bbox_inches="tight")
    plt.show()

    ax = gd.plot_persistence_barcode(diag, max_intervals=50)
    fig = ax.figure
    fig_path = (
        output_dir
        / f"persistence_barcode_univ{universe_index}_pca{pca_dim}_m{tda_sample_size}.png"
    )
    fig.savefig(fig_path, bbox_inches="tight")
    plt.show()

    ax = gd.plot_persistence_density(diag)
    fig = ax.figure
    fig_path = (
        output_dir
        / f"persistence_density_univ{universe_index}_pca{pca_dim}_m{tda_sample_size}.png"
    )
    fig.savefig(fig_path, bbox_inches="tight")
    plt.show()


@app.command("simple-pd-grid")
def simple_pd_grid(
    universe_index: Annotated[int, typer.Argument()] = 1,
    pca_dims: Annotated[str, typer.Option()] = "2,3,4",
    tda_sample_sizes: Annotated[str, typer.Option()] = "500,1000,2000",
    output_dir: Annotated[Path, typer.Option()] = Path(
        "data/experiments/simple_pd_grid"
    ),
    show: Annotated[bool, typer.Option()] = False,
):
    """
    Create a grid of persistence diagrams for a single universe, varying:
      - PCA dimension (columns)
      - TDA sample size (rows)

    Each cell is a GUDHI persistence diagram for the corresponding (pca_dim, tda_sample_size).
    """
    logger.info(
        "[TEST] Running PD grid for universe_index=%d, pca_dims=%s, tda_sample_sizes=%s",
        universe_index,
        pca_dims,
        tda_sample_sizes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get universe
    universe = get_universe(universe_index)

    # 2. Get embedding space for universe (latent, no subsampling here)
    latent = _get_latent_embeddings_for_universe(universe, retrain_if_missing=True)
    logger.info("[TEST] Latent shape for PD grid: %s", latent.shape)

    # 3. Parse PCA dims and sample sizes
    pca_dim_list = [int(x) for x in pca_dims.split(",") if x.strip()]
    tda_sizes_list = [int(x) for x in tda_sample_sizes.split(",") if x.strip()]

    logger.info("[TEST] PCA dims: %s", pca_dim_list)
    logger.info("[TEST] TDA sample sizes: %s", tda_sizes_list)

    n_rows = len(tda_sizes_list)
    n_cols = len(pca_dim_list)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    # 4. Loop over grid: rows = tda_sample_size, cols = pca_dim
    for i, m in enumerate(tda_sizes_list):
        for j, d in enumerate(pca_dim_list):
            logger.info(
                "[TEST] Computing PD for PCA dim d=%d, TDA sample size m=%d",
                d,
                m,
            )
            # 4a. Prepare point cloud (normalize + PCA + subsample)
            points_for_tda = from_latent_to_point_cloud(
                latent,
                pca_dim=d,
                target_size=m,
                seed=universe.seed,
                normalize=True,
            )

            # 4b. Build alpha simplex tree and compute persistence
            st = build_alpha_complex_simplex_tree(points_for_tda)
            diag = st.persistence()

            # 4c. Plot persistence diagram into the corresponding subplot
            ax = axes[i, j]

            gd.plot_persistence_diagram(diag, axes=ax)
            ax.set_title(f"d={d}, m={m}", fontsize="small")

    fig.suptitle(
        f"Persistence diagrams grid\nUniverse {universe_index}",
        fontsize="medium",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = output_dir / f"pd_grid_univ{universe_index}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    logger.info("[TEST] Saved PD grid to %s", fig_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------- Experiment: Subsample size sweep ----------------- #
# example: uv run exp subsample-sweep 0 --max-rows 50000 --subsample-sizes "500,1000,2000,5000,10000" --homology-dim 1 --pca-dim 3 --num-landscapes 5 --resolution 500 --output-dir data/experiments/subsampling
@app.command("subsample-sweep")
def subsample_sweep(
    universe_index: int = typer.Argument(
        ..., help="Index into generate_multiverse() to pick a universe."
    ),
    max_rows: int = typer.Option(
        50_000, help="Max rows to consider from preprocessed dataset."
    ),
    subsample_sizes: str = typer.Option(
        "500,1000,2000,5000,10000",
        help="Comma-separated list of subsample sizes to test.",
    ),
    homology_dim: int = typer.Option(
        1, help="Homology dimension to analyze (e.g. 0,1,2)."
    ),
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
    universe = get_universe(universe_index)

    # 2. Get latent embeddings from AE
    latent = _get_latent_embeddings_for_universe(universe, retrain_if_missing=True)

    # 3. Prepare point cloud for TDA (subsample, normalize, PCA)
    points_for_tda = from_latent_to_point_cloud(
        latent,
        pca_dim=pca_dim,
        target_size=max_rows,
        seed=universe.seed,
        normalize=True,
    )

    # 5. Parse subsample sizes
    subsample_list = [int(x) for x in subsample_sizes.split(",") if x.strip()]
    subsample_list = sorted(subsample_list)
    logger.info("[EXP] Subsample sizes: %s", subsample_list)

    # 6. Compute landscapes for each subsample
    hom_dims = [homology_dim]
    landscapes_per_m: Dict[int, Dict[int, np.ndarray | None]] = {}
    diagrams_per_m: Dict[int, Dict[int, np.ndarray]] = {}

    for m in subsample_list:
        pts = _subsample_points(points_for_tda, m, seed=universe.seed + m)
        logger.info("[EXP] Computing landscapes for subsample size m=%d", m)

        per_dim, lsc = _compute_persistence_and_landscapes_for_points(
            points=pts,
            homology_dims=hom_dims,
            num_landscapes=num_landscapes,
            resolution=resolution,
        )
        diagrams_per_m[m] = per_dim
        landscapes_per_m[m] = lsc

        intervals = per_dim.get(homology_dim)

        # Single persistence diagram
        _plot_persistence_diagram(
            intervals=intervals,
            title=(
                f"Persistence diagram (H={homology_dim})\n"
                f"Universe {universe_index}, PCA={pca_dim}, m={m}"
            ),
            save_path=output_dir
            / f"pd_univ{universe_index}_pca{pca_dim}_H{homology_dim}_m{m}.png",
        )

        # Single barcode plot
        _plot_multiple_barcodes(
            diagrams={m: intervals},
            title=(
                f"Barcodes (H={homology_dim})\n"
                f"Universe {universe_index}, PCA={pca_dim}, m={m}"
            ),
            save_path=output_dir
            / f"barcodes_univ{universe_index}_pca{pca_dim}_H{homology_dim}_m{m}.png",
            label_prefix="m=",
            max_bars_per_group=100,
        )

        # Single landscape plot for this m
        ls = lsc.get(homology_dim)
        if ls is not None:
            y = ls.reshape(-1)
            x = np.linspace(0, 1, y.shape[0])
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("Landscape sample index (normalized)")
            plt.ylabel("Landscape value")
            plt.title(
                f"Landscape (H={homology_dim})\n"
                f"Universe {universe_index}, PCA={pca_dim}, m={m}"
            )
            fig_path_single_ls = (
                output_dir
                / f"landscape_univ{universe_index}_pca{pca_dim}_H{homology_dim}_m{m}.png"
            )
            plt.savefig(fig_path_single_ls, bbox_inches="tight")
            plt.close()
            logger.info("[EXP] Saved single landscape plot to %s", fig_path_single_ls)

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
    np.save(
        output_dir / f"distances_univ{universe_index}_pca{pca_dim}_H{homology_dim}.npy",
        np.array(distances),
    )

    # 9. Plot distance curve
    fig_path = (
        output_dir
        / f"distance_curve_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png"
    )
    _plot_distance_curve(
        x_values=subsample_list,
        distances=distances,
        xlabel="Subsample size m",
        ylabel=f"L2 distance vs m={base_m}",
        title=(
            f"Landscape stability vs subsample size\n"
            f"Universe {universe_index}, PCA={pca_dim}, H={homology_dim}"
        ),
        save_path=fig_path,
    )

    # 10. Plot a few landscapes overlaid
    chosen_ms = sorted(
        {
            subsample_list[0],
            subsample_list[len(subsample_list) // 2],
            subsample_list[-1],
        }
    )

    landscapes_for_plot_m: Dict[int, np.ndarray | None] = {}
    for m in chosen_ms:
        landscapes_for_plot_m[m] = landscapes_per_m[m].get(homology_dim)

    fig_path = (
        output_dir / f"landscapes_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png"
    )
    _plot_landscape_overlay(
        landscapes=landscapes_for_plot_m,
        title=(
            f"Selected landscapes vs subsample size\n"
            f"Universe {universe_index}, PCA={pca_dim}, H={homology_dim}"
        ),
        save_path=fig_path,
        label_prefix="m=",
    )

    # 11. Plot persistence diagrams for the same chosen subsample sizes
    diagrams_for_plot: Dict[int, np.ndarray | None] = {}
    for m in chosen_ms:
        intervals = diagrams_per_m[m].get(homology_dim)
        diagrams_for_plot[m] = intervals

    _plot_multiple_persistence_diagrams(
        diagrams=diagrams_for_plot,
        homology_dim=homology_dim,
        title=f"Persistence diagrams (H={homology_dim})\nUniverse {universe_index}, PCA={pca_dim}, varying m",
        save_path=output_dir
        / f"pd_combined_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png",
        label_prefix="m=",
    )
    _plot_multiple_barcodes(
        diagrams=diagrams_for_plot,
        title=f"Barcodes (H={homology_dim})\nUniverse {universe_index}, PCA={pca_dim}, varying m",
        save_path=output_dir
        / f"barcodes_univ{universe_index}_pca{pca_dim}_H{homology_dim}.png",
        label_prefix="m=",
        max_bars_per_group=100,
    )


# ----------------- Experiments: PCA dim sweep ----------------- #
# example: uv run exp pca-sweep 0 --max-rows 50000 --pca-dims "2,3,4" --homology-dim 1 --subsample-size 10000 --num-landscapes 5 --resolution 500 --output-dir data/experiments/pca
@app.command("pca-sweep")
def pca_sweep(
    universe_index: int = typer.Argument(
        ..., help="Index into generate_multiverse() to pick a universe."
    ),
    max_rows: int = typer.Option(
        50_000, help="Max rows to consider from preprocessed dataset."
    ),
    pca_dims: str = typer.Option(
        "2,3,4,6,8",
        help="Comma-separated list of PCA dims to test.",
    ),
    homology_dim: int = typer.Option(
        1, help="Homology dimension to analyze (e.g. 0,1,2)."
    ),
    subsample_size: int = typer.Option(
        10_000, help="Subsample size m for all PCA dims."
    ),
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

    # 1. Pick universe
    universe = get_universe(universe_index)

    # 1. Get latent embeddings
    latent = _get_latent_embeddings_for_universe(universe, retrain_if_missing=True)
    if latent.shape[0] > max_rows:
        latent = latent[:max_rows]
        logger.info("[EXP] Truncated latent to first %d rows for experiment.", max_rows)

    logger.info(
        "[EXP] Latent descriptives: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        np.min(latent),
        np.max(latent),
        np.mean(latent),
        np.std(latent),
    )

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
        pts = _subsample_points(
            pts_full, subsample_size, seed=universe.seed + d + subsample_size
        )
        logger.info("[EXP] Computing persistence & landscapes for PCA dim d=%d", d)

        per_dim, lsc = _compute_persistence_and_landscapes_for_points(
            points=pts,
            homology_dims=hom_dims,
            num_landscapes=num_landscapes,
            resolution=resolution,
        )
        diagrams_per_d[d] = per_dim
        landscapes_per_d[d] = lsc

        intervals = per_dim.get(homology_dim)
        _plot_persistence_diagram(
            intervals=intervals,
            title=(
                f"Persistence diagram (H={homology_dim})\n"
                f"Universe {universe_index}, m={subsample_size}, d={d}"
            ),
            save_path=output_dir
            / f"pd_univ{universe_index}_H{homology_dim}_m{subsample_size}_d{d}.png",
        )

        _plot_multiple_barcodes(
            diagrams={d: intervals},
            title=(
                f"Barcodes (H={homology_dim})\n"
                f"Universe {universe_index}, m={subsample_size}, d={d}"
            ),
            save_path=output_dir
            / f"barcodes_univ{universe_index}_H{homology_dim}_m{subsample_size}_d{d}.png",
            label_prefix="d=",
            max_bars_per_group=100,
        )

        # landscape plot for this d
        ls = lsc.get(homology_dim)
        if ls is not None:
            y = ls.reshape(-1)
            x = np.linspace(0, 1, y.shape[0])
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("Landscape sample index (normalized)")
            plt.ylabel("Landscape value")
            plt.title(
                f"Landscape (H={homology_dim})\n"
                f"Universe {universe_index}, m={subsample_size}, d={d}"
            )
            fig_path_single_ls = (
                output_dir
                / f"landscape_univ{universe_index}_H{homology_dim}_m{subsample_size}_d{d}.png"
            )
            plt.savefig(fig_path_single_ls, bbox_inches="tight")
            plt.close()
            logger.info("[EXP] Saved single landscape plot to %s", fig_path_single_ls)

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

    np.save(
        output_dir
        / f"distances_univ{universe_index}_m{subsample_size}_H{homology_dim}.npy",
        np.array(distances),
    )

    # 5. Plot distance curve
    fig_path = (
        output_dir
        / f"distance_curve_univ{universe_index}_m{subsample_size}_H{homology_dim}.png"
    )
    _plot_distance_curve(
        x_values=pca_dim_list,
        distances=distances,
        xlabel="PCA dimension d",
        ylabel=f"L2 distance vs d={base_d}",
        title=(
            f"Landscape stability vs PCA dim\n"
            f"Universe {universe_index}, m={subsample_size}, H={homology_dim}"
        ),
        save_path=fig_path,
    )

    # 6. Plot selected landscapes overlaid
    chosen_ds = sorted(
        {pca_dim_list[0], pca_dim_list[len(pca_dim_list) // 2], pca_dim_list[-1]}
    )
    landscapes_for_plot: Dict[int, np.ndarray | None] = {}
    for d in chosen_ds:
        landscapes_for_plot[d] = landscapes_per_d[d].get(homology_dim)

    fig_path = (
        output_dir
        / f"landscapes_univ{universe_index}_m{subsample_size}_H{homology_dim}.png"
    )
    _plot_landscape_overlay(
        landscapes=landscapes_for_plot,
        title=(
            f"Selected landscapes vs PCA dim\n"
            f"Universe {universe_index}, m={subsample_size}, H={homology_dim}"
        ),
        save_path=fig_path,
        label_prefix="d=",
    )

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
        save_path=output_dir
        / f"pd_combined_univ{universe_index}_H{homology_dim}_m{subsample_size}.png",
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
        save_path=output_dir
        / f"barcodes_univ{universe_index}_H{homology_dim}_m{subsample_size}.png",
        label_prefix="d=",
        max_bars_per_group=100,
    )


if __name__ == "__main__":
    app()
