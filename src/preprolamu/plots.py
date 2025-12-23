from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer

from preprolamu.io.storage import load_projected
from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.embeddings import downsample_latent
from preprolamu.pipeline.universes import get_universe

# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed to enable 3D projection)


logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


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


def _scatter_3d(
    ax, pts: np.ndarray, title: str, cmap: str, point_size: float, alpha: float
):
    n = pts.shape[0]
    colors = np.linspace(0.0, 1.0, n)
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("PC 1", labelpad=8)
    ax.set_ylabel("PC 2", labelpad=16)
    ax.set_zlabel("PC 3", labelpad=13)
    return sc


def _universe_meta_text(universe, split: str, n_points: int) -> str:
    # Prefer explicit parameters (more readable than the full uid string)
    lines = [
        f"dataset_id={universe.dataset_id}",
        f"scaling={universe.scaling.value}",
        f"log_transform={universe.log_transform.value}",
        f"feature_subset={universe.feature_subset.value}",
        f"duplicate_handling={universe.duplicate_handling.value}",
        f"missingness={universe.missingness.value}",
        f"seed={universe.seed}",
        f"split={split}",
        f"n={n_points}",
    ]
    # Multi-line keeps it readable and prevents overlap
    return "\n".join(lines)


@app.command("3d")
def plot_3d(
    universe_index: int = typer.Option(..., help="Index of the universe."),
    split: str = typer.Option(
        "test", help="Data split: 'train', 'validation' or 'test'."
    ),
    target_size: Optional[int] = typer.Option(
        None,
        help="Number of points to subsample for plotting. If omitted, uses universe.tda_config.subsample_size.",
    ),
    point_size: float = typer.Option(2.0, help="Marker size."),
    alpha: float = typer.Option(0.7, help="Point alpha."),
    cmap: str = typer.Option("viridis", help="Colormap for sample index coloring."),
    show: bool = typer.Option(False, help="Show interactive window after saving."),
    out_dir: Path = typer.Option(
        Path("data/figures"), help="Directory to save the PNG."
    ),
):
    """
    Load RAW and NORMALIZED projected point clouds (both 3D PCA),
    downsample to target_size, and plot side-by-side.
    """
    universe = get_universe(universe_index)
    uid = getattr(universe, "id", universe.to_id_string())
    logger.info("Selected universe: %s (index=%d)", uid, universe_index)

    if target_size is None:
        target_size = int(universe.tda_config.subsample_size)

    # Load both projections
    try:
        projected_raw = load_projected(universe, split=split, normalized=False)
        projected_norm = load_projected(universe, split=split, normalized=True)
    except FileNotFoundError as e:
        logger.error("Projected file missing for %s (split=%s): %s", uid, split, e)
        raise typer.Exit(code=2)

    logger.info("Loaded RAW projected shape: %s", projected_raw.shape)
    logger.info("Loaded NORM projected shape: %s", projected_norm.shape)

    if projected_raw.shape[1] != 3 or projected_norm.shape[1] != 3:
        logger.error(
            "Expected both projections to be 3D. Got raw=%s norm=%s",
            projected_raw.shape,
            projected_norm.shape,
        )
        raise typer.Exit(code=4)

    # Downsample both with the same seed/size (so plots are comparable)
    pts_raw = downsample_latent(
        projected_raw, target_size=target_size, seed=universe.seed
    )
    pts_norm = downsample_latent(
        projected_norm, target_size=target_size, seed=universe.seed
    )

    # Plot side-by-side
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    _ = _scatter_3d(
        ax1,
        pts_raw,
        title="RAW PCA (no diameter norm)",
        cmap=cmap,
        point_size=point_size,
        alpha=alpha,
    )
    sc2 = _scatter_3d(
        ax2,
        pts_norm,
        title="NORMALIZED PCA (diameter norm)",
        cmap=cmap,
        point_size=point_size,
        alpha=alpha,
    )

    meta = _universe_meta_text(universe, split=split, n_points=pts_norm.shape[0])
    fig.suptitle(
        meta, fontsize=9, x=0.02, horizontalalignment="left", verticalalignment="top"
    )

    # One colorbar for both (same mapping)
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.65, pad=0.06)
    cbar.set_label("sample index (normalized)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uid}_projected_raw_vs_norm_{split}_n{target_size}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Saved side-by-side 3D scatter to %s", out_path)

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    app()
