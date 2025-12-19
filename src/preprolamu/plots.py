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


@app.command("3d")
def plot_3d(
    universe_index: int = typer.Option(
        ..., help="Index of the universe (use your usual indexing)."
    ),
    split: str = typer.Option(
        "test", help="Data split: 'train', 'validation' or 'test'."
    ),
    target_size: Optional[int] = typer.Option(
        None,
        help="Number of points to subsample for plotting. If omitted, uses universe.tda_config.subsample_size.",
    ),
    point_size: float = typer.Option(
        2.0, help="Matplotlib marker size for scatter points."
    ),
    alpha: float = typer.Option(0.7, help="Point alpha for scatter plot."),
    cmap: str = typer.Option(
        "viridis", help="Colormap to color points by sample index."
    ),
    show: bool = typer.Option(
        False, help="If set, show interactive window after saving."
    ),
    out_dir: Path = typer.Option(
        Path("data/figures"), help="Directory to save the output PNG."
    ),
):
    """
    Load an embedding for the given universe index and split, project to 3D (PCA),
    and save a 3D scatter plot PNG.
    """
    # Resolve universe
    universe = get_universe(universe_index)
    uid = getattr(
        universe, "id_string", getattr(universe, "id", universe.to_id_string())
    )
    logger.info("Selected universe: %s (index=%d)", uid, universe_index)

    # Load embedding (will raise FileNotFoundError if not present)
    try:
        projected = load_projected(universe, split)
    except FileNotFoundError as e:
        logger.error(
            "Projected point cloud for universe %s (split=%s) not found. Run embedding creation first. Error: %s",
            uid,
            split,
            e,
        )
        raise typer.Exit(code=2)

    # Determine target_size (subsample)
    if target_size is None:
        target_size = int(universe.tda_config.subsample_size)

    points = downsample_latent(projected, target_size=target_size, seed=universe.seed)

    if points.shape[1] != 3:
        logger.error(
            "Projected point cloud does not have 3 dimensions (got shape %s)",
            points.shape,
        )
        raise typer.Exit(code=4)

    logger.info("Prepared point cloud for plotting with shape %s", points.shape)

    # Prepare colors (optionally color by index)
    n_points = points.shape[0]
    colors = np.linspace(0.0, 1.0, n_points)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    ax.set_title(f"AE embedding (universe={uid}, split={split})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("sample index (normalized)")

    # Save figure
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uid}_embedding_{split}_3d.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Saved 3D scatter to %s", out_path)

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    app()
