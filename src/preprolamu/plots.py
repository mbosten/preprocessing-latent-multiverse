from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from preprolamu.io.storage import load_projected
from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.embeddings import downsample_latent
from preprolamu.pipeline.metrics import build_metrics_table
from preprolamu.pipeline.universes import generate_multiverse, get_universe

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


def _contiguous_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """
    Given a list of labels (ordered), return spans of identical consecutive labels.
    Returns (start_idx, end_idx_exclusive, label).
    """
    if not labels:
        return []
    spans = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            spans.append((start, i, cur))
            start = i
            cur = labels[i]
    spans.append((start, len(labels), cur))
    return spans


def _draw_brackets_top(
    ax, col_tuples, level_names, *, y0=0.0, dy=0.35, fontsize=8, color="black"
):
    """
    Draw hierarchical brackets above a heatmap for MultiIndex-like column tuples.

    col_tuples: list[tuple] length = n_cols
    level_names: e.g. ["dataset_id","scaling","feature_subset"]
    """
    n_levels = len(level_names)
    n_cols = len(col_tuples)

    # Build progressively deeper keys:
    # level 0 key: (dataset,)
    # level 1 key: (dataset, scaling)
    # level 2 key: (dataset, scaling, feature_subset)
    for lvl in range(n_levels):
        keys = [col_tuples[i][: lvl + 1] for i in range(n_cols)]

        # But spans must respect higher-level grouping: use the full key prefix
        spans = []
        start = 0
        cur_key = keys[0]
        for i in range(1, n_cols):
            if keys[i] != cur_key:
                spans.append((start, i, cur_key))
                start = i
                cur_key = keys[i]
        spans.append((start, n_cols, cur_key))

        y = y0 + (n_levels - 1 - lvl) * dy  # top level highest
        tick = dy * 0.25

        for s, e, key in spans:
            # bracket from s..e in heatmap coordinates: columns centered on integers
            x0 = s - 0.5
            x1 = e - 0.5
            ax.plot(
                [x0, x0, x1, x1], [y - tick, y, y, y - tick], linewidth=1.2, color=color
            )
            label = _short_label(key[lvl])
            ax.text(
                (x0 + x1) / 2,
                y + tick * 0.6,
                label,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    ax.axis("off")


def _draw_brackets_right(
    ax, row_tuples, level_names, *, x0=0.0, dx=0.55, fontsize=8, color="black"
):
    """
    Draw hierarchical brackets to the right of a heatmap for MultiIndex-like row tuples.

    row_tuples: list[tuple] length = n_rows
    level_names: e.g. ["log_transform","duplicate_handling","missingness"]
    """
    n_levels = len(level_names)
    n_rows = len(row_tuples)

    for lvl in range(n_levels):
        keys = [row_tuples[i][: lvl + 1] for i in range(n_rows)]

        spans = []
        start = 0
        cur_key = keys[0]
        for i in range(1, n_rows):
            if keys[i] != cur_key:
                spans.append((start, i, cur_key))
                start = i
                cur_key = keys[i]
        spans.append((start, n_rows, cur_key))

        x = x0 + lvl * dx
        tick = dx * 0.14

        for s, e, key in spans:
            # bracket from s..e in heatmap coordinates: rows centered on integers
            y0 = s - 0.5
            y1 = e - 0.5
            ax.plot(
                [x, x + tick, x + tick, x], [y0, y0, y1, y1], linewidth=1.2, color=color
            )
            label = _short_label(key[lvl])
            ax.text(
                x + tick + 0.07,
                (y0 + y1) / 2,
                label,
                ha="left",
                va="center",
                fontsize=fontsize,
            )

    ax.axis("off")


def _format_sci(x: float) -> str:
    if pd.isna(x):
        return ""
    # 3 significant digits in scientific notation
    return f"{x:.2e}"


def _best_contrast_text_color(norm_value: float) -> str:
    """
    Choose white text on dark backgrounds, black on light backgrounds.
    norm_value should be in [0,1] after colormap normalization.
    """
    if not np.isfinite(norm_value):
        return "black"
    return "white" if norm_value < 0.45 else "black"


def _build_norm_grid(
    df: pd.DataFrame,
    *,
    homology_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - grid: pivot table with rows=(log_transform, duplicate_handling, missingness)
              cols=(dataset_id, scaling, feature_subset)
              values=mean l2_dim{homology_dim} across seeds
      - grouped_df: the grouped/averaged table (useful for debugging)
    """
    dcol = f"l2_dim{homology_dim}"
    if dcol not in df.columns:
        raise ValueError(f"Missing column {dcol!r} in metrics table.")

    # Only OK metrics
    df = df[df["metrics_status"] == "ok"].copy()

    group_cols = [
        "dataset_id",
        "scaling",
        "feature_subset",
        "log_transform",
        "duplicate_handling",
        "missingness",
    ]

    grouped = (
        df.groupby(group_cols, dropna=False)[dcol]
        .mean()
        .reset_index()
        .rename(columns={dcol: "l2_mean_across_seeds"})
    )

    grid = grouped.pivot_table(
        index=["missingness", "log_transform", "duplicate_handling"],
        columns=["dataset_id", "scaling", "feature_subset"],
        values="l2_mean_across_seeds",
        aggfunc="mean",
    )

    # Ensure deterministic ordering
    grid = grid.sort_index(axis=1)  # Default column order
    grid = _sort_multiindex_rows(grid)  # Custom order of rows

    return grid, grouped


def _sort_multiindex_rows(grid: pd.DataFrame) -> pd.DataFrame:
    # Desired orders (adjust if you have other values)
    miss_order = ["impute_median", "drop_rows"]
    log_order = ["log1p", "none"]
    dup_order = ["drop", "keep"]

    idx = grid.index.to_frame(index=False)

    # Make ordered categoricals (unknown values go last)
    idx["missingness"] = pd.Categorical(
        idx["missingness"], categories=miss_order, ordered=True
    )
    idx["log_transform"] = pd.Categorical(
        idx["log_transform"], categories=log_order, ordered=True
    )
    idx["duplicate_handling"] = pd.Categorical(
        idx["duplicate_handling"], categories=dup_order, ordered=True
    )

    idx_sorted = idx.sort_values(["missingness", "log_transform", "duplicate_handling"])
    new_index = pd.MultiIndex.from_frame(idx_sorted)

    return grid.reindex(new_index)


def _plot_multiverse_grid(
    grid: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    show_values: bool = True,
    color_log10: bool = True,
):
    """
    Heatmap + bracket "trees" (nested groupings) like Steegen Fig.2 style.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Data for coloring
    values = grid.to_numpy(dtype=float)
    color_data = values.copy()

    if color_log10:
        # log10 compresses huge outliers while still showing structure.
        # add epsilon to avoid log(0)
        eps = 1e-12
        color_data = np.log10(np.maximum(color_data, eps))

    # Layout: top brackets, main heatmap, right brackets
    n_rows, n_cols = grid.shape
    fig = plt.figure(figsize=(max(16, n_cols * 0.95), max(7, n_rows * 0.55)))

    # axes positions: [left, bottom, width, height]
    ax_heat = fig.add_axes([0.06, 0.10, 0.78, 0.65])
    ax_top = fig.add_axes([0.06, 0.75, 0.78, 0.18], sharex=ax_heat)
    ax_right = fig.add_axes([0.85, 0.10, 0.12, 0.65], sharey=ax_heat)

    im = ax_heat.imshow(color_data, aspect="auto")

    norm = im.norm

    # lock heatmap coordinates so every axis uses the same [col,row] geometry
    ax_heat.set_xlim(-0.5, n_cols - 0.5)
    ax_heat.set_ylim(n_rows - 0.5, -0.5)  # invert rows to match imshow default
    fig.suptitle(title, y=0.98, fontsize=14)

    # Remove left and bottom tick labels
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xticklabels([])
    ax_heat.set_yticklabels([])
    ax_heat.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Cell text values
    if show_values:
        for r in range(n_rows):
            for c in range(n_cols):
                txt = _format_sci(values[r, c])
                if txt:
                    nv = norm(color_data[r, c])
                    txt_color = _best_contrast_text_color(nv)
                    ax_heat.text(
                        c, r, txt, ha="center", va="center", fontsize=6, color=txt_color
                    )

    # Colorbar
    cax = fig.add_axes([0.08, 0.05, 0.65, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("log10(L2 norm)" if color_log10 else "L2 norm", fontsize=9)

    # Brackets
    _draw_brackets_top(
        ax_top,
        col_tuples=grid.columns.to_list(),
        level_names=["dataset_id", "scaling", "feature_subset"],
        fontsize=8,
        color="0.25",
    )
    _draw_brackets_right(
        ax_right,
        row_tuples=grid.index.to_list(),
        level_names=["missingness", "log_transform", "duplicate_handling"],
        fontsize=8,
        color="0.25",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


_LABEL_MAP = {
    "impute_median": "Impute",
    "drop_rows": "Drop_rows",
    "without_confounders": "No_conf",
    "all": "All",
    "log1p": "Log1p",
    "none": "None",
    "minmax": "MinMax",
    "zscore": "Z-score",
    "quantile": "Quantile",
    "drop": "Drop",
    "keep": "Keep",
}


def _short_label(x) -> str:
    s = str(x)
    return _LABEL_MAP.get(s, s)


@app.command("multiverse-norm-grid")
def multiverse_norm_grid(
    split: str = typer.Option("test", help="train/val/test"),
    dims: str = typer.Option(
        "0,1,2", help="Comma-separated homology dims, e.g. '0,1,2'"
    ),
    out_dir: Path = typer.Option(Path("data/figures"), help="Output directory"),
    show_values: bool = typer.Option(True, help="Draw numeric values in each cell"),
    color_log10: bool = typer.Option(
        True, help="Color by log10(norm) to handle outliers"
    ),
):
    """
    Steegen-style multiverse grid, but with cell values = mean landscape L2 norms
    averaged across seeds (42,420,4200,42000).

    Columns: dataset_id / scaling / feature_subset
    Rows:    log_transform / duplicate_handling / missingness
    """
    dims_list = [int(x.strip()) for x in dims.split(",") if x.strip()]

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)

    for d in dims_list:
        grid, _ = _build_norm_grid(df, homology_dim=d)
        out_path = out_dir / f"multiverse_norm_grid_split-{split}_dim{d}.png"
        _plot_multiverse_grid(
            grid,
            title=f"Multiverse grid: mean landscape L2 norms (split={split}, homology dim={d})",
            out_path=out_path,
            show_values=show_values,
            color_log10=color_log10,
        )
        logger.info("Saved multiverse norm grid to %s", out_path)


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
