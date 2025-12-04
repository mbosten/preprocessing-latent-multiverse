# src/alphacomplexbenchmarking/visualization.py
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from typing import Dict

logger = logging.getLogger(__name__)

def _plot_persistence_diagram(
    intervals: np.ndarray | None,
    title: str,
    save_path: Path,
) -> None:
    """
    Simple persistence diagram plot: birth vs death, with diagonal.
    intervals: array of shape (n_intervals, 2) or None.
    """
    if intervals is None or len(intervals) == 0:
        logger.warning("[EXP] No intervals to plot for %s", title)
        return

    births = intervals[:, 0]
    deaths = intervals[:, 1]

    plt.figure()
    plt.scatter(births, deaths, s=10)
    mn = min(births.min(), deaths.min())
    mx = max(births.max(), deaths.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")  # diagonal
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved persistence diagram to %s", save_path)

def _plot_multiple_persistence_diagrams(
    diagrams: Dict[int, np.ndarray | None],
    homology_dim: int,
    title: str,
    save_path: Path,
    label_prefix: str = "",
) -> None:
    """
    Plot multiple persistence diagrams (same homology dim) in a single figure,
    color-coded by key (e.g. subsample size or PCA dimension).

    diagrams: dict key -> intervals array (n_intervals, 2) or None.
              keys are typically subsample sizes (m) or PCA dims (d).
    """
    # Filter out empty/None diagrams
    filtered = {k: v for k, v in diagrams.items() if v is not None and len(v) > 0}
    if not filtered:
        logger.warning("[EXP] No intervals to plot for multiple PDs: %s", title)
        return

    # Prepare color map
    keys = sorted(filtered.keys())
    cmap = get_cmap("viridis")
    colors = {k: cmap(i / max(len(keys) - 1, 1)) for i, k in enumerate(keys)}

    plt.figure()
    all_vals = []

    for k in keys:
        intervals = filtered[k]
        births = intervals[:, 0]
        deaths = intervals[:, 1]
        all_vals.extend(births.tolist())
        all_vals.extend(deaths.tolist())
        label = f"{label_prefix}{k}"
        plt.scatter(births, deaths, s=8, alpha=0.7, color=colors[k], label=label)

    mn = min(all_vals)
    mx = max(all_vals)
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray")  # diagonal

    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(title=label_prefix.rstrip())
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved combined persistence diagram to %s", save_path)

def _plot_multiple_barcodes(
    diagrams: Dict[int, np.ndarray | None],
    title: str,
    save_path: Path,
    label_prefix: str = "",
    max_bars_per_group: int | None = 100,
) -> None:
    """
    Plot multiple barcodes (birth-death intervals) in one figure, grouped and
    color-coded by key (e.g. subsample size or PCA dim).

    diagrams: dict key -> intervals array (n_intervals, 2) or None.
              keys are typically subsample sizes (m) or PCA dims (d).
    max_bars_per_group: optional cap on how many intervals per group to plot
                        (for visualization sanity).
    """
    # Filter out empty/None diagrams
    filtered = {
        k: v for k, v in diagrams.items()
        if v is not None and len(v) > 0
    }
    if not filtered:
        logger.warning("[EXP] No intervals to plot for multiple barcodes: %s", title)
        return

    keys = sorted(filtered.keys())
    cmap = get_cmap("viridis")
    colors = {k: cmap(i / max(len(keys) - 1, 1)) for i, k in enumerate(keys)}

    plt.figure()
    ax = plt.gca()

    y_offset = 0
    y_step = 1.0  # vertical spacing between bars
    legend_handles = []
    legend_labels = []

    for k in keys:
        intervals = filtered[k]
        n_intervals = intervals.shape[0]
        if max_bars_per_group is not None and n_intervals > max_bars_per_group:
            n_plot = max_bars_per_group
        else:
            n_plot = n_intervals

        # Sample a subset if too many intervals
        if n_plot < n_intervals:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_intervals, size=n_plot, replace=False)
            intervals_plot = intervals[idx]
        else:
            intervals_plot = intervals

        color = colors[k]
        for i, (b, d) in enumerate(intervals_plot):
            y = y_offset + i * y_step
            ax.hlines(y, b, d, color=color, linewidth=1.0)

        # Build legend entry: one dummy line per group
        line = plt.Line2D([0], [0], color=color)
        legend_handles.append(line)
        label = f"{label_prefix}{k}"
        legend_labels.append(label)

        # Add extra gap between groups
        y_offset += (n_plot + 5) * y_step

    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Interval index (stacked by group)")
    ax.set_title(title)
    ax.legend(legend_handles, legend_labels, title=label_prefix.rstrip(), fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("[EXP] Saved combined barcode plot to %s", save_path)
