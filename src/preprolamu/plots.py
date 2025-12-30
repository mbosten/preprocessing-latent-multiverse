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


# ------- Multiverse index plot helpers ------- #
def _build_index_grid(
    df: pd.DataFrame,
    *,
    universes: list,
) -> pd.DataFrame:
    """
    Build a grid with rows=(missingness, log_transform, duplicate_handling),
    cols=(dataset_id, scaling, feature_subset),
    values = list of universe indices in that cell (typically 4: one per seed).
    """
    if "universe_id" not in df.columns:
        raise ValueError("Metrics table must include 'universe_id'.")

    id_to_index = {u.id: i for i, u in enumerate(universes)}
    df = df.copy()
    df["universe_index"] = df["universe_id"].map(id_to_index)

    df = df[df["metrics_status"] == "ok"].copy()

    group_cols = [
        "dataset_id",
        "scaling",
        "feature_subset",
        "missingness",
        "log_transform",
        "duplicate_handling",
    ]

    grouped = (
        df.groupby(group_cols, dropna=False)["universe_index"]
        .apply(lambda s: sorted(int(x) for x in s.dropna().tolist()))
        .reset_index()
        .rename(columns={"universe_index": "indices"})
    )

    grid = grouped.pivot_table(
        index=["missingness", "log_transform", "duplicate_handling"],
        columns=["dataset_id", "scaling", "feature_subset"],
        values="indices",
        aggfunc="first",  # already unique per cell
    )

    grid = grid.sort_index(axis=1)
    grid = _sort_multiindex_rows(grid)

    # Optional: move fully-empty rows to bottom (same as your norm grid)
    empty_mask = grid.isna().all(axis=1)
    grid = pd.concat([grid.loc[~empty_mask], grid.loc[empty_mask]], axis=0)

    return grid


def _format_index_cell(idxs, mode: str) -> str:
    if idxs is None or (isinstance(idxs, float) and np.isnan(idxs)):
        return ""
    if not isinstance(idxs, (list, tuple)) or len(idxs) == 0:
        return ""
    idxs = [int(x) for x in idxs]

    if mode == "range":
        if len(idxs) == 1:
            return str(idxs[0])
        return f"{min(idxs)}–{max(idxs)}"

    if mode == "list":
        # Put on two lines to avoid excessive width
        if len(idxs) <= 4:
            return "\n".join(str(x) for x in idxs)
        return "\n".join(str(x) for x in idxs[:4]) + "\n..."

    if mode == "both":
        rng = f"{min(idxs)}–{max(idxs)} (n={len(idxs)})"
        lst = ", ".join(str(x) for x in idxs[:6]) + ("…" if len(idxs) > 6 else "")
        return f"{rng}\n{lst}"

    raise ValueError("mode must be 'range', 'list', or 'both'")


def _plot_multiverse_index_grid(
    grid: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    mode: str = "range",
    fontsize: int = 9,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = grid.shape

    # blank background heatmap just to get the grid layout
    fig = plt.figure(figsize=(max(16, n_cols * 0.95), max(7, n_rows * 0.55)))

    ax_heat = fig.add_axes([0.06, 0.10, 0.78, 0.65])
    ax_top = fig.add_axes([0.06, 0.75, 0.78, 0.18], sharex=ax_heat)
    ax_right = fig.add_axes([0.85, 0.10, 0.12, 0.65], sharey=ax_heat)

    fig.suptitle(title, y=0.995, fontsize=14)

    ax_heat.set_xlim(-0.5, n_cols - 0.5)
    ax_heat.set_ylim(n_rows - 0.5, -0.5)

    # draw cell borders (helps readability in print)
    ax_heat.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_heat.grid(which="minor", linewidth=0.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # cell text
    for r in range(n_rows):
        for c in range(n_cols):
            txt = _format_index_cell(grid.iat[r, c], mode=mode)
            if txt:
                ax_heat.text(c, r, txt, ha="center", va="center", fontsize=fontsize)

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


# ---------- 3D Scatter plot helpers ---------- #
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


# ---------- Norm table plot helpers ---------- #
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
    # 2 significant digits in scientific notation
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
              values=median l2_dim{homology_dim} across seeds
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
        .median()
        .reset_index()
        .rename(columns={dcol: "l2_median_across_seeds"})
    )

    grid = grouped.pivot_table(
        index=["missingness", "log_transform", "duplicate_handling"],
        columns=["dataset_id", "scaling", "feature_subset"],
        values="l2_median_across_seeds",
        aggfunc="median",
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
        eps = 1e-30
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


# ---------- Norm histogram plot helpers ---------- #
def _parse_split_by(spec: str) -> list[str]:
    """
    Parse --split-by "dataset,scaling" into list of 1-2 normalized keys.
    Allowed keys map to columns in the metrics table.
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    keys = [k.strip().lower() for k in spec.split(",") if k.strip()]
    if len(keys) > 2:
        raise typer.BadParameter(
            "--split-by supports at most 2 keys (e.g., 'dataset' or 'dataset,scaling')."
        )

    alias = {
        "dataset": "dataset_id",
        "dataset_id": "dataset_id",
        "scaling": "scaling",
        "feature_subset": "feature_subset",
        "features": "feature_subset",
        "log_transform": "log_transform",
        "log": "log_transform",
        "duplicate_handling": "duplicate_handling",
        "duplicates": "duplicate_handling",
        "missingness": "missingness",
    }
    out = []
    for k in keys:
        if k not in alias:
            raise typer.BadParameter(
                f"Unknown split key {k!r}. Allowed: dataset, scaling, feature_subset, log_transform, duplicate_handling, missingness."
            )
        out.append(alias[k])
    return out


def _group_title(keys: list[str], values: tuple) -> str:
    if not keys:
        return "ALL"
    parts = []
    for k, v in zip(keys, values):
        parts.append(f"{_short_label(k)}={_short_label(v)}")
    return ",\n".join(parts)


# ---------- Outlier plot helpers ---------- #
def _parse_index_list(spec: str) -> set[int]:
    """
    Parse a comma-separated list of integers and/or ranges like:
      "1,2,5-10,42"
    Returns a set of indices.
    """
    spec = (spec or "").strip()
    if not spec:
        return set()

    out: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = [t.strip() for t in p.split("-", 1)]
            if not a or not b:
                raise typer.BadParameter(f"Invalid range in --exclude: {p!r}")
            try:
                lo = int(a)
                hi = int(b)
            except ValueError as e:
                raise typer.BadParameter(f"Non-integer in --exclude: {p!r}") from e
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(p))
            except ValueError as e:
                raise typer.BadParameter(f"Non-integer in --exclude: {p!r}") from e
    return out


def _aggregate_across_seeds(
    df: pd.DataFrame,
    *,
    homology_dim: int,
    agg: str = "median",
) -> pd.DataFrame:
    """
    Collapse the 4 seeds into one value per multiverse config (excluding seed).

    Returns a df with one row per configuration, including dataset_id and agg_norm.
    """
    dcol = f"l2_dim{homology_dim}"
    if dcol not in df.columns:
        raise ValueError(f"Missing column {dcol!r} in metrics table.")

    df = df[df["metrics_status"] == "ok"].copy()

    group_cols = [
        "dataset_id",
        "scaling",
        "feature_subset",
        "log_transform",
        "duplicate_handling",
        "missingness",
    ]

    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")

    if agg == "median":
        out = df.groupby(group_cols, dropna=False)[dcol].median().reset_index()
    else:
        out = df.groupby(group_cols, dropna=False)[dcol].mean().reset_index()

    out = out.rename(columns={dcol: "agg_norm"})
    return out


def _iqr_bounds(x: np.ndarray, k: float = 1.5) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def _mad_bounds(x: np.ndarray, k: float = 3.5) -> tuple[float, float]:
    """
    Median ± k * MAD (MAD = median(|x - median(x)|)).
    k=3.5 is a common robust outlier threshold.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        # If MAD=0, everything equals the median OR there are many identical values.
        # Fall back to "no bounds" except exact equality.
        return (med, med)
    return (med - k * mad, med + k * mad)


def _compute_outlier_mask(
    x: np.ndarray,
    method: str,
    k: float,
) -> tuple[np.ndarray, float, float]:
    """
    Returns mask (True=outlier), lower_bound, upper_bound
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)

    if method == "iqr":
        lo, hi = _iqr_bounds(x[finite], k=k)
    elif method == "mad":
        lo, hi = _mad_bounds(x[finite], k=k)
    else:
        raise ValueError("method must be 'iqr' or 'mad'")

    # If bounds collapse (e.g., MAD=0), treat values different from the bound as outliers
    if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
        mask = finite & (x != lo)
    else:
        mask = finite & ((x < lo) | (x > hi))
    return mask, lo, hi


def _maybe_log_transform(x: np.ndarray, yscale: str) -> tuple[np.ndarray, str]:
    """
    For visualization only. Keeps raw values in outlier computations.
    """
    if yscale == "linear":
        return x, "norm"
    if yscale == "log10":
        eps = 1e-30  # allows log10(0) -> very negative
        return np.log10(np.maximum(x, eps)), "log10(norm)"
    if yscale == "symlog":
        # useful if you ever get negatives; shouldn't happen for norms
        return x, "norm (symlog)"
    raise ValueError("yscale must be 'linear', 'log10', or 'symlog'")


def _get_universe_level_norms(df: pd.DataFrame, homology_dim: int) -> pd.DataFrame:
    dcol = f"l2_dim{homology_dim}"
    if dcol not in df.columns:
        raise ValueError(f"Missing column {dcol!r} in metrics table.")
    df = df[df["metrics_status"] == "ok"].copy()
    return df[["dataset_id", dcol]].rename(columns={dcol: "norm"})


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
    Steegen-style multiverse grid, but with cell values = median landscape L2 norms
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
            title=f"Multiverse grid: median landscape L2 norms (split={split}, homology dim={d})",
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


@app.command("norm-outlier-overview")
def norm_outlier_overview_boxplot_only(
    split: str = typer.Option("test", help="train/val/test"),
    homology_dim: int = typer.Option(1, help="Homology dimension: 0, 1, or 2"),
    whis: float = typer.Option(
        1.5, help="Matplotlib whisker multiplier (default outlier rule)."
    ),
    max_labels: int = typer.Option(
        30, help="Max outlier labels per panel (avoid clutter)."
    ),
    exclude: str = typer.Option(
        "",
        help="Universe indices to exclude, e.g. '56,57,60' or '48-63,120'.",
    ),
    out_dir: Path = typer.Option(Path("data/figures"), help="Output directory"),
):
    """
    2x2 grid: ALL datasets + each dataset separately (first 3 datasets alphabetically).
    Plot ONLY matplotlib's default boxplot; only fliers are shown as points.
    Fliers are labeled with universe_index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)

    dcol = f"l2_dim{homology_dim}"
    if dcol not in df.columns:
        raise typer.BadParameter(f"Missing column {dcol!r} in metrics table.")
    df = df[df["metrics_status"] == "ok"].copy()

    if "universe_id" not in df.columns:
        raise typer.BadParameter("Metrics table missing 'universe_id' column.")

    id_to_index = {u.id: i for i, u in enumerate(universes)}
    df["universe_index"] = df["universe_id"].map(id_to_index)

    u_df = df[["dataset_id", "universe_index", dcol]].rename(columns={dcol: "norm"})
    u_df = u_df[np.isfinite(u_df["norm"].to_numpy(dtype=float))].copy()

    # Exclude indices if requested
    exclude_set = _parse_index_list(exclude)
    if exclude_set:
        before = len(u_df)
        u_df = u_df[~u_df["universe_index"].isin(exclude_set)].copy()
        logger.info(
            "Excluded %d universes by index (rows: %d -> %d)",
            len(exclude_set),
            before,
            len(u_df),
        )

    if u_df.empty:
        raise typer.BadParameter(
            "No universes remain to plot (after filtering/exclusion)."
        )

    datasets = sorted(u_df["dataset_id"].dropna().unique().tolist())
    per_ds = datasets[:3]
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", u_df)]
    scopes += [(ds, u_df[u_df["dataset_id"] == ds].copy()) for ds in per_ds]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_list = axes.ravel()

    fig.suptitle(
        f"Matplotlib boxplot fliers (split={split}, dim={homology_dim}, whis={whis})",
        y=0.98,
        fontsize=12,
    )

    for ax, (scope_name, sdf) in zip(axes_list, scopes):
        # reset_index so position p aligns with arrays (important for labeling)
        sdf = sdf.reset_index(drop=True)
        x = sdf["norm"].to_numpy(dtype=float)
        uidx = sdf["universe_index"].to_numpy(dtype=int)

        # Boxplot ONLY (no scatter). Matplotlib draws fliers as points.
        bp = ax.boxplot(
            x,
            positions=[1.0],
            widths=0.45,
            vert=True,
            showfliers=True,
            whis=whis,
        )

        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([])
        ax.set_title(scope_name)
        ax.set_ylabel("L2 norm (linear)")

        # Extract fliers from matplotlib and label them
        flier_line = bp["fliers"][0]  # only one box
        flier_x = np.asarray(flier_line.get_xdata(), dtype=float)
        flier_y = np.asarray(flier_line.get_ydata(), dtype=float)

        # --- add slight horizontal jitter to fliers only ---
        if flier_x.size > 0:
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.12, 0.12, size=flier_x.size)
            flier_x_jittered = flier_x + jitter
            flier_line.set_xdata(flier_x_jittered)
        else:
            flier_x_jittered = flier_x

        n_fliers = int(flier_y.size)

        # Annotate up to max_labels most extreme fliers (by distance to median)
        if n_fliers > 0:
            med = float(np.median(x))
            # indices of flier points sorted by extremeness
            order = np.argsort(np.abs(flier_y - med))[::-1][:max_labels]

            # Map each flier y-value back to a universe index.
            # Use a tolerance for float comparison.
            tol = max(1e-12, 1e-9 * float(np.nanmax(np.abs(x))) if len(x) else 1e-12)

            for j in order:
                y = flier_y[j]

                # find matching original observations (can be multiple if ties)
                matches = np.where(np.isclose(x, y, rtol=0.0, atol=tol))[0]
                if matches.size == 0:
                    continue

                # label the first match (avoids clutter when duplicates exist)
                p = int(matches[0])
                ax.text(
                    flier_x_jittered[j] + 0.02,
                    y,
                    str(int(uidx[p])),
                    fontsize=7,
                    ha="left",
                    va="center",
                )

        ax.text(
            0.02,
            0.98,
            f"n={len(x)}\nfliers={n_fliers}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    # Turn off any unused panels
    for j in range(len(scopes), 4):
        axes_list[j].axis("off")

    out_path = (
        out_dir
        / f"norm_outlier_overview_boxplot_only_split-{split}_dim{homology_dim}_whis{whis}_exclude{exclude}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved boxplot-only outlier figure to %s", out_path)


@app.command("norm-distribution")
def norm_distribution(
    split: str = typer.Option("test", help="train/val/test"),
    homology_dim: int = typer.Option(1, help="Homology dimension: 0, 1, or 2"),
    split_by: str = typer.Option(
        "",
        help="Comma-separated grouping keys (max 2). Examples: 'dataset' or 'dataset,scaling'.",
    ),
    max_norm: float | None = typer.Option(
        None,
        help="If set, exclude universes with norm > max_norm before plotting (e.g. 100).",
    ),
    bins: int = typer.Option(30, help="Number of histogram bins."),
    out_dir: Path = typer.Option(Path("data/figures"), help="Output directory"),
):
    """
    Plot distributions of landscape L2 norms as histograms.
    Uses default matplotlib histogram style.
    Creates one subplot per group defined by --split-by (up to 2 keys).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)

    dcol = f"l2_dim{homology_dim}"
    if dcol not in df.columns:
        raise typer.BadParameter(f"Missing column {dcol!r} in metrics table.")

    # Only OK metrics and finite values
    df = df[df["metrics_status"] == "ok"].copy()
    df = df[np.isfinite(df[dcol].to_numpy(dtype=float))].copy()

    if max_norm is not None:
        before = len(df)
        df = df[df[dcol] <= max_norm].copy()
        logger.info(
            "Applied max_norm=%s filter (dropped %d rows)", max_norm, before - len(df)
        )

        if df.empty:
            raise typer.BadParameter(
                "After applying --max-norm, no data remains to plot."
            )

    keys = _parse_split_by(split_by)

    # Ensure columns exist if requested
    for k in keys:
        if k not in df.columns:
            raise typer.BadParameter(
                f"--split-by key {k!r} not present in metrics table columns."
            )

    # Decide grouping
    if not keys:
        groups = [(("ALL",), df)]
        nrows, ncols = 1, 1
    elif len(keys) == 1:
        k1 = keys[0]
        levels = sorted(df[k1].dropna().unique().tolist(), key=lambda x: str(x))
        groups = [((v,), df[df[k1] == v].copy()) for v in levels]
        ncols = min(3, len(groups))
        nrows = int(np.ceil(len(groups) / ncols)) if groups else 1
    else:
        k1, k2 = keys
        lv1 = sorted(df[k1].dropna().unique().tolist(), key=lambda x: str(x))
        lv2 = sorted(df[k2].dropna().unique().tolist(), key=lambda x: str(x))

        # 2D grid: rows = k1, cols = k2 (stable, easy to read)
        nrows, ncols = len(lv1), len(lv2)
        groups = []
        for v1 in lv1:
            for v2 in lv2:
                sdf = df[(df[k1] == v1) & (df[k2] == v2)].copy()
                groups.append(((v1, v2), sdf))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows))
    # Normalize axes to 2D array for uniform indexing
    if nrows == 1 and ncols == 1:
        axes_arr = np.array([[axes]])
    elif nrows == 1:
        axes_arr = np.array([axes])
    elif ncols == 1:
        axes_arr = np.array([[ax] for ax in axes])
    else:
        axes_arr = axes

    title_keys = ", ".join(keys) if keys else "none"
    fig.suptitle(
        f"Norm distributions (split={split}, dim={homology_dim}, split_by={title_keys})",
        y=0.98,
        fontsize=12,
    )
    fig.subplots_adjust(top=0.88, hspace=0.45)

    # Plot
    if len(keys) <= 1:
        # Fill row-major
        for i, (vals, sdf) in enumerate(groups):
            r = i // ncols
            c = i % ncols
            ax = axes_arr[r, c]
            x = sdf[dcol].to_numpy(dtype=float)
            ax.hist(x, bins=bins)  # default matplotlib histogram
            ax.set_title(_group_title(keys, vals))
            ax.set_xlabel("L2 norm")
            ax.set_ylabel("count")

        # Turn off unused axes
        total = nrows * ncols
        for j in range(len(groups), total):
            r = j // ncols
            c = j % ncols
            axes_arr[r, c].axis("off")
    else:
        # 2D grid: rows = lv1, cols = lv2
        k1, k2 = keys
        lv1 = sorted(df[k1].dropna().unique().tolist(), key=lambda x: str(x))
        lv2 = sorted(df[k2].dropna().unique().tolist(), key=lambda x: str(x))

        for ri, v1 in enumerate(lv1):
            for ci, v2 in enumerate(lv2):
                ax = axes_arr[ri, ci]
                sdf = df[(df[k1] == v1) & (df[k2] == v2)]
                x = sdf[dcol].to_numpy(dtype=float)
                ax.hist(x, bins=bins)
                ax.set_title(_group_title(keys, (v1, v2)))

                # Keep labels minimal to reduce clutter
                if ri == nrows - 1:
                    ax.set_xlabel("L2 norm")
                if ci == 0:
                    ax.set_ylabel("count")

                # If empty, make it visually clear but unobtrusive
                if x.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "empty",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

    out_path = (
        out_dir
        / f"norm_distribution_split-{split}_dim{homology_dim}_splitby-{(split_by or 'none').replace(',', '-')}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved norm distribution figure to %s", out_path)


if __name__ == "__main__":
    app()
