from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from preprolamu.example_plots import plot_example_figures
from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.metrics import (
    build_metrics_table,
    compute_presto_variance_from_metrics_table,
)
from preprolamu.pipeline.universes import generate_multiverse
from preprolamu.utils_analyses_plots import (
    _ok_only,
    filter_by_norm_threshold,
    filter_exclude_zero_norms,
    spearmanr_permutation,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


# set up logging.
@app.callback()
def main():

    setup_logging(log_dir=Path("logs"))
    logger = logging.getLogger(__name__)
    logger.info("CLI started ...")


# presto variance violin function
def _absdev_sum_for_dataset(gdf: pd.DataFrame, *, dims: list[int]) -> pd.DataFrame:

    cols = [f"l2_dim{d}" for d in dims]
    X = gdf[cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)

    mu = X.mean(axis=0)
    abs_dev_sum = np.abs(X - mu).sum(axis=1)  # combine across dims

    return pd.DataFrame(
        {
            "dataset_id": gdf["dataset_id"].to_numpy(),
            "abs_dev": abs_dev_sum,
        }
    )


# Presto variance violin function
# Deprecated but kept for bugfixing original PV code comparison
# Reason: Using squared deviation makes distribution graphs difficult to interpret
# Therefore the absolute deviation version is preferred,
# which keeps the relative ordering and distribution information.
def _sqdev_long_for_dataset(gdf: pd.DataFrame, *, dims: list[int]) -> pd.DataFrame:

    cols = [f"l2_dim{d}" for d in dims]
    X = gdf[cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    mu = X.mean(axis=0)  # per-dimension means within this dataset

    sq = (X - mu) ** 2  # shape (N, D)

    out = pd.DataFrame(
        {
            "dataset_id": np.repeat(gdf["dataset_id"].to_numpy(), len(dims)),
            "dimension": np.tile([f"dim {d}" for d in dims], X.shape[0]),
            "sq_dev": sq.reshape(-1),
        }
    )
    return out


# individual violin plot helper
PARAMS = [
    "scaling",
    "log_transform",
    "feature_subset",
    "duplicate_handling",
    "missingness",
    "seed",
]


# presto individual violin plot helper
def _individual_sensitivities_for_param(
    df: pd.DataFrame,
    *,
    param: str,
    homology_dims=(0, 1, 2),
) -> tuple[np.ndarray, dict]:

    if param not in df.columns:
        raise ValueError(f"Param {param} not in df columns.")

    # Fix all other params ~> equivalence class
    fixed_cols = [c for c in PARAMS if c != param]

    if len(df) == 0:
        return np.array([], dtype=float), {
            "q": 0,
            "singletons": 0,
            "mean_class_size": np.nan,
        }

    grouped = df.groupby(fixed_cols, dropna=False)

    vals = []
    sizes = []
    singletons = 0

    for _, g in grouped:
        # equivalence class size = how many universes share the same other params
        m = len(g)
        sizes.append(m)
        if m < 2:
            singletons += 1
            # presto variance of a singleton is 0 so sensitivity is also 0, which is uninformative
            vals.append(0.0)
            continue

        pv = compute_presto_variance_from_metrics_table(g, homology_dims=homology_dims)
        vals.append(float(np.sqrt(max(pv, 0.0))))

    vals = np.asarray(vals, dtype=float)
    return vals, {
        "q": int(len(vals)),
        "singletons": int(singletons),
        "mean_class_size": float(np.mean(sizes)) if sizes else np.nan,
    }


# Example plot function
@app.command("example-plots")
def example_plots(
    out_dir: Path = typer.Option(Path("data/figures"), help="Output directory"),
):
    plot_example_figures(out_dir=out_dir)


@app.command("presto-variance-violin")
def presto_variance_violin(
    split: str = typer.Option("test", help="train/val/test"),
    dims: str = typer.Option(
        "0,1,2", help="Comma-separated homology dims, e.g. '0,1,2'"
    ),
    out_dir: Path = typer.Option(Path("data/figures"), help="Output directory"),
    show: bool = typer.Option(False, help="Show interactive window after saving."),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (e.g. 100).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
    log_y: bool = typer.Option(
        False,
        "--log-y/--no-log-y",
        help="Plot y-axis (abs_dev) on a log scale. Non-positive values are excluded.",
    ),
):

    out_dir.mkdir(parents=True, exist_ok=True)

    dims_list = [int(x.strip()) for x in dims.split(",") if x.strip()]
    if not dims_list:
        raise typer.BadParameter("No dims provided. Example: --dims '0,1,2'.")

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)
    df = df[df["metrics_status"] == "ok"].copy()
    if df.empty:
        raise typer.BadParameter("No 'ok' rows found in metrics table for this split.")

    # Ensure required columns exist
    needed = [f"l2_dim{d}" for d in dims_list]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"Missing required columns in metrics table: {missing}"
        )

    # Apply filters
    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)
    if df.empty:
        raise typer.BadParameter("No rows remain after filtering.")

    # Build combined absolute deviation per universe
    parts = []
    for _, gdf in df.groupby("dataset_id", dropna=False):
        parts.append(_absdev_sum_for_dataset(gdf, dims=dims_list))
    df_plot = pd.concat(parts, ignore_index=True)

    df_plot = df_plot[df_plot["dataset_id"].notna()].copy()
    df_plot = df_plot[np.isfinite(df_plot["abs_dev"].to_numpy(dtype=float))].copy()
    if df_plot.empty:
        raise typer.BadParameter("No finite combined absolute deviations to plot.")

    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(
        data=df_plot,
        x="dataset_id",
        y="abs_dev",
        inner="box",
        cut=0.0,
        linewidth=1,
    )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("", fontsize=14)

    ax.set_ylabel(
        # r"$\sum^h_{x=0}|\|L\|_p-\mu_{\mathcal{L}^x}|$  (summed absolute norm deviation)",
        "Summed absolute norm deviation",
        fontsize=14,
    )
    ax.tick_params(labelsize=12)

    log_tag = "logy-1" if log_y else "logy-0"
    out_path = out_dir / (
        f"presto_absdev_combined_violin_split-{split}"
        f"_dims-{dims.replace(',', '-')}"
        f"_thr-{norm_threshold}_zero-{exclude_zero_norms}_{log_tag}.png"
    )
    ax.figure.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved Presto variance violin plot to %s", out_path)

    if show:
        plt.show()
    plt.close(ax.figure)


@app.command("presto-individual-violin")
def presto_individual_violin(
    dataset: str = typer.Option(
        "all",
        help="Dataset id (e.g. 'NF-ToN-IoT-v3') or 'all' to pool across datasets.",
    ),
    split: str = typer.Option("test"),
    norm_threshold: float | None = typer.Option(None),
    exclude_zero_norms: bool = typer.Option(False),
    show: bool = typer.Option(False, help="Show interactive window after saving."),
    out_dir: Path = typer.Option(Path("data/figures")),
    log_y: bool = typer.Option(
        False,
        "--log-y/--no-log-y",
        help="Plot y-axis (individual_sensitivity) on a log scale. Non-positive values are excluded.",
    ),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split, require_exists=True))
    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    if dataset.lower() != "all":
        df = df[df["dataset_id"] == dataset].copy()
        if df.empty:
            raise typer.BadParameter(
                f"No rows for dataset={dataset!r} after filtering."
            )

    rows = []
    plot_params = []

    for p in PARAMS:
        n_unique = df[p].nunique(dropna=False) if p in df.columns else 0
        if n_unique >= 2:
            plot_params.append(p)

    for p in plot_params:
        vals, meta = _individual_sensitivities_for_param(
            df, param=p, homology_dims=(0, 1, 2)
        )
        for v in vals:
            rows.append(
                {
                    "dataset": dataset if dataset != "all" else "ALL",
                    "param": p,
                    "individual_sensitivity": v,
                }
            )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise typer.BadParameter("No individual sensitivities to plot.")

    plt.figure(figsize=(9, 3.4))
    ax = sns.violinplot(
        data=plot_df,
        x="param",
        y="individual_sensitivity",
        inner="box",
        cut=0,
        linewidth=1,
    )

    if log_y:
        ax.set_yscale("log")

    # modify duplicates tick text to ensure readability
    x_ticks = ax.get_xticks()
    if "UNSW" in dataset:
        x_labels = [
            "Scaling",
            "Log transform",
            "Feature subset",
            "Duplicates",
            "Missing",
            "Seed",
        ]
    else:
        x_labels = ["Scaling", "Log transform", "Feature subset", "Duplicates", "Seed"]

    # set axes labels to empty string to reproduce the combination figure in the paper
    if "UNSW" in dataset:
        ax.set_ylabel("", fontsize=1)
        ax.set_xlabel("Preprocessing parameter", fontsize=14)
    elif "ToN" in dataset:
        ax.set_ylabel("Individual PRESTO sensitivity", fontsize=14)
        ax.set_xlabel("", fontsize=1)
    elif "CIC" in dataset:
        ax.set_ylabel("", fontsize=1)
        ax.set_xlabel("", fontsize=1)
    else:
        ax.set_ylabel("Individual PRESTO sensitivity", fontsize=14)
        ax.set_xlabel("Preprocessing parameter", fontsize=14)

    scope = dataset if dataset != "all" else "ALL datasets pooled"
    ax.set_title(f"{scope}", fontsize=14)
    ax.set_xticks(ticks=x_ticks)
    ax.set_xticklabels(labels=x_labels, ha="center", fontsize=12)
    ax.tick_params(labelsize=12)

    log_tag = "logy-1" if log_y else "logy-0"
    out_path = out_dir / (
        f"presto_individual_violin_split-{split}_dataset-{dataset}_{log_tag}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()

    print(f"Saved: {out_path}")


# Violin plots of median MSE per dataset
@app.command("performance-summary")
def performance_summary_plot(
    split: str = typer.Option("test"),
    out_dir: Path = typer.Option(Path("data/figures")),
    perf_col: str = typer.Option("recon_mse_mean"),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (optional).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False, help="Exclude universes with all-zero l2_dim* (optional)."
    ),
    show: bool = typer.Option(False, help="Show interactive window after saving."),
):

    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)
    df = df[df["metrics_status"] == "ok"].copy()

    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    if perf_col not in df.columns:
        raise typer.BadParameter(f"Missing {perf_col!r} in metrics table.")

    df = df[np.isfinite(df[perf_col].to_numpy(dtype=float))].copy()

    g = sns.FacetGrid(
        df,
        col="dataset_id",
        sharey=False,
        height=3,
        aspect=0.9,
    )

    g.map_dataframe(
        sns.violinplot,
        y=perf_col,
        inner="box",
        cut=0,
        linewidth=1,
    )

    g.set_axis_labels("", "Mean MSE", fontsize=14)
    g.set_titles(col_template="{col_name}", size=14)

    for ax in g.axes.flat:
        ax.tick_params(axis="both", which="major", labelsize=12)

    out_path = out_dir / f"perf_violin_facet_split-{split}_eval-{perf_col}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    logger.info("Saved performance summary plot to %s", out_path)


# Scatter plot of median MSE vs per-universe L2-norm average, plotted per dataset
@app.command("topology-vs-performance")
def topology_vs_performance_plot(
    split: str = typer.Option("test"),
    out_dir: Path = typer.Option(Path("data/figures")),
    topo_col: str = typer.Option("l2_average"),
    perf_col: str = typer.Option("recon_mse_median"),
    log_topology: bool = typer.Option(
        False, help="If set, plot log10(topo_col) for readability."
    ),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (optional).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False, help="Exclude universes with all-zero l2_dim* (optional)."
    ),
    show: bool = typer.Option(False, help="Show interactive window after saving."),
):

    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)
    df = _ok_only(df)

    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    for c in [topo_col, perf_col, "dataset_id"]:
        if c not in df.columns:
            raise typer.BadParameter(f"Missing required column {c!r} in metrics table.")

    df = df.copy()
    df = df[np.isfinite(df[topo_col].to_numpy(dtype=float))]
    df = df[np.isfinite(df[perf_col].to_numpy(dtype=float))]

    if log_topology:
        eps = 1e-30
        df["_topo_x"] = np.log10(np.maximum(df[topo_col].to_numpy(dtype=float), eps))
        if topo_col == "l2_average":
            xlab = r"$\log_{10}($mean $L^2)$"

    else:
        df["_topo_x"] = df[topo_col].to_numpy(dtype=float)
        if topo_col == "l2_average":
            xlab = r"Average $L^2$"

    datasets = sorted(df["dataset_id"].dropna().unique().tolist())
    n = len(datasets)
    if n == 0:
        raise typer.BadParameter("No datasets available after filtering.")

    fig, axes = plt.subplots(n, 1, figsize=(5.5, 4 * n), sharey=False)
    if n == 1:
        axes = [axes]
    fig.tight_layout(pad=3.6)

    for ax, ds in zip(axes, datasets):
        sdf = df[df["dataset_id"] == ds].copy()

        sns.scatterplot(
            data=sdf,
            x="_topo_x",
            y=perf_col,
            ax=ax,
            s=18,
            alpha=0.65,
            linewidth=0,
        )

        # Spearman annotation
        x = sdf["_topo_x"].to_numpy(dtype=float)
        y = sdf[perf_col].to_numpy(dtype=float)

        if len(x) >= 3:
            r, p = spearmanr_permutation(x, y)
            txt = f"Spearman r={r:.2g}\nperm p={p:.2g}\nn={len(x)}"
        else:
            txt = f"n={len(x)}"

        ax.text(
            0.96,
            0.50,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
        )

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_title(ds, fontsize=16)
        ax.set_xlabel(xlab if ax is axes[-1] else "", fontsize=14)
        ax.set_ylabel("Median MSE", fontsize=14)

    out_path = (
        out_dir
        / f"topo_vs_perf_split-{split}_logtopo-{int(log_topology)}_topocol-{topo_col}_perfeval-{perf_col}.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Saved topology-vs-performance plot to %s", out_path)


if __name__ == "__main__":
    app()
