# src/preprolamu/analyses.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import typer
from typing_extensions import Annotated

from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.metrics import (
    build_metrics_table,
    compute_presto_variance_from_metrics_table,
)
from preprolamu.pipeline.universes import generate_multiverse
from preprolamu.plots import _parse_split_by

logger = logging.getLogger(__name__)


app = typer.Typer(help="Data analyses based on landscapes and embeddings.")


# ----------- Global CLI options and commands ----------- #
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


# ----------- Helper functions ----------- #
def _ok_only(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Dropping %d universes with metrics_status != 'ok'",
        len(df) - df[df["metrics_status"] == "ok"].shape[0],
    )
    return df[df["metrics_status"] == "ok"].copy()


def _filter_by_norm_threshold(
    df: pd.DataFrame,
    *,
    threshold: float | None,
) -> pd.DataFrame:
    """
    Keep only universes whose landscape L2 norms are <= threshold
    for *all* available homology dimensions (l2_dim* columns).

    If threshold is None: return df unchanged.
    """
    if threshold is None:
        return df

    # Find all l2_dim{d} columns present
    dim_cols = sorted([c for c in df.columns if c.startswith("l2_dim")])
    if not dim_cols:
        raise typer.BadParameter(
            "Requested norm threshold filtering, but no 'l2_dim*' columns exist in the table."
        )

    before = len(df)

    # Keep universes where every dimension norm is <= threshold (NaNs treated as fail-safe drop)
    mask = pd.Series(True, index=df.index)
    for c in dim_cols:
        mask &= df[c].notna() & (df[c] <= threshold)

    df2 = df[mask].copy()

    logger.info(
        "Applied norm threshold across dims: kept %d/%d where max(%s) <= %.6g (dropped=%d)",
        len(df2),
        before,
        ",".join(dim_cols),
        threshold,
        before - len(df2),
    )
    return df2


def _filter_exclude_zero_norms(df: pd.DataFrame, *, exclude_zero: bool) -> pd.DataFrame:
    """
    Optionally drop universes whose landscape L2 norms are all exactly zero
    across all available homology dimensions (l2_dim* columns).

    If exclude_zero is False: return df unchanged.
    """
    if not exclude_zero:
        return df

    dim_cols = sorted([c for c in df.columns if c.startswith("l2_dim")])
    if not dim_cols:
        raise typer.BadParameter(
            "Requested exclude_zero filtering, but no 'l2_dim*' columns exist in the table."
        )

    before = len(df)

    # Drop rows where ALL dimension norms are exactly 0.0
    # (NaNs do not trigger dropping.)
    is_all_zero = (df[dim_cols] == 0).all(axis=1)

    df2 = df[~is_all_zero].copy()

    logger.info(
        "Excluded all-zero norms across dims: kept %d/%d (dropped=%d) using cols=%s",
        len(df2),
        before,
        before - len(df2),
        ",".join(dim_cols),
    )
    return df2


# ----------- CLI commands ----------- #
@app.command("table")
def make_table(
    split: str = typer.Option("test", help="train/val/test"),
    out: Path = typer.Option(Path("data/processed/analysis/metrics_table.csv")),
):
    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved table to %s (rows=%d)", out, len(df))


@app.command("dataset-summary")
def dataset_summary(
    split: str = typer.Option("test"),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Only include universes with norm <= this threshold (for outlier-robust summaries).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split))

    df = _filter_by_norm_threshold(df, threshold=norm_threshold)
    df = _filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    # Example: compare distributions of l2_average across datasets
    summary = (
        df.groupby("dataset_id")["l2_average"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())


@app.command("parameter-effect")
def parameter_effect(
    param: Annotated[
        Literal[
            "scaling",
            "log_transform",
            "feature_subset",
            "duplicate_handling",
            "missingness",
            "seed",
        ],
        typer.Option(
            "--param",
            help="Which multiverse parameter to compare?",
            show_choices=True,
        ),
    ],
    split: Annotated[
        Literal["train", "val", "test"],
        typer.Option(
            "--split",
            help="Dataset split to analyze",
            show_choices=True,
        ),
    ] = "test",
    dataset_id: Optional[str] = typer.Option(
        None,
        help="Optional: restrict to one dataset ('NF-ToN-IoT-v3', 'NF-UNSW-NB15-v3', 'NF-CICIDS2018-v3')",
    ),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Only include universes with norm <= this threshold (for outlier-robust comparisons).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    """
    Compare distributions of l2_average between the settings of one parameter.
    Minimal version: group summaries + a simple nonparametric test.
    """
    try:
        from scipy.stats import kruskal, mannwhitneyu
    except Exception:
        raise typer.Exit(code=2)  # scipy missing; keep script minimal and fail early

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split))

    df = _filter_by_norm_threshold(
        df,
        threshold=norm_threshold,
    )
    df = _filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id].copy()

    if param not in df.columns:
        raise typer.BadParameter(
            f"Unknown param {param!r}. Available: {list(df.columns)}"
        )

    # summaries
    grp = df.groupby(param)["l2_average"]
    summ = grp.agg(["count", "mean", "median", "std"]).sort_values(
        "median", ascending=False
    )
    print("\nSummary:\n", summ.to_string())

    # stats: 2 groups => Mann–Whitney, >2 => Kruskal
    groups = [g.dropna().to_numpy() for _, g in grp]
    labels = list(grp.groups.keys())

    if len(groups) == 2:
        stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        print(f"\nMann–Whitney U: {labels[0]} vs {labels[1]}: U={stat:.3g}, p={p:.3g}")
    elif len(groups) > 2:
        stat, p = kruskal(*groups)
        print(f"\nKruskal–Wallis: H={stat:.3g}, p={p:.3g}")
    else:
        print("\nNot enough groups to test.")


@app.command("presto-variance")
def presto_variance(
    split: str = typer.Option("test"),
    split_by: str = typer.Option(
        "dataset",
        help="Comma-separated grouping keys (max 2). Examples: 'dataset' or 'dataset,scaling'."
        'Use two double quotes ("") for no grouping (all universes together).',
    ),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (e.g. 100).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    """
    Compute PRESTO variance per group using precomputed norms from metrics JSON
    (no landscape loading).
    """
    # parse split_by (reuse your parsing logic / aliases)
    keys = _parse_split_by(split_by)  # use the same alias parser you already wrote

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split, require_exists=True))

    # threshold across dims (your updated _filter_by_norm_threshold)
    df = _filter_by_norm_threshold(df, threshold=norm_threshold)

    df = _filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    # Ensure required columns exist
    needed = [f"l2_dim{d}" for d in (0, 1, 2)]
    for c in needed:
        if c not in df.columns:
            raise typer.BadParameter(
                f"Missing {c} in metrics table; cannot compute PRESTO variance."
            )

    # Group and compute
    if not keys:
        grouped = [(("ALL",), df)]
    else:
        grouped = list(df.groupby(keys, dropna=False))

    rows = []
    for group_key, gdf in grouped:
        logger.info(
            "Computing PRESTO variance for group %s with %d universes",
            group_key,
            len(gdf),
        )
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        if gdf.empty:
            continue

        v = compute_presto_variance_from_metrics_table(gdf, homology_dims=(0, 1, 2))

        row = {"n": len(gdf), "presto_variance": v}
        for k, val in zip(keys, group_key):
            row[k] = val
        rows.append(row)

    if not rows:
        raise typer.BadParameter("No groups had any rows after filtering.")

    out = pd.DataFrame(rows)
    if keys:
        out = out.sort_values(
            keys + ["presto_variance"], ascending=[True] * len(keys) + [False]
        )
    else:
        out = out.sort_values("presto_variance", ascending=False)

    print(out.to_string(index=False))


if __name__ == "__main__":
    app()
