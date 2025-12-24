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
    compute_presto_variance_across_universes,
)
from preprolamu.pipeline.universes import generate_multiverse

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


def _ok_only(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Dropping %d universes with metrics_status != 'ok'",
        len(df) - df[df["metrics_status"] == "ok"].shape[0],
    )
    return df[df["metrics_status"] == "ok"].copy()


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
):
    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split))

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
):
    """
    Reuse your existing compute_presto_variance_across_universes,
    and report per dataset.
    """
    universes = generate_multiverse()

    # by dataset
    for ds in sorted({u.dataset_id for u in universes}):
        ds_universes = [u for u in universes if u.dataset_id == ds]
        v = compute_presto_variance_across_universes(
            ds_universes, split=split, homology_dims=(0, 1, 2)
        )
        print(f"{ds}: presto_variance={v:.6g}")


if __name__ == "__main__":
    app()
