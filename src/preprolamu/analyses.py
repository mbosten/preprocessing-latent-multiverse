# src/preprolamu/analyses.py
from __future__ import annotations

import logging
from pathlib import Path

import typer

from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.metrics import build_landscape_norm_table
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


@app.command("summarize")
def summarize(
    split: str = typer.Option("test", help="Which split to analyze (train/val/test)."),
    out_dir: Path = typer.Option(
        Path("data/processed/analysis"), help="Where to write outputs."
    ),
):
    """
    Build a tidy table of landscape norms (one row per universe), then print dataset summaries,
    including mean aggregate norm and mean per homology dimension.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = build_landscape_norm_table(universes, split=split, require_exists=True)

    # Save the tidy table
    tidy_path = out_dir / f"landscape_norms_{split}.csv"
    df.to_csv(tidy_path, index=False)
    print(f"Wrote: {tidy_path}")

    # Only summarize successful rows
    ok = df[df["landscape_status"] == "ok"].copy()

    console_summ = (
        ok.groupby("dataset_id")
        .agg(
            count=("norm_aggregate", "count"),
            norm_aggregate_mean=("norm_aggregate", "mean"),
            norm_aggregate_std=("norm_aggregate", "std"),
            norm_average_mean=("norm_average", "mean"),
            norm_average_std=("norm_average", "std"),
        )
        .reset_index()
        .sort_values("dataset_id")
    )

    print("\n=== Summary by dataset (aggregate & average landscape norm) ===")
    print(console_summ.to_string(index=False))
