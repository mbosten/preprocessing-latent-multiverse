# src/preprolamu/cli.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import typer
from typing_extensions import Annotated

from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.create_embeddings import get_or_compute_latent
from preprolamu.pipeline.create_tda import run_tda_for_universe
from preprolamu.pipeline.metrics import compute_presto_variance_across_universes
from preprolamu.pipeline.preprocessing import preprocess_variant
from preprolamu.pipeline.universes import generate_multiverse, get_universe

app = typer.Typer(help="Simulation + TDA pipeline")


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


# ----------- Create preprocessing multiverse ----------- #
@app.command("prepare-preprocessing")
def prepare_preprocessing(
    universe_index: Annotated[int | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
):
    if universe_index is None:
        universes = generate_multiverse()

        for u in universes:
            preprocess_variant(u, overwrite=overwrite)
    else:
        u = get_universe(universe_index)
        preprocess_variant(u, overwrite=overwrite)


# ----------- Train AEs and create embeddings ----------- #
@app.command("prepare-embeddings")
def prepare_embeddings(
    universe_index: Annotated[int | None, typer.Option()] = None,
    split: Annotated[Literal["train", "val", "test"], typer.Option()] = "test",
    retrain_regardless: Annotated[bool, typer.Option()] = False,
    force_recompute: Annotated[bool, typer.Option()] = False,
):
    if universe_index is None:
        universes = generate_multiverse()

        for u in universes:
            get_or_compute_latent(
                u,
                split=split,
                retrain_regardless=retrain_regardless,
                force_recompute=force_recompute,
            )
    else:
        u = get_universe(universe_index)
        get_or_compute_latent(
            u,
            split=split,
            retrain_regardless=retrain_regardless,
            force_recompute=force_recompute,
        )


# ----------- Compute simplexes and TDA metrics ----------- #
@app.command("prepare-tda")
def prepare_tda(
    universe_index: Annotated[int | None, typer.Option()] = None,
):
    if universe_index is None:
        universes = generate_multiverse()

        for u in universes:
            run_tda_for_universe(u)
    else:
        u = get_universe(universe_index)
        run_tda_for_universe(u)

    if universe_index is None:
        compute_presto_variance_across_universes(
            universes,
        )


@app.command("list-universes")
def list_universes():
    """
    List all default Universes with their indices.
    """
    universes = generate_multiverse()
    for i, universe in enumerate(universes):
        typer.echo(f"{i:3d}: {universe.to_id_string()}")


if __name__ == "__main__":
    app()
