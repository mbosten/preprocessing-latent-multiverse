# src/alphacomplexbenchmarking/cli.py
from __future__ import annotations
import logging
from pathlib import Path
import typer
from rich import print
from typing import Optional
from typing_extensions import Annotated


from alphacomplexbenchmarking.logging_config import setup_logging
from alphacomplexbenchmarking.pipeline.universes import Universe, generate_multiverse, get_universe
from alphacomplexbenchmarking.pipeline.parallel import run_full_pipeline_for_universe, run_many_universes
from alphacomplexbenchmarking.pipeline.create_embeddings import get_or_compute_latent
from alphacomplexbenchmarking.pipeline.create_tda import run_tda_for_universe


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
    logger.debug("CLI started with verbose=%s", verbose)

# ----------- Train AEs and create embeddings ----------- #
@app.command("prepare-embeddings")
def prepare_embeddings(
    universe_index: Annotated[int | None, typer.Option()] = None,
    force_recompute: Annotated[bool, typer.Option()] = False
):
    if universe_index is None:
        universes = generate_multiverse()

        for u in universes:
            get_or_compute_latent(u, force_recompute=force_recompute)
    else:
        u = get_universe(universe_index)
        get_or_compute_latent(u, force_recompute=force_recompute)


# ----------- Compute simplexes and get TDA metrics ----------- #
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


# Will be removed in the future in favor of "prepare-embeddings" and "prepare-tda"
@app.command("run-universe")
def run_universe(
    index: int = typer.Argument(
        ..., help="Index of universe in the default universe list (0-based)."
    ),
):
    """
    Run the full pipeline for a single Universe from the default grid.
    """
    universes = generate_multiverse()
    if not (0 <= index < len(universes)):
        raise typer.BadParameter(f"index must be in [0, {len(universes) - 1}]")
    universe = universes[index]
    typer.echo(f"Running universe[{index}] = {universe.to_id_string()}")
    run_full_pipeline_for_universe(universe)

# Will be removed in the future in favor of "prepare-embeddings" and "prepare-tda"
@app.command("run-universe-batch")
def run_universe_batch(
    start: int = typer.Option(0, help="Start index (inclusive) in default universe list."),
    end: Optional[int] = typer.Option(
        None, help="End index (exclusive) in default universe list. If None, run to end."
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        help="Max parallel workers for local run. Default uses number of CPUs.",
    ),
):
    """
    Run a batch of Universes (potentially in parallel) from the default grid.
    """
    universes = generate_multiverse()
    n = len(universes)
    if end is None or end > n:
        end = n
    if start < 0 or start >= end:
        raise typer.BadParameter(f"Invalid start/end: start={start}, end={end}, total={n}")

    subset = universes[start:end]
    typer.echo(f"Running Universes[{start}:{end}] ({len(subset)} universes)")
    completed = run_many_universes(subset, max_workers=max_workers)
    typer.echo(f"Completed {len(completed)} universes.")


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