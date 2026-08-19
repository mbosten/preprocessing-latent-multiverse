from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import typer
from project_utils import setup_logging
from typing_extensions import Annotated

from preprolamu.pipeline.autoencoder import create_embedding
from preprolamu.pipeline.create_tda import run_tda_for_universe
from preprolamu.pipeline.cross_dataset_evaluation import save_generalization
from preprolamu.pipeline.evaluation import save_evaluation
from preprolamu.pipeline.preprocessing import Preprocessor
from preprolamu.pipeline.universes import generate_multiverse, get_universe

logger = logging.getLogger(__name__)
app = typer.Typer(help="Simulation + TDA pipeline")


# set up logging.
@app.callback()
def main():
    setup_logging(
        log_dir=Path("logs"),
        suppress_loggers=[
            "PIL",
            "matplotlib.font_manager",
            "matplotlib.texmanager",
            "matplotlib.dviread",
        ],
    )
    logger.info("CLI started ...")


# preprocessing CLI function
@app.command("prepare-preprocessing")
def prepare_preprocessing(
    universe_index: Annotated[int | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
):
    if universe_index is None:
        raise typer.BadParameter("Universe parameter can not currently be None!")

    universe = get_universe(universe_index)
    Preprocessor(universe).run(overwrite=overwrite)


# train AEs and retrieve embedding space.
@app.command("prepare-embeddings")
def prepare_embeddings(
    universe_index: Annotated[int | None, typer.Option()] = None,
    split: Annotated[Literal["train", "val", "test"], typer.Option()] = "test",
    retrain: Annotated[bool, typer.Option()] = False,
    overwrite: Annotated[bool, typer.Option()] = False,
):
    if universe_index is None:
        raise typer.BadParameter("Universe parameter can not currently be None!")

    u = get_universe(universe_index)
    create_embedding(
        universe=u,
        split=split,
        retrain=retrain,
        overwrite=overwrite,
    )


# compute persistent homology and related metrics from embeddings.
@app.command("prepare-tda")
def prepare_tda(
    universe_index: Annotated[int, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
):
    """
    Compute PH metrics for a single universe.
    This function should be parallelized externally to iterate over the multiverse.
    """
    if universe_index is None:
        raise typer.BadParameter("Universe parameter can not currently be None!")
    
    u = get_universe(universe_index)
    run_tda_for_universe(u, overwrite=overwrite)


# Evaluate each model's performance on test set
@app.command("prepare-evaluation")
def prepare_eval(
    universe_index: Annotated[int | None, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
):
    if universe_index is None:
        raise typer.BadParameter("Universe parameter can not currently be None!")

    u = get_universe(universe_index)
    save_evaluation(u, overwrite=overwrite)


@app.command("prepare-cross-evaluation")
def prepare_cross_eval(
    universe_index: Annotated[int, typer.Option()] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
):
    if universe_index is None:
        raise typer.BadParameter("Universe parameter can not currently be None!")

    universes = generate_multiverse()
    u = get_universe(universe_index)

    save_generalization(
        universe=u,
        universes=universes,
        overwrite=overwrite,
    )
    

# list all universes that can be simulated.
@app.command("list-universes")
def list_universes():

    universes = generate_multiverse()
    for i, universe in enumerate(universes):
        typer.echo(f"{universe.universe_index:3d}: {universe.id}")


if __name__ == "__main__":
    app()
