from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import typer
from project_utils import setup_logging
from typing_extensions import Annotated

from preprolamu.pipeline.create_embeddings import get_or_compute_latent
from preprolamu.pipeline.create_tda import run_tda_for_universe
from preprolamu.pipeline.evaluation import evalae
from preprolamu.pipeline.preprocessing import preprocess_variant
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
        universes = generate_multiverse()

        for u in universes:
            preprocess_variant(u, overwrite=overwrite)
    else:
        u = get_universe(universe_index)
        preprocess_variant(u, overwrite=overwrite)


# train AEs and retrieve embedding space.
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
    if universe_index is not None:
        try:
            u = get_universe(universe_index)
        except Exception:
            raise typer.BadParameter("Not a valid universe index.")
        run_tda_for_universe(u, overwrite=overwrite)
    else:
        raise typer.BadParameter("Universe parameter can not currently be None!+")


# Evaluate each model's performance on test set
@app.command("prepare-evaluation")
def prepare_eval(
    universe_index: Annotated[int | None, typer.Option()] = None,
    batch_size: int = typer.Option(2048, help="Batch size for reconstruction eval"),
    overwrite: Annotated[bool, typer.Option()] = False,
    include_stratified: bool = typer.Option(
        True, help="Also compute Benign vs Attack summaries (no thresholding)"
    ),
):
    if universe_index is None:
        universes = generate_multiverse()

        for u in universes:
            evalae(
                universe=u,
                batch_size=batch_size,
                overwrite=overwrite,
                include_stratified=include_stratified,
            )
    else:
        u = get_universe(universe_index)
        evalae(
            universe=u,
            batch_size=batch_size,
            overwrite=overwrite,
            include_stratified=include_stratified,
        )
    return None


# list all universes that can be simulated.
@app.command("list-universes")
def list_universes():

    universes = generate_multiverse()
    for i, universe in enumerate(universes):
        typer.echo(f"{universe.universe_index:3d}: {universe.id}")


if __name__ == "__main__":
    app()
