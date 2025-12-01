# src/alphacomplexbenchmarking/cli.py
from __future__ import annotations
import logging
from pathlib import Path
import typer
from rich import print
from typing import Optional

from alphacomplexbenchmarking.io.run_id import parse_run_id
from alphacomplexbenchmarking.logging_config import setup_logging
from alphacomplexbenchmarking.pipeline.specs import RunSpec, generate_default_specs
from alphacomplexbenchmarking.pipeline.parallel import run_full_pipeline_for_spec, run_many_specs

# For running the pipeline and self-checks
from alphacomplexbenchmarking.io.storage import (
    generate_and_store,
    compute_and_store_persistence_for_run,
    compute_and_store_landscapes_for_run,
    generate_simulation_matrix,
    compute_alpha_complex_persistence,
    compute_landscapes
)


app = typer.Typer(help="Simulation + TDA pipeline")


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


@app.command("run-spec")
def run_spec(
    index: int = typer.Argument(
        ..., help="Index of spec in the default spec list (0-based)."
    ),
):
    """
    Run the full pipeline for a single RunSpec (universe) from the default grid.
    """
    specs = generate_default_specs()
    if not (0 <= index < len(specs)):
        raise typer.BadParameter(f"index must be in [0, {len(specs) - 1}]")
    spec = specs[index]
    typer.echo(f"Running spec[{index}] = {spec.to_id_string()}")
    run_full_pipeline_for_spec(spec)


@app.command("run-spec-batch")
def run_spec_batch(
    start: int = typer.Option(0, help="Start index (inclusive) in default spec list."),
    end: Optional[int] = typer.Option(
        None, help="End index (exclusive) in default spec list. If None, run to end."
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        help="Max parallel workers for local run. Default uses number of CPUs.",
    ),
):
    """
    Run a batch of RunSpecs (potentially in parallel) from the default grid.
    """
    specs = generate_default_specs()
    n = len(specs)
    if end is None or end > n:
        end = n
    if start < 0 or start >= end:
        raise typer.BadParameter(f"Invalid start/end: start={start}, end={end}, total={n}")

    subset = specs[start:end]
    typer.echo(f"Running specs[{start}:{end}] ({len(subset)} specs)")
    completed = run_many_specs(subset, max_workers=max_workers)
    typer.echo(f"Completed {len(completed)} specs.")


@app.command("list-specs")
def list_specs():
    """
    List all default RunSpecs with their indices.
    """
    specs = generate_default_specs()
    for i, spec in enumerate(specs):
        typer.echo(f"{i:3d}: {spec.to_id_string()}")


@app.command("run-pipeline")
def run_pipeline(
    n_samples: int = 100,
    n_dims: int = 3,
    seed: int = 42,
    dim: list[int] = typer.Option([0, 1, 2], "--dim"),
):
    """
    Full pipeline:
      - generate data → data/raw/<run_id>.npz
      - persistence   → data/interim/persistence/<run_id>.npz
      - landscapes    → data/interim/landscapes/<run_id>.npz
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline with n_samples={n_samples}, n_dims={n_dims}, seed={seed}, dims={dim}")

    # 1) generate & store
    data, run_id, raw_path = generate_and_store(n_samples, n_dims, seed)
    print(f"[blue]Run ID:[/blue] {run_id}")
    print(f"[green]Raw data stored at:[/green] {raw_path}")

    # 2) persistence
    per_dim, pers_path = compute_and_store_persistence_for_run(data, run_id, dim)
    print(f"[green]Persistence stored at:[/green] {pers_path}")

    # 3) landscapes
    landscapes, land_path = compute_and_store_landscapes_for_run(per_dim, run_id, dim)
    print(f"[green]Landscapes stored at:[/green] {land_path}")

    logger.info(f"Pipeline finished for run_id={run_id}")
    print("[bold green]Pipeline complete.[/bold green]")


@app.command("self-check")
def self_check():
    """
    Run a small in-memory pipeline to check that the core steps still work.
    Prints step-by-step status; exits with code 0 on success, 1 on error.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running self-check")

    ok = True

    # --- Step 1: simulation ---
    try:
        print("[bold]Step 1: generate_simulation_matrix[/bold]")
        data = generate_simulation_matrix(n_samples=10, n_dims=3, seed=0)
        if data.shape != (10, 3):
            print(f"[red]❌ Unexpected shape: {data.shape}, expected (10, 3)[/red]")
            ok = False
        else:
            print("[green]✅ Simulation step succeeded[/green]")
    except Exception as exc:
        print(f"[red]❌ Simulation step failed: {exc}[/red]")
        ok = False

    # --- Step 2: persistence ---
    try:
        print("[bold]Step 2: compute_alpha_complex_persistence[/bold]")
        pers = compute_alpha_complex_persistence(data, homology_dimensions=[0, 1])

        # Basic sanity checks
        if not isinstance(pers, dict):
            print("[red]❌ Persistence result is not a dict[/red]")
            ok = False
        else:
            for dim in [0, 1]:
                arr = pers.get(dim)
                if arr is None:
                    print(f"[red]❌ Missing persistence for dim {dim}[/red]")
                    ok = False
                else:
                    print(f"[green]✅ Got {arr.shape[0]} intervals for dim {dim}[/green]")
    except Exception as exc:
        print(f"[red]❌ Persistence step failed: {exc}[/red]")
        ok = False

    # --- Step 3: landscapes ---
    try:
        print("[bold]Step 3: compute_landscapes[/bold]")
        num_landscapes = 5   # or whatever you actually use in compute_landscapes
        resolution = 50

        lands = compute_landscapes(
            pers,
            num_landscapes=num_landscapes,
            resolution=resolution,
            homology_dimensions=[0, 1],
        )

        if not isinstance(lands, dict):
            print("[red]❌ Landscapes result is not a dict[/red]")
            ok = False
        else:
            for dim in [0, 1]:
                arr = lands.get(dim)
                if arr is None:
                    print(f"[red]❌ Missing landscapes for dim {dim}[/red]")
                    ok = False
                    continue

                if arr.ndim != 2:
                    print(f"[red]❌ Landscapes dim {dim} has wrong ndim: {arr.ndim}, expected 2[/red]")
                    ok = False
                    continue

                expected_cols = num_landscapes * resolution
                if arr.shape[0] != 1 or arr.shape[1] != expected_cols:
                    print(
                        f"[red]❌ Unexpected shape for landscapes dim {dim}: "
                        f"{arr.shape}, expected (1, {expected_cols})[/red]"
                    )
                    ok = False
                else:
                    print(f"[green]✅ Landscapes for dim {dim} have shape {arr.shape}[/green]")
                    
    except Exception as exc:
        print(f"[red]❌ Landscapes step failed: {exc}[/red]")
        ok = False

    # --- overall result ---
    if ok:
        print("[bold green]🎉 Self-check passed: pipeline runs end-to-end.[/bold green]")
        raise typer.Exit(code=0)
    else:
        print("[bold red]⚠️ Self-check failed. See messages above.[/bold red]")
        raise typer.Exit(code=1)


@app.command("inspect-run")
def inspect_run(filename: str):
    """
    Given a raw/persistence/landscape filename, decode its parameters.
    Example:
      uv run fast inspect-run 00012300000050000320251126142015.npz
    """
    logger = logging.getLogger(__name__)
    logger.info("Inspecting run from filename=%s", filename)
    
    info = parse_run_id(filename)
    print("[bold]Decoded run ID:[/bold]")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    app()