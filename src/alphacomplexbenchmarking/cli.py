from pathlib import Path
import json
import typer
from rich import print


from .features.simulate_data import generate_simulation_matrix
from .complexes.alpha_complex_persistence import compute_alpha_complex_persistence


app = typer.Typer(help="Minimal CLI")

@app.command("test-sample-gen")
def test_sample_gen(
    n_samples: int = 100,
    n_dims: int = 3,
    seed: int = 42,
):
    data = generate_simulation_matrix(n_samples, n_dims, seed)
    
    print(f"[green]Wrote:[/green] {type(data)}, shape={data.shape}")


@app.command("run-pipeline")
def run_pipeline(
    n_samples: int = 100,
    n_dims: int = 3,
    seed: int = 42,
    dim: list[int] = typer.Option([0, 1, 2, 3], "--dim"),
    out_json: Path = Path("outputs/persistence.json")
):
    data = generate_simulation_matrix(n_samples, n_dims, seed)
    pers = compute_alpha_complex_persistence(data, homology_dimensions=dim)


    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({int(k): v.tolist() for k, v in pers.items()}, indent=2), encoding="utf-8")
    print(f"[green]Wrote:[/green] {out_json}")


if __name__ == "__main__":
    app()