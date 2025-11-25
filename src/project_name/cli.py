from pathlib import Path
import typer
from rich import print

app = typer.Typer(help="Project CLI")


@app.command()
def hello(name: str = typer.Argument("world")):
    """Say hello."""
    print(f"[bold green]Hello, {name}![/bold green]")


@app.command()
def process(config: Path = Path("configs/default.yaml")):
    """Example command that would load config & run a pipeline."""
    if not config.exists():
        print(f"[red]Config not found:[/red] {config}")
        raise typer.Exit(code=1)
    print(f"Using config: {config}")


if __name__ == "__main__":
    app()
