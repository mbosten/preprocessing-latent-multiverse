from typer.testing import CliRunner
from project_name.cli import app

runner = CliRunner()


def test_hello():
    result = runner.invoke(app, ["hello", "Ada"])
    assert result.exit_code == 0
    assert "Ada" in result.stdout
