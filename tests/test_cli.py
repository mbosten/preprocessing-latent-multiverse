from pathlib import Path
import numpy as np
from typer.testing import CliRunner
from alphacomplexbenchmarking.cli import app

runner = CliRunner()


def test_alpha_persistence_cli(tmp_path: Path):
    # create a tiny 2D point set
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    mpath = tmp_path / "pts.npy"
    np.save(mpath, pts)

    out_npz = tmp_path / "pers.npz"
    out_json = tmp_path / "pers.json"
    args = [
        "alpha-persistence",
        "--input", str(mpath),
        "--dim", "0", "--dim", "1",
        "--output-npz", str(out_npz),
        "--output-json", str(out_json),
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0
    assert out_npz.exists()
    assert out_json.exists()
