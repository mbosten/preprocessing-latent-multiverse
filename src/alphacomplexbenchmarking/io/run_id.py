# src/alphacomplexbenchmarking/io/run_id.py
from __future__ import annotations
from datetime import datetime

def make_run_id(seed: int, n_samples: int, n_dims: int, dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now()

    # Pad parameters with zeros for unambiguity in parameter values
    # samples:  8 digits
    # dims:     4 digits
    # seed:     6 digits
    # date:     8 digits (YYYYMMDD)
    # time:     6 digits (HHMMSS)

    return f"{n_samples:08d}{n_dims:04d}{seed:06d}{dt:%Y%m%d}{dt:%H%M%S}"


def parse_run_id(run_id: str) -> dict[str, int | str]:
    """
    Reverse of make_run_id(). Accepts either the bare run_id or a filename like
    '000123...142015.npz' (extension is ignored).
    Returns seed, n_samples, n_dims, date (YYYYMMDD), time (HHMMSS).
    """
    stem = run_id.split(".")[0]

    seed      = int(stem[0:6])
    n_samples = int(stem[6:14])
    n_dims    = int(stem[14:18])
    date      = stem[18:26]  # keep as string
    time      = stem[26:32]  # keep as string

    return {
        "seed": seed,
        "n_samples": n_samples,
        "n_dims": n_dims,
        "date": date,
        "time": time,
    }