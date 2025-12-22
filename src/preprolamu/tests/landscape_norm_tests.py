# src/preprolamu/tests/landscape_norm_tests.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import typer

from preprolamu.io.storage import load_embedding, load_projected
from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.universes import get_universe

logger = logging.getLogger(__name__)


app = typer.Typer(help="debugging")


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


def approx_diameter_farthest_first(
    X: np.ndarray, iters: int = 1000, seed: int = 42
) -> Tuple[float, int]:
    """
    Re-implements your normalize_space() diameter estimate:
    farthest-point traversal + diameter = max pairwise distance within the chosen subset.
    Returns (diameter, n_unique_in_subset).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return float("nan"), 0

    rng = np.random.default_rng(seed)
    subset = [int(rng.integers(0, n))]
    # farthest-point hopping
    for _ in range(iters - 1):
        x0 = X[subset[-1]]
        dists = np.linalg.norm(X - x0, axis=1)
        subset.append(int(np.argmax(dists)))

    S = X[np.array(subset, dtype=int)]
    # exact-duplicate count (cheap)
    n_unique = np.unique(S, axis=0).shape[0]

    # max pairwise distance on subset
    # (simple O(k^2) with k=iters, but 1000 -> fine)
    diffs = S[:, None, :] - S[None, :, :]
    dist_mat = np.linalg.norm(diffs, axis=2)
    diameter = float(np.max(dist_mat))
    return diameter, n_unique


def summarize_array(name: str, A: np.ndarray) -> Dict[str, float]:
    A = np.asarray(A)
    out: Dict[str, float] = {"n": float(A.shape[0])}
    out["finite_frac"] = float(np.isfinite(A).mean())
    if A.size == 0:
        return out
    if np.isfinite(A).any():
        Af = A[np.isfinite(A)]
        out["min"] = float(Af.min())
        out["max"] = float(Af.max())
        out["mean"] = float(Af.mean())
        out["std"] = float(Af.std())
    return out


def row_duplicate_fraction(
    X: np.ndarray, max_rows: int = 200_000, seed: int = 42
) -> float:
    """
    Approximate duplicate-row fraction by sampling up to max_rows rows.
    Exact unique over millions can be expensive; sampling is enough for diagnosis.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n == 0:
        return 0.0
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    u = np.unique(Xs, axis=0).shape[0]
    return float(1.0 - (u / Xs.shape[0]))


@app.command("check-geometry")
def check_geometry(
    universe_index: int = typer.Option(..., help="Universe index to inspect."),
    split: str = typer.Option("test", help="Split to inspect."),
    diameter_iters: int = typer.Option(
        1000, help="Iterations used for farthest-first diameter estimate."
    ),
    diameter_eps: float = typer.Option(
        1e-12, help="If diameter <= eps, normalization would blow up."
    ),
):
    """
    Diagnose whether giant persistence values are caused by (near-)zero diameter or
    extreme scaling in latent/projected spaces.
    """
    logging.basicConfig(level=logging.INFO)

    uni = get_universe(universe_index)
    uid = getattr(uni, "id_string", getattr(uni, "id", uni.to_id_string()))
    logger.info("Universe=%s split=%s", uid, split)

    # 1) Load latent (pre-normalization/PCA)
    latent = load_embedding(uni, split=split, force_recompute=False)
    logger.info("Loaded latent: shape=%s dtype=%s", latent.shape, latent.dtype)

    # 2) Load projected (post-normalization+PCA, pre-subsample)
    projected = load_projected(uni, split=split)
    logger.info("Loaded projected: shape=%s dtype=%s", projected.shape, projected.dtype)

    # Basic finiteness checks
    logger.info("[latent] finite=%s", bool(np.isfinite(latent).all()))
    logger.info("[projected] finite=%s", bool(np.isfinite(projected).all()))

    # Scale checks
    logger.info("[latent] max_abs=%.3e", float(np.nanmax(np.abs(latent))))
    logger.info("[projected] max_abs=%.3e", float(np.nanmax(np.abs(projected))))

    # Duplicate checks (approx)
    dup_lat = row_duplicate_fraction(latent, seed=uni.seed)
    dup_proj = row_duplicate_fraction(projected, seed=uni.seed)
    logger.info("[latent] approx duplicate-row fraction ~ %.3f", dup_lat)
    logger.info("[projected] approx duplicate-row fraction ~ %.3f", dup_proj)

    # Recompute the approximate diameter you used for normalization
    diam, n_unique_subset = approx_diameter_farthest_first(
        latent, iters=diameter_iters, seed=42
    )
    logger.info(
        "[normalize] approx diameter=%.6e (iters=%d), unique_in_subset=%d/%d",
        diam,
        diameter_iters,
        n_unique_subset,
        diameter_iters,
    )

    if not np.isfinite(diam):
        logger.error("Diameter is not finite -> normalization is invalid.")
        raise typer.Exit(code=3)

    if diam <= diameter_eps:
        logger.error(
            "Diameter is tiny (<= %.1e). Dividing by this would blow up coordinates and persistence.",
            diameter_eps,
        )
        raise typer.Exit(code=4)

    # Additional: implied blow-up factor check
    # If your normalization does latent / diameter, then typical scale becomes ~|latent|/diameter.
    blow = float(np.nanmax(np.abs(latent))) / diam
    logger.info("[normalize] implied max_abs_after_norm ~ %.3e", blow)

    logger.info(
        "Done. If projected max_abs is enormous or implied blow-up is huge, that's your culprit."
    )


if __name__ == "__main__":
    app()
