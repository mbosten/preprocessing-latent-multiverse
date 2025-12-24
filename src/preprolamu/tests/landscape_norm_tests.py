# src/preprolamu/tests/landscape_norm_tests.py
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

from preprolamu.io.storage import load_embedding, load_landscapes, load_projected
from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.metrics import build_metrics_table
from preprolamu.pipeline.universes import generate_multiverse, get_universe

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


# ----------- Support functions ----------- #
def row_duplicate_fraction(X: np.ndarray, seed: int, max_rows: int = 200_000) -> float:
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


def _finite_l2(arr: np.ndarray) -> float:
    """L2 norm, ignoring non-finite entries. Returns inf if no finite entries."""
    a = np.asarray(arr, dtype=float)
    mask = np.isfinite(a)
    if not mask.any():
        return float("inf")
    return float(np.linalg.norm(a[mask]))


def _summarize_landscape_array(arr: np.ndarray) -> dict[str, float | int]:
    """Quick diagnostics for a landscape array."""
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    out: dict[str, float | int] = {
        "shape0": int(a.shape[0]) if a.ndim >= 1 else 0,
        "size": int(a.size),
        "finite_frac": float(finite.mean()) if a.size else 1.0,
        "n_nan": int(np.isnan(a).sum()),
        "n_posinf": int(np.isposinf(a).sum()),
        "n_neginf": int(np.isneginf(a).sum()),
    }
    if a.size and finite.any():
        af = a[finite]
        out["min"] = float(af.min())
        out["max"] = float(af.max())
        out["max_abs"] = float(np.max(np.abs(af)))
        out["l2_finite"] = float(np.linalg.norm(af))
    else:
        out["min"] = float("nan")
        out["max"] = float("nan")
        out["max_abs"] = float("nan")
        out["l2_finite"] = float("inf")
    return out


def _robust_z(x: pd.Series) -> pd.Series:
    """Median/MAD-based z-score, safer for outliers."""
    x = x.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or not math.isfinite(mad):
        return pd.Series([float("nan")] * len(x), index=x.index)
    return 0.6745 * (x - med) / mad


def _print_outlier_block(
    df: pd.DataFrame, dim: int, id_to_index: dict[str, int]
) -> None:
    """Pretty-print landscape outliers one universe at a time, narrow terminal-friendly."""
    for _, row in df.iterrows():
        uid = row["universe_id"]
        uidx = id_to_index.get(uid, None)

        typer.echo("-" * 72)
        typer.echo(f"Universe index:\t{uidx if uidx is not None else 'UNKNOWN'}")
        typer.echo(f"Universe ID:\t{uid}")
        typer.echo(f"L2 dim {dim:<2}:\t{float(row[f'l2_dim{dim}']):.3e}")

        if "robust_z" in row and pd.notna(row["robust_z"]):
            typer.echo(f"Robust Z:\t{float(row['robust_z']):.2f}")


_PARAM_KEYS = ["sc", "log", "fs", "dup", "miss", "sd"]


def _parse_params_from_universe_id(universe_id: str) -> dict[str, str]:
    """
    Extracts parameters from universe_id like:
    ds-XYZ_sc-minmax_log-log1p_fs-all_dup-keep_miss-impute_median_sd-42
    Returns dict { "sc": "minmax", "log": "log1p", ... }
    """
    parts = universe_id.split("_")
    out: dict[str, str] = {}
    for p in parts:
        for k in _PARAM_KEYS:
            prefix = f"{k}-"
            if p.startswith(prefix):
                out[k] = p[len(prefix) :]
    return out


def _print_param_distribution(
    outliers_df: pd.DataFrame,
    *,
    title: str,
    max_values_per_param: int = 12,
) -> None:
    """
    Print a compact distribution of universe parameters among the outliers.
    """
    typer.echo("\n" + "=" * 72)
    typer.echo(title)
    typer.echo("=" * 72)

    if outliers_df.empty:
        typer.echo("(no outliers to summarize)")
        return

    # Parse params for each outlier
    parsed = outliers_df["universe_id"].apply(_parse_params_from_universe_id)
    params_df = pd.DataFrame(list(parsed))

    n = len(params_df)
    for k in _PARAM_KEYS:
        if k not in params_df.columns:
            continue

        counts = params_df[k].fillna("(missing)").value_counts(dropna=False)
        typer.echo(f"\nParameter: {k} (n={n})")
        typer.echo("-" * 72)

        # Limit values to keep it readable
        shown = counts.head(max_values_per_param)
        for val, c in shown.items():
            pct = (c / n) * 100.0
            typer.echo(f"{val:<30}  {c:>3}  ({pct:>5.1f}%)")

        if len(counts) > max_values_per_param:
            typer.echo(f"... +{len(counts) - max_values_per_param} more values")


def _print_param_distribution_comparison(
    flagged_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    *,
    title: str,
) -> None:
    """
    Compare flagged vs baseline distributions for each param.
    Prints values that are enriched in flagged set.
    """
    typer.echo("\n" + "=" * 72)
    typer.echo(title)
    typer.echo("=" * 72)

    if flagged_df.empty:
        typer.echo("(no flagged universes)")
        return

    flagged_params = pd.DataFrame(
        list(flagged_df["universe_id"].apply(_parse_params_from_universe_id))
    )
    base_params = pd.DataFrame(
        list(baseline_df["universe_id"].apply(_parse_params_from_universe_id))
    )

    # Number of flagged and baseline universes, per dataset
    nf = len(flagged_params)
    nb = len(base_params)

    for k in _PARAM_KEYS:
        if k not in base_params.columns:
            continue

        # parameter values and their counts for the flagged and baseline set, respectively.
        f_counts = flagged_params[k].fillna("(missing)").value_counts()
        b_counts = base_params[k].fillna("(missing)").value_counts()

        # Union of all values so baseline distribution is complete
        all_vals = sorted(set(b_counts.index).union(set(f_counts.index)))

        rows = []

        for val in all_vals:

            # Flagged and baseline counts for this parameter value
            fc = int(f_counts.get(val, 0))
            bc = int(b_counts.get(val, 0))

            # Flagged and baseline proportions relative to their sets
            fp = (fc / nf) if nf else 0.0
            bp = (bc / nb) if nb else 0.0

            rows.append((val, fc, fp, bc, bp))

        # Sort by baseline proportion (so base% visually sums to ~100%)
        rows.sort(key=lambda t: t[4], reverse=True)

        typer.echo(f"\nParameter: {k}  (flagged n={nf}, baseline n={nb})")
        typer.echo("-" * 72)
        typer.echo(
            f"{'value':<28} {'flagged':>7} {'flag%':>7} {'base':>7} {'base%':>7}"
        )

        for val, fc, fp, bc, bp in rows:
            typer.echo(f"{val:<28} {fc:>7} {fp*100:>6.1f}% {bc:>7} {bp*100:>6.1f}%")


# ----------- CLI Functions ----------- #
@app.command("scan-landscape-outliers")
def scan_landscape_outliers(
    split: str = typer.Option("test", help="train/val/test"),
    dims: List[int] = typer.Option([1, 2], help="Homology dimensions to scan"),
    top_k: int = typer.Option(25, help="Show top K universes per dim"),
    dataset_id: Optional[str] = typer.Option(
        None, help="If provided, only scan universes for this dataset_id"
    ),
    threshold: Optional[float] = typer.Option(
        None, help="If set, only show universes with l2_dim{d} >= threshold"
    ),
    out_csv: Optional[Path] = typer.Option(
        None, help="Optional path to write a CSV with flagged universes"
    ),
):
    """
    Uses stored metrics JSONs (data/processed/metrics/*_metrics_{split}.json)
    to identify which universes create extreme landscape L2 norms.
    Does NOT touch analyses pipeline; does NOT recompute anything.
    """
    universes = generate_multiverse()
    id_to_index = {u.id: i for i, u in enumerate(universes)}
    df = build_metrics_table(universes, split=split, require_exists=True)

    if df is not None:
        if "dataset_id" not in df.columns:
            raise typer.BadParameter("Metrics table does not contain 'dataset_id'.")
        df = df[df["dataset_id"] == dataset_id].copy()
        if df.empty:
            raise typer.BadParameter(
                f"No rows found for dataset_id={dataset_id!r} (split={split})."
            )

    scope = f"dataset={dataset_id}" if dataset_id is not None else "all datasets"

    # Keep only usable rows
    if "metrics_status" in df.columns:
        df = df[df["metrics_status"] == "ok"].copy()

    # Ensure columns exist
    for d in dims:
        col = f"l2_dim{d}"
        if col not in df.columns:
            raise typer.BadParameter(
                f"Column {col} not found in metrics table. "
                f"Available: {list(df.columns)}"
            )

    flagged_rows = []

    for d in dims:
        col = f"l2_dim{d}"

        sub = df[["universe_id", "dataset_id", col, "metrics_path"]].copy()
        sub["robust_z"] = _robust_z(sub[col])
        sub = sub.sort_values(col, ascending=False)

        if threshold is not None:
            sub = sub[sub[col] >= threshold]

        show = sub.head(top_k)

        typer.echo(
            f"\n=== Global top outliers (split={split}, {scope}) — homology dim {d} ==="
        )
        if show.empty:
            typer.echo("(none)")
        else:
            _print_outlier_block(show, dim=d, id_to_index=id_to_index)

        # choose which rows were "flagged"/printed for this dim
        flagged = sub if threshold is not None else show
        baseline = df  # after dataset filter, status filter etc.

        _print_param_distribution_comparison(
            flagged_df=flagged,
            baseline_df=baseline,
            title=f"Parameter distribution comparison (flagged vs baseline) (dim={d}, split={split}, dataset={dataset_id or 'ALL'})",
        )

        # _print_param_distribution(
        #     flagged,
        #     title=f"Parameter distribution among flagged outliers (dim={d}, split={split}, dataset={dataset_id or 'ALL'})",
        # )

        if not sub.empty:
            # Collect all flagged if threshold is set, else collect top_k
            collect = sub if threshold is not None else show
            for _, r in collect.iterrows():
                flagged_rows.append(
                    {
                        "split": split,
                        "dim": d,
                        "universe_id": r["universe_id"],
                        "dataset_id": r["dataset_id"],
                        "l2": float(r[col]),
                        "robust_z": (
                            float(r["robust_z"]) if pd.notna(r["robust_z"]) else None
                        ),
                        "metrics_path": r["metrics_path"],
                    }
                )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flagged_rows).to_csv(out_csv, index=False)
        typer.echo(f"\nWrote {len(flagged_rows)} flagged rows to {out_csv}")


@app.command("inspect-landscape")
def inspect_landscape(
    universe_index: int = typer.Option(
        ..., help="Universe index (see: acb list-universes)"
    ),
    split: str = typer.Option("test", help="train/val/test"),
    dims: List[int] = typer.Option([1, 2], help="Homology dimensions to inspect"),
):
    """
    Loads the stored landscape NPZ for a single universe and prints diagnostics:
    finite fraction, NaN/Inf counts, max abs, and finite-only L2 norm.
    """
    uni = get_universe(universe_index)
    typer.echo(f"Universe: {uni.id}  (split={split})")
    L = load_landscapes(uni, split=split)

    for d in dims:
        arr = L.get(d, None)
        typer.echo(f"\n--- dim {d} ---")
        if arr is None:
            typer.echo("No landscape stored for this dim (None/missing).")
            continue

        summ = _summarize_landscape_array(arr)
        for k, v in summ.items():
            typer.echo(f"{k}: {v}")

        # Show first few problematic positions if any
        a = np.asarray(arr, dtype=float)
        bad = np.where(~np.isfinite(a))
        if a.size and len(bad) > 0 and bad[0].size > 0:
            idxs = list(zip(*[b[:10] for b in bad]))
            typer.echo(f"First non-finite indices (up to 10): {idxs}")


# Will likely be deprecated later
@app.command("check-geometry")
def check_geometry(
    universe_index: int = typer.Option(..., help="Universe index to inspect."),
    split: str = typer.Option("test", help="Split to inspect."),
    normalized_projections: bool = typer.Option(
        True, "--normalized", "-n", help="Whether to load normalized projections."
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
    projected = load_projected(uni, split=split, normalized=normalized_projections)
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


if __name__ == "__main__":
    app()
