# src/preprolamu/analyses.py
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

from preprolamu.logging_config import setup_logging
from preprolamu.pipeline.evaluation import (
    evaluate_autoencoder_reconstruction,
    save_eval_metrics,
)
from preprolamu.pipeline.metrics import (
    build_metrics_table,
    compute_presto_variance_from_metrics_table,
)
from preprolamu.pipeline.universes import generate_multiverse
from preprolamu.plots import _parse_split_by
from preprolamu.utils_analyses_plots import (
    _ok_only,
    filter_by_norm_threshold,
    filter_exclude_zero_norms,
)

logger = logging.getLogger(__name__)


app = typer.Typer(help="Data analyses based on landscapes and embeddings.")


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


_PRESTO_PARAMS: list[str] = [
    "scaling",
    "log_transform",
    "feature_subset",
    "duplicate_handling",
    "missingness",
    "seed",
]


def _presto_local_sensitivity_from_metrics_table(
    df_ds: pd.DataFrame,
    *,
    param: str,
    homology_dims: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """
    Wayland-style LOCAL PRESTO sensitivity for one parameter i, within ONE dataset.
    Uses equivalence classes defined by equality on all other parameters (j != i).

    Returns dict with:
      - local_sensitivity (sqrt of avg PV over classes)
      - avg_pv_over_classes
      - q (number of equivalence classes)
      - n (rows)
      - singleton_classes (classes of size 1)
      - mean_class_size
    """
    if df_ds.empty:
        return {
            "local_sensitivity": float("nan"),
            "avg_pv_over_classes": float("nan"),
            "q": 0,
            "n": 0,
            "singleton_classes": 0,
            "mean_class_size": float("nan"),
        }

    if param not in df_ds.columns:
        raise typer.BadParameter(
            f"Parameter {param!r} not found in metrics table columns."
        )

    # Equivalence classes: fix everything except param
    other_cols = [c for c in _PRESTO_PARAMS if c != param]
    missing = [c for c in other_cols if c not in df_ds.columns]
    if missing:
        raise typer.BadParameter(
            f"Cannot form equivalence classes for {param!r}: missing columns {missing}."
        )

    # Group into equivalence classes Q in Q_i
    grouped = list(df_ds.groupby(other_cols, dropna=False))
    q = len(grouped)

    # Compute PV(L[Q]) for each class Q and average
    pvs: list[float] = []
    sizes: list[int] = []
    singletons = 0

    for key, g in grouped:
        sizes.append(len(g))
        if len(g) <= 1:
            logger.debug(
                f" Equivalence class {key} is a singleton for dataset with length {q}."
            )
            singletons += 1
            pvs.append(0.0)
            continue

        pv = compute_presto_variance_from_metrics_table(g, homology_dims=homology_dims)
        pvs.append(float(pv))

    avg_pv = float(sum(pvs) / q) if q else float("nan")
    local = (
        math.sqrt(avg_pv)
        if (q and np.isfinite(avg_pv) and avg_pv >= 0)
        else float("nan")
    )

    return {
        "local_sensitivity": local,
        "avg_pv_over_classes": avg_pv,
        "q": q,
        "n": int(len(df_ds)),
        "singleton_classes": int(singletons),
        "mean_class_size": float(np.mean(sizes)) if sizes else float("nan"),
    }


def _presto_global_sensitivity_from_metrics_table(
    df_ds: pd.DataFrame,
    *,
    params: list[str] | None = None,
    homology_dims: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """
    Wayland-style GLOBAL PRESTO sensitivity within ONE dataset:
      sqrt( (1/c) * sum_i (1/q_i) * sum_Q PV(L[Q]) )
    which equals:
      sqrt( average over parameters of (average PV over equivalence classes for that param) )

    Returns dict with:
      - global_sensitivity
      - avg_pv_across_params   (the quantity under sqrt)
      - c (#params)
      - n (rows)
    """
    if df_ds.empty:
        return {
            "global_sensitivity": float("nan"),
            "avg_pv_across_params": float("nan"),
            "c": 0,
            "n": 0,
            "active_params": [],
        }

    use_params = params or _PRESTO_PARAMS

    active_params: list[str] = []
    avg_pvs: list[float] = []

    for p in use_params:
        if p not in df_ds.columns:
            logger.warning(
                f"Parameter {p!r} not found in metrics table columns; skipping."
            )
            continue

        n_unique = df_ds[p].nunique(dropna=False)
        if n_unique <= 1:
            continue

        res = _presto_local_sensitivity_from_metrics_table(
            df_ds, param=p, homology_dims=homology_dims
        )
        avg_pv = float(res["avg_pv_over_classes"])
        if np.isfinite(avg_pv):
            active_params.append(p)
            avg_pvs.append(avg_pv)

    c = len(avg_pvs)

    if c == 0:
        return {
            "global_sensitivity": float("nan"),
            "avg_pv_across_params": float("nan"),
            "c": 0,
            "n": int(len(df_ds)),
            "active_params": [],
        }

    avg_pv_across = float(np.mean(avg_pvs))
    global_ps = math.sqrt(avg_pv_across) if avg_pv_across >= 0 else float("nan")

    return {
        "global_sensitivity": global_ps,
        "avg_pv_across_params": avg_pv_across,
        "c": c,
        "n": int(len(df_ds)),
        "active_params": active_params,
    }


# ----------- CLI commands ----------- #
@app.command("table")
def make_table(
    split: str = typer.Option("test", help="train/val/test"),
    out: Path = typer.Option(Path("data/processed/analysis/metrics_table.csv")),
):
    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved table to %s (rows=%d)", out, len(df))


@app.command("dataset-summary")
def dataset_summary(
    split: str = typer.Option("test"),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Only include universes with norm <= this threshold (for outlier-robust summaries).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split))

    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    # Example: compare distributions of l2_average across datasets
    summary = (
        df.groupby("dataset_id")["l2_average"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())


@app.command("parameter-effect")
def parameter_effect(
    param: Annotated[
        Literal[
            "scaling",
            "log_transform",
            "feature_subset",
            "duplicate_handling",
            "missingness",
            "seed",
        ],
        typer.Option(
            "--param",
            help="Which multiverse parameter to compare?",
            show_choices=True,
        ),
    ],
    split: Annotated[
        Literal["train", "val", "test"],
        typer.Option(
            "--split",
            help="Dataset split to analyze",
            show_choices=True,
        ),
    ] = "test",
    dataset_id: Optional[str] = typer.Option(
        None,
        help="Optional: restrict to one dataset ('NF-ToN-IoT-v3', 'NF-UNSW-NB15-v3', 'NF-CICIDS2018-v3')",
    ),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Only include universes with norm <= this threshold (for outlier-robust comparisons).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    """
    Compare distributions of l2_average between the settings of one parameter.
    Minimal version: group summaries + a simple nonparametric test.
    """
    try:
        from scipy.stats import kruskal, mannwhitneyu
    except Exception:
        raise typer.Exit(code=2)  # scipy missing; keep script minimal and fail early

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split))

    df = filter_by_norm_threshold(
        df,
        threshold=norm_threshold,
    )
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    if dataset_id is not None:
        df = df[df["dataset_id"] == dataset_id].copy()

    if param not in df.columns:
        raise typer.BadParameter(
            f"Unknown param {param!r}. Available: {list(df.columns)}"
        )

    # summaries
    grp = df.groupby(param)["l2_average"]
    summ = grp.agg(["count", "mean", "median", "std"]).sort_values(
        "median", ascending=False
    )
    print("\nSummary:\n", summ.to_string())

    # stats: 2 groups => Mann–Whitney, >2 => Kruskal
    groups = [g.dropna().to_numpy() for _, g in grp]
    labels = list(grp.groups.keys())

    if len(groups) == 2:
        stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        print(f"\nMann–Whitney U: {labels[0]} vs {labels[1]}: U={stat:.3g}, p={p:.3g}")
    elif len(groups) > 2:
        stat, p = kruskal(*groups)
        print(f"\nKruskal–Wallis: H={stat:.3g}, p={p:.3g}")
    else:
        print("\nNot enough groups to test.")


@app.command("presto-variance")
def presto_variance(
    split: str = typer.Option("test"),
    split_by: str = typer.Option(
        "dataset",
        help="Comma-separated grouping keys (max 2). Examples: 'dataset' or 'dataset,scaling'."
        'Use two double quotes ("") for no grouping (all universes together).',
    ),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (e.g. 100).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    """
    Compute PRESTO variance per group using precomputed norms from metrics JSON
    (no landscape loading).
    """
    # parse split_by (reuse your parsing logic / aliases)
    keys = _parse_split_by(split_by)  # use the same alias parser you already wrote

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split, require_exists=True))

    # threshold across dims (your updated filter_by_norm_threshold)
    df = filter_by_norm_threshold(df, threshold=norm_threshold)

    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    # Ensure required columns exist
    needed = [f"l2_dim{d}" for d in (0, 1, 2)]
    for c in needed:
        if c not in df.columns:
            raise typer.BadParameter(
                f"Missing {c} in metrics table; cannot compute PRESTO variance."
            )

    # Group and compute
    if not keys:
        grouped = [(("ALL",), df)]
    else:
        grouped = list(df.groupby(keys, dropna=False))

    rows = []
    for group_key, gdf in grouped:
        logger.info(
            "Computing PRESTO variance for group %s with %d universes",
            group_key,
            len(gdf),
        )
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        if gdf.empty:
            continue

        v = compute_presto_variance_from_metrics_table(gdf, homology_dims=(0, 1, 2))

        row = {"n": len(gdf), "presto_variance": v}
        for k, val in zip(keys, group_key):
            row[k] = val
        rows.append(row)

    if not rows:
        raise typer.BadParameter("No groups had any rows after filtering.")

    out = pd.DataFrame(rows)
    if keys:
        out = out.sort_values(
            keys + ["presto_variance"], ascending=[True] * len(keys) + [False]
        )
    else:
        out = out.sort_values("presto_variance", ascending=False)

    print(out.to_string(index=False))


@app.command("presto-local-sensitivity")
def presto_local_sensitivity(
    split: str = typer.Option("test"),
    param: str = typer.Option(
        "all",
        help="One of: scaling, log_transform, feature_subset, duplicate_handling, missingness, seed, or 'all'.",
    ),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (e.g. 100).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    """
    Compute LOCAL PRESTO sensitivity per dataset from the metrics table only.
    If param='all', prints one row per parameter per dataset.
    """
    allowed = set(_PRESTO_PARAMS + ["all"])
    if param not in allowed:
        raise typer.BadParameter(f"--param must be one of {sorted(allowed)}")

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split, require_exists=True))
    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    datasets = sorted(df["dataset_id"].dropna().unique().tolist())
    if not datasets:
        raise typer.BadParameter("No datasets found after filtering.")

    params_to_run = _PRESTO_PARAMS if param == "all" else [param]

    rows: list[dict] = []
    for ds in datasets:
        df_ds = df[df["dataset_id"] == ds].copy()
        for p in params_to_run:
            res = _presto_local_sensitivity_from_metrics_table(
                df_ds, param=p, homology_dims=(0, 1, 2)
            )
            rows.append(
                {
                    "dataset_id": ds,
                    "param": p,
                    "n": res["n"],
                    "q": res["q"],
                    "singletons": res["singleton_classes"],
                    "mean_class_size": res["mean_class_size"],
                    "local_presto_sensitivity": res["local_sensitivity"],
                }
            )

    out = pd.DataFrame(rows)

    # nicer ordering for paper tables
    out["param"] = pd.Categorical(out["param"], categories=_PRESTO_PARAMS, ordered=True)
    out = out.sort_values(["dataset_id", "param"])

    # pretty numeric formatting
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print(out.to_string(index=False))


@app.command("presto-global-sensitivity")
def presto_global_sensitivity(
    split: str = typer.Option("test"),
    norm_threshold: Optional[float] = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (e.g. 100).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False,
        help="Exclude universes where all l2_dim* norms are exactly zero.",
    ),
):
    """
    Compute GLOBAL PRESTO sensitivity per dataset from the metrics table only.
    Global = sqrt( average over parameters of (average PV over equivalence classes for that param) ).
    """
    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split, require_exists=True))
    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    datasets = sorted(df["dataset_id"].dropna().unique().tolist())
    if not datasets:
        raise typer.BadParameter("No datasets found after filtering.")

    rows: list[dict] = []
    for ds in datasets:
        df_ds = df[df["dataset_id"] == ds].copy()
        res = _presto_global_sensitivity_from_metrics_table(
            df_ds, params=_PRESTO_PARAMS, homology_dims=(0, 1, 2)
        )
        rows.append(
            {
                "dataset_id": ds,
                "n": res["n"],
                "c": res["c"],
                "global_presto_sensitivity": res["global_sensitivity"],
            }
        )

    out = pd.DataFrame(rows).sort_values("dataset_id")
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print(out.to_string(index=False))


# CLI function that evaluates autoencoders and saves reconstruction-error metrics
@app.command("ae-eval")
def ae_eval(
    split: str = typer.Option("test", help="val or test"),
    batch_size: int = typer.Option(2048, help="Batch size for reconstruction eval"),
    overwrite: bool = typer.Option(False, help="Overwrite existing eval json files"),
    include_stratified: bool = typer.Option(
        True, help="Also compute Benign vs Attack summaries (no thresholding)"
    ),
):
    """
    Evaluate trained autoencoders on a split and store reconstruction-error metrics per universe.
    Writes: data/processed/eval_metrics/{universe_id}_eval_{split}.json
    """
    if split == "validation":
        split = "val"
    if split not in {"val", "test"}:
        raise typer.BadParameter("split must be 'val' or 'test'")

    universes = generate_multiverse()

    n_done = 0
    n_skipped = 0
    n_missing_model = 0

    for u in universes:
        out_path = u.eval_metrics_path(split=split)

        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue

        if not u.ae_model_path().exists():
            n_missing_model += 1
            continue

        try:
            payload = evaluate_autoencoder_reconstruction(
                u,
                split=split,
                batch_size=batch_size,
                include_stratified=include_stratified,
            )
            save_eval_metrics(payload, out_path)
            n_done += 1
        except FileNotFoundError as e:
            logger.warning("[AE-EVAL] Missing file for %s: %s", u.id, e)
        except Exception as e:
            logger.exception("[AE-EVAL] Failed for %s: %s", u.id, e)

    print(
        f"AE eval done. wrote={n_done}, skipped_existing={n_skipped}, missing_model={n_missing_model}"
    )


if __name__ == "__main__":
    app()
