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
from preprolamu.utils_analyses_plots import (
    _ok_only,
    _parse_split_by,
    _print_excluded_param_overview,
    _subset_excluded_universes,
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


# Variance, global, local and stability functions
_PRESTO_PARAMS: list[str] = [
    "scaling",
    "log_transform",
    "feature_subset",
    "duplicate_handling",
    "missingness",
    "seed",
]


# local sensitivity
def _presto_local_sensitivity_from_metrics_table(
    df_ds: pd.DataFrame,
    *,
    param: str,
    homology_dims: tuple[int, ...] = (0, 1, 2),
) -> dict:

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


# global sensitivity
def _presto_global_sensitivity_from_metrics_table(
    df_ds: pd.DataFrame,
    *,
    params: list[str] | None = None,
    homology_dims: tuple[int, ...] = (0, 1, 2),
) -> dict:
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


def _individual_sensitivity_table_for_param(
    df: pd.DataFrame,
    *,
    param: str,
    homology_dims=(0, 1, 2),
) -> pd.DataFrame:
    """
    Build a table with one row per equivalence class Q for `param`.
    Each equivalence class is defined by fixing all other _PRESTO_PARAMS (and dataset_id).
    """
    if param not in df.columns:
        raise ValueError(f"Param {param!r} not in df columns.")

    # Define context = all OTHER params (plus dataset_id for safety)
    fixed_cols = ["dataset_id"] + [
        c for c in _PRESTO_PARAMS if c != param and c in df.columns
    ]

    if df.empty:
        return pd.DataFrame(
            columns=fixed_cols + ["class_size", "pv", "individual_sensitivity"]
        )

    grouped = df.groupby(fixed_cols, dropna=False)

    rows = []
    for fixed_key, g in grouped:
        if not isinstance(fixed_key, tuple):
            fixed_key = (fixed_key,)

        m = int(len(g))
        if m < 2:
            pv = 0.0
            ind = 0.0
        else:
            pv = float(
                compute_presto_variance_from_metrics_table(
                    g, homology_dims=homology_dims
                )
            )
            ind = float(np.sqrt(max(pv, 0.0)))

        row = {col: val for col, val in zip(fixed_cols, fixed_key)}
        row["class_size"] = m
        row["pv"] = pv
        row["individual_sensitivity"] = ind
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


# stability regions
def _value_enrichment_table(
    stable_df: pd.DataFrame,
    unstable_df: pd.DataFrame,
    col: str,
) -> pd.DataFrame:
    """
    Compare distributions of a fixed parameter value in stable vs unstable regions.
    Returns a small table with proportions and difference.
    """
    s = stable_df[col].astype("object").value_counts(normalize=True, dropna=False)
    u = unstable_df[col].astype("object").value_counts(normalize=True, dropna=False)
    out = pd.concat(
        [s.rename("stable_prop"), u.rename("unstable_prop")], axis=1
    ).fillna(0.0)
    out["diff_unstable_minus_stable"] = out["unstable_prop"] - out["stable_prop"]
    return out.sort_values("diff_unstable_minus_stable", ascending=False)


# stability regions
def _print_top_contexts_with_universes(
    *,
    label: str,
    top_ctx: pd.DataFrame,
    fixed_cols: list[str],
):
    print(f"\n{label} contexts (equivalence classes):")
    cols_show = fixed_cols + ["class_size", "individual_sensitivity"]
    cols_show = [c for c in cols_show if c in top_ctx.columns]
    print(top_ctx[cols_show].to_string(index=False))


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
    param: Annotated[
        Literal[
            "l2_dim0",
            "l2_dim1",
            "l2_dim2",
            "l2_average",
        ],
        typer.Option(
            "--param",
            help="For which homology dimension should L2-norms be compared across datasets?",
            show_choices=True,
        ),
    ] = "l2_average",
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
        df.groupby("dataset_id")[param]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())


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


@app.command("presto-stability-regions")
def presto_stability_regions(
    split: str = typer.Option("test"),
    param: str = typer.Option(
        "all",
        help="One of: scaling, log_transform, feature_subset, duplicate_handling, missingness, seed, or 'all'.",
    ),
    q_low: float = typer.Option(
        0.2, help="Lower quantile for 'stable' equivalence classes."
    ),
    q_high: float = typer.Option(
        0.8, help="Upper quantile for 'unstable' equivalence classes."
    ),
    top_k: int = typer.Option(
        10, help="How many stable/unstable contexts to print per dataset+param."
    ),
    norm_threshold: float | None = typer.Option(
        None,
        help="Exclude universes where any l2_dim* exceeds this threshold (optional).",
    ),
    exclude_zero_norms: bool = typer.Option(
        False, help="Exclude universes where all l2_dim* norms are exactly zero."
    ),
    drop_nonvarying_params: bool = typer.Option(
        True,
        help="If True, skip params with <2 unique values within a dataset (e.g. collapsed missingness).",
    ),
):
    """
    Stable vs unstable regions via *individual PRESTO sensitivity* (sqrt(PV) per equivalence class).
    Prints BOTH top stable and top unstable contexts per dataset + param,
    and prints the universes contained in each context.
    """
    if not (0.0 < q_low < q_high < 1.0):
        raise typer.BadParameter("Require 0 < q_low < q_high < 1.")

    universes = generate_multiverse()
    df = _ok_only(build_metrics_table(universes, split=split, require_exists=True))

    df = filter_by_norm_threshold(df, threshold=norm_threshold)
    df = filter_exclude_zero_norms(df, exclude_zero=exclude_zero_norms)

    # Determine params to run
    if param != "all":
        if param not in _PRESTO_PARAMS:
            raise typer.BadParameter(f"param must be one of {_PRESTO_PARAMS} or 'all'.")
        params_to_run = [param]
    else:
        params_to_run = _PRESTO_PARAMS.copy()

    datasets = sorted(df["dataset_id"].dropna().unique().tolist())
    if not datasets:
        raise typer.BadParameter("No datasets available after filtering.")

    for ds in datasets:
        ds_df = df[df["dataset_id"] == ds].copy()
        if ds_df.empty:
            continue

        print("\n" + "=" * 100)
        print(f"DATASET: {ds} | split={split} | n_universes={len(ds_df)}")
        print("=" * 100)

        for p in params_to_run:
            if p not in ds_df.columns:
                continue

            # Skip collapsed params (e.g. missingness in datasets without missing values)
            if drop_nonvarying_params and ds_df[p].nunique(dropna=False) < 2:
                print(f"\n[param={p}] skipped (non-varying within dataset; nunique<2).")
                continue

            # One row per equivalence class (context defined by all OTHER params)
            ind = _individual_sensitivity_table_for_param(
                ds_df, param=p, homology_dims=(0, 1, 2)
            )

            # Drop singleton equivalence classes (uninformative: sensitivity forced to 0)
            before_classes = len(ind)
            ind = ind[ind["class_size"] >= 2].copy()
            dropped = before_classes - len(ind)
            if dropped > 0:
                logger.info(
                    "[stability-regions] Dropped %d/%d singleton equivalence classes for ds=%s param=%s",
                    dropped,
                    before_classes,
                    ds,
                    p,
                )

            if ind.empty:
                print(f"\n[param={p}] no equivalence classes after filtering.")
                continue

            x = ind["individual_sensitivity"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                print(f"\n[param={p}] no finite sensitivity values.")
                continue

            lo = float(np.quantile(x, q_low))
            hi = float(np.quantile(x, q_high))

            stable = ind[ind["individual_sensitivity"] <= lo].copy()
            unstable = ind[ind["individual_sensitivity"] >= hi].copy()

            top_stable = stable.sort_values(
                "individual_sensitivity", ascending=True
            ).head(top_k)
            top_unstable = unstable.sort_values(
                "individual_sensitivity", ascending=False
            ).head(top_k)

            print("\n" + "-" * 100)
            print(f"param={p}")
            print(
                f"equiv_classes={len(ind)} | "
                f"stable<=Q{int(q_low*100)} (thr={lo:.6g}) -> {len(stable)} | "
                f"unstable>=Q{int(q_high*100)} (thr={hi:.6g}) -> {len(unstable)}"
            )

            # These columns define the context and are used to retrieve universes
            # fixed_cols = _fixed_cols_for_param(ds_df, param=p)
            fixed_cols = ["dataset_id"] + [
                c for c in _PRESTO_PARAMS if c != p and c in ds_df.columns
            ]

            # Print top unstable + universes
            if not top_unstable.empty:
                _print_top_contexts_with_universes(
                    label="Top UNSTABLE",
                    top_ctx=top_unstable,
                    fixed_cols=fixed_cols,
                )
            else:
                print("\nTop UNSTABLE contexts: none (after filtering/thresholding).")

            # Print top stable + universes
            if not top_stable.empty:
                _print_top_contexts_with_universes(
                    label="Top STABLE",
                    top_ctx=top_stable,
                    fixed_cols=fixed_cols,
                )
            else:
                print("\nTop STABLE contexts: none (after filtering/thresholding).")

            # Optional quick “what characterizes instability?” enrichment table
            # (kept minimal: only show for params with >=2 unique values)
            fixed_cols_no_ds = [c for c in fixed_cols if c != "dataset_id"]
            if fixed_cols_no_ds and len(stable) > 0 and len(unstable) > 0:
                print(
                    "\nEnrichment (unstable minus stable) by fixed setting (top 5 each):"
                )
                for c in fixed_cols_no_ds:
                    if (
                        stable[c].nunique(dropna=False) < 2
                        and unstable[c].nunique(dropna=False) < 2
                    ):
                        continue
                    enr = _value_enrichment_table(stable, unstable, c)
                    print(f"\n  {c}:")
                    print(enr.head(5).to_string())


# analyze excluded universes
@app.command("excluded-universes")
def excluded_universes(
    split: str = typer.Option("test", help="train/val/test"),
    threshold: float = typer.Option(
        100.0,
        help="Outlier threshold: flag universe if any l2_dim* > threshold.",
    ),
    preview_rows: int = typer.Option(25, help="How many excluded rows to preview."),
    save_csv: bool = typer.Option(
        False, help="Save excluded subset + overview CSV files."
    ),
    out_dir: Path = typer.Option(
        Path("data/processed/analysis"),
        help="Output directory for CSV exports (if --save-csv).",
    ),
):
    """
    Analyze universes you normally exclude:
      - any l2_dim* > threshold
      - OR all l2_dim* == 0

    Prints excluded subset preview + parameter distribution tables (per dataset).
    Optionally saves CSV exports.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    universes = generate_multiverse()
    df = build_metrics_table(universes, split=split, require_exists=True)
    df = _ok_only(df)

    excluded, meta = _subset_excluded_universes(df, threshold=threshold)

    print("\n" + "=" * 90)
    print("Excluded universes extraction summary")
    print("=" * 90)
    for k, v in meta.items():
        print(f"{k}: {v}")

    if excluded.empty:
        print("\nNo excluded universes found.")
        return

    # Preview subset
    show_cols = [
        c
        for c in (
            ["universe_id", "dataset_id", "excluded_reason"]
            + _PRESTO_PARAMS
            + [f"l2_dim{d}" for d in (0, 1, 2)]
            + ["l2_average"]
        )
        if c in excluded.columns
    ]
    print("\n" + "=" * 90)
    print(f"Deviant universes (first {preview_rows} rows)")
    print("=" * 90)
    print(excluded[show_cols].head(preview_rows).to_string(index=False))

    # Counts by dataset + reason
    if "dataset_id" in excluded.columns:
        print("\nCounts by dataset:")
        print(excluded["dataset_id"].value_counts(dropna=False).to_string())
    if "excluded_reason" in excluded.columns:
        print("\nCounts by excluded reason:")
        print(excluded["excluded_reason"].value_counts(dropna=False).to_string())

    # Parameter overview tables
    overview = _print_excluded_param_overview(excluded)

    if save_csv:
        subset_path = (
            out_dir / f"excluded_universes_split-{split}_thr-{threshold:g}.csv"
        )
        excluded[show_cols].to_csv(subset_path, index=False)
        logger.info("Saved excluded subset to %s", subset_path)

        overview_path = (
            out_dir / f"excluded_param_overview_split-{split}_thr-{threshold:g}.csv"
        )
        overview.to_csv(overview_path, index=False)
        logger.info("Saved excluded overview to %s", overview_path)


if __name__ == "__main__":
    app()
