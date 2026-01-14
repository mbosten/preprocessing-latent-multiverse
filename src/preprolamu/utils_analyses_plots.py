# src/reprolamu/utils_analyses_plots.py
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import typer
from scipy.stats import permutation_test, spearmanr

logger = logging.getLogger(__name__)


# Keep only universes with metrics_status == 'ok'
def _ok_only(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Dropping %d universes with metrics_status != 'ok'",
        len(df) - df[df["metrics_status"] == "ok"].shape[0],
    )
    return df[df["metrics_status"] == "ok"].copy()


# Format a float in scientific notation with 2 significant digits, or empty string if NaN.
def _format_sci(x: float) -> str:
    if pd.isna(x):
        return ""
    # 2 significant digits in scientific notation
    return f"{x:.2e}"


# Cap the L2 norm dataframe to exclude norms above a certain threshold.
def filter_by_norm_threshold(
    df: pd.DataFrame,
    *,
    threshold: float | None,
) -> pd.DataFrame:
    """
    Keep only universes whose landscape L2 norms are <= threshold
    for *all* available homology dimensions (l2_dim* columns).

    If threshold is None: return df unchanged.
    """
    if threshold is None:
        return df

    if "l2_average" not in df.columns:
        raise typer.BadParameter(
            "Requested norm threshold filtering, but 'l2_average' column does not exist in the table."
        )

    before = len(df)

    mask = df["l2_average"].notna() & (df["l2_average"] <= threshold)
    df2 = df[mask].copy()

    logger.info(
        "Applied norm threshold on l2_average: kept %d/%d where l2_average <= %.6g (dropped=%d)",
        len(df2),
        before,
        threshold,
        before - len(df2),
    )
    return df2


# Drop L2 norm rows that are all exactly zero across dimensions.
def filter_exclude_zero_norms(df: pd.DataFrame, *, exclude_zero: bool) -> pd.DataFrame:
    """
    Optionally drop universes whose landscape L2 norms are all exactly zero
    across all available homology dimensions (l2_dim* columns).

    If exclude_zero is False: return df unchanged.
    """
    if not exclude_zero:
        return df

    dim_cols = sorted([c for c in df.columns if c.startswith("l2_dim")])
    if not dim_cols:
        raise typer.BadParameter(
            "Requested exclude_zero filtering, but no 'l2_dim*' columns exist in the table."
        )

    before = len(df)

    # Drop rows where ALL dimension norms are exactly 0.0
    # (NaNs do not trigger dropping.)
    is_all_zero = (df[dim_cols] == 0).all(axis=1)

    df2 = df[~is_all_zero].copy()

    logger.info(
        "Excluded all-zero norms across dims: kept %d/%d (dropped=%d) using cols=%s",
        len(df2),
        before,
        before - len(df2),
        ",".join(dim_cols),
    )
    return df2


def _parse_split_by(spec: str) -> list[str]:
    """
    Parse --split-by "dataset,scaling" into list of 1-2 normalized keys.
    Allowed keys map to columns in the metrics table.
    """
    spec = (spec or "").strip()
    if not spec:
        return []

    keys = [k.strip().lower() for k in spec.split(",") if k.strip()]
    if len(keys) > 2:
        raise typer.BadParameter(
            "--split-by supports at most 2 keys (e.g., 'dataset' or 'dataset,scaling')."
        )

    alias = {
        "dataset": "dataset_id",
        "dataset_id": "dataset_id",
        "scaling": "scaling",
        "feature_subset": "feature_subset",
        "features": "feature_subset",
        "log_transform": "log_transform",
        "log": "log_transform",
        "duplicate_handling": "duplicate_handling",
        "duplicates": "duplicate_handling",
        "missingness": "missingness",
    }
    out = []
    for k in keys:
        if k not in alias:
            raise typer.BadParameter(
                f"Unknown split key {k!r}. Allowed: dataset, scaling, feature_subset, log_transform, duplicate_handling, missingness."
            )
        out.append(alias[k])
    return out


def spearmanr_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:

    rs = spearmanr(x, y).statistic

    # Permutation test on Spearman's rs directly (simplest + matches intent).
    def spearmanr_statistic(x_perm):
        return spearmanr(x_perm, y).statistic

    res = permutation_test(
        (x,),
        spearmanr_statistic,
        alternative="two-sided",
        permutation_type="pairings",
        n_resamples=50000,
        random_state=0,
    )
    return float(rs), float(res.pvalue)


_L2_DIMS = (0, 1, 2)


def _get_l2_dim_cols(df: pd.DataFrame) -> list[str]:
    cols = [f"l2_dim{d}" for d in _L2_DIMS if f"l2_dim{d}" in df.columns]
    if not cols:
        raise typer.BadParameter("No l2_dim* columns found in metrics table.")
    return cols


def _subset_excluded_universes(
    df: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Deviant universes are those that are normally excluded:
      A) any l2_dim* > threshold
      OR
      B) all l2_dim* == 0

    Returns: (excluded_df, meta)
    """
    df = _ok_only(df)
    dim_cols = _get_l2_dim_cols(df)

    X = df[dim_cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)

    outlier_mask = np.nanmax(X, axis=1) > threshold
    all_zero_mask = np.all(X == 0.0, axis=1) & np.all(np.isfinite(X), axis=1)

    excluded_mask = outlier_mask | all_zero_mask

    excluded = df.loc[excluded_mask].copy()
    breakpoint()
    excluded["excluded_reason"] = np.where(
        outlier_mask & all_zero_mask,
        "both",
        np.where(outlier_mask, f"outlier_norm_gt_{threshold:g}", "all_zero_norms"),
    )

    meta = {
        "n_total_ok": int(len(df)),
        "n_excluded": int(len(excluded)),
        "n_outlier": int(outlier_mask.sum()),
        "n_all_zero": int(all_zero_mask.sum()),
        "threshold": float(threshold),
        "dim_cols": dim_cols,
    }
    return excluded, meta


_PRESTO_PARAMS: list[str] = [
    "scaling",
    "log_transform",
    "feature_subset",
    "duplicate_handling",
    "missingness",
    "seed",
]


def _print_excluded_param_overview(excluded: pd.DataFrame) -> pd.DataFrame:
    """
    Prints (and returns) a long-form overview table:
      dataset_id, param, value, count, pct
    """
    if excluded.empty:
        print("\nNo excluded universes.")
        return pd.DataFrame(columns=["dataset_id", "param", "value", "count", "pct"])

    params_present = [
        c for c in (["dataset_id"] + _PRESTO_PARAMS) if c in excluded.columns
    ]
    datasets = (
        sorted(excluded["dataset_id"].dropna().unique().tolist())
        if "dataset_id" in excluded.columns
        else ["ALL"]
    )

    rows: list[dict] = []
    for ds in datasets:
        sdf = excluded if ds == "ALL" else excluded[excluded["dataset_id"] == ds]
        if sdf.empty:
            continue
        for p in params_present:
            if p == "dataset_id":
                continue
            vc = sdf[p].astype("object").value_counts(dropna=False)
            for val, cnt in vc.items():
                rows.append(
                    {
                        "dataset_id": ds,
                        "param": p,
                        "value": val,
                        "count": int(cnt),
                        "pct": float(cnt / max(len(sdf), 1)),
                    }
                )

    overview = pd.DataFrame(rows).sort_values(
        ["dataset_id", "param", "count"], ascending=[True, True, False]
    )

    print("\n" + "=" * 90)
    print("Deviant universes: parameter-value counts (per dataset)")
    print("=" * 90)
    for ds in overview["dataset_id"].unique().tolist():
        print("\n" + "-" * 90)
        print(f"Dataset: {ds}")
        sub = overview[overview["dataset_id"] == ds].copy()
        for p in sub["param"].unique().tolist():
            s2 = sub[sub["param"] == p].copy()
            # show all values for this param
            s2["pct"] = (s2["pct"] * 100.0).round(1)
            print(f"\n{p}:")
            print(s2[["value", "count", "pct"]].to_string(index=False))

    return overview
