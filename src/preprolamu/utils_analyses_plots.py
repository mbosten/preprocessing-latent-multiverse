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


# Format a float in scientific notation with 2 significant digits, or empty string if nan.
def _format_sci(x: float) -> str:
    if pd.isna(x):
        return ""
    # 2 significant digits in scientific notation
    return f"{x:.2e}"


# Cap the L2 norm dataframe to exclude norms above a certain threshold.
# Caps average l2 across homology dimensions.
def filter_by_norm_threshold(
    df: pd.DataFrame,
    *,
    threshold: float | None,
) -> pd.DataFrame:

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

    if not exclude_zero:
        return df

    dim_cols = sorted([c for c in df.columns if c.startswith("l2_dim")])
    if not dim_cols:
        raise typer.BadParameter(
            "Requested exclude_zero filtering, but no 'l2_dim*' columns exist in the table."
        )

    before = len(df)

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


# used for splitting the data by two parameters in some analyses functions
def _parse_split_by(spec: str) -> list[str]:

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


# permutation test for Spearman's rank-order correlation (as is advised)
def spearmanr_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:

    rs = spearmanr(x, y).statistic

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
