# src/reprolamu/utils_analyses_plots.py
from __future__ import annotations

import logging

import pandas as pd
import typer

logger = logging.getLogger(__name__)


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

    # Find all l2_dim{d} columns present
    dim_cols = sorted([c for c in df.columns if c.startswith("l2_dim")])
    if not dim_cols:
        raise typer.BadParameter(
            "Requested norm threshold filtering, but no 'l2_dim*' columns exist in the table."
        )

    before = len(df)

    # Keep universes where every dimension norm is <= threshold (NaNs treated as fail-safe drop)
    mask = pd.Series(True, index=df.index)
    for c in dim_cols:
        mask &= df[c].notna() & (df[c] <= threshold)

    df2 = df[mask].copy()

    logger.info(
        "Applied norm threshold across dims: kept %d/%d where max(%s) <= %.6g (dropped=%d)",
        len(df2),
        before,
        ",".join(dim_cols),
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
