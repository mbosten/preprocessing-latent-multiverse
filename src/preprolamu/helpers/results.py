from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_output_by_norm_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Filter the landscape norm table to exclude universes above a certain threshold.
    Can be used in analyses when dealing with outliers.
    Filtering takes place on the average L2 across all homology dimensions.
    """

    if "l2_average" not in df.columns:
        logger.warning(
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


def exclude_zero_norms_from_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude universes where the average L2 norm across dimensions is zero.
    These universes typically point towards issues earlier in the pipeline and
    are not informative for analyses or plotting. Ensure proper handling of these cases.
    """

    dim_cols = sorted([c for c in df.columns if c.startswith("l2_dim")])
    if not dim_cols:
        logger.warning(
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
