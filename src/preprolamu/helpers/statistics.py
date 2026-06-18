from __future__ import annotations

import logging

import numpy as np
from scipy.stats import permutation_test, spearmanr

logger = logging.getLogger(__name__)


def spearmanr_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Permutation test of Spearman's rank"""
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
