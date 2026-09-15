from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def compute_presto_variance_from_metrics_table(
    df: pd.DataFrame,
    *,
    homology_dims: Iterable[int] = (0, 1, 2),
) -> float:

    dims = list(homology_dims)

    if df.empty:
        raise ValueError("metrics table is empty.")

    cols = [f"l2_dim{d}" for d in dims]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for PRESTO variance: {missing}")

    # Separate only the homology-specific L2 norm columns per universe.
    X = df[cols].to_numpy(dtype=float)

    # Treat non-finite as 0.0
    if any(~np.isfinite(X.flatten())):
        finite_mask = np.isfinite(X).all(axis=1)
        X = X[finite_mask]
        logger.warning(
            "[TDA] Non-finite L2 norms found; Dropping these rows for PRESTO variance. Potential data issue."
        )

    # Number of universes
    N = X.shape[0]
    if N == 0:
        raise ValueError("No rows available for PRESTO variance.")

    # Compute mean L2 norm per dim across universes
    mu = X.mean(axis=0)

    sse = ((X - mu) ** 2).sum()
    return float(sse / N)


def build_metrics_table(
    universes: Iterable[Universe],
    split: str = "test",
    require_exists: bool = True,
    homology_dims: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """
    Build a dataframe from all universe-level metric files that result from the pipeline.
    """
    rows: list[dict[str, Any]] = []

    for u in universes:
        path = u.paths.metrics(split=split)
        if require_exists and not path.exists():
            logger.warning("[METRICS] Missing metrics path for universe: %s", u.id)
            continue

        try:
            payload = u.io.load_metrics(split=split)

        except FileNotFoundError as e:
            logger.warning("[METRICS] Skipping %s: metrics file missing: %s", u.id, e)
            continue

        except Exception as e:
            logger.warning(
                "[METRICS] Skipping %s: failed to load metrics: %s: %s",
                u.id,
                type(e).__name__,
                e,
            )
            continue

        if not payload:
            logger.warning("[METRICS] Skipping %s: metrics JSON is empty", u.id)
            continue

        row = u.to_param_dict()

        # LIKELY REDUNDANT: This data is already covered by the line above.
        row["universe_id"] = u.id
        row["split"] = split

        # Why do we need the metrics path here again?
        row["metrics_path"] = str(path)

        # Retrieve metrics stored as json dicts
        l2_raw = payload.get("landscape_norms", {}) or {}
        tp_raw = payload.get("total_persistence", {}) or {}

        l2 = {int(k): float(v) for k, v in l2_raw.items()}
        tp = {int(k): float(v) for k, v in tp_raw.items()}

        l2_vals: list[float] = []

        for d in homology_dims:
            v = float(l2.get(d, 0.0))
            row[f"l2_dim{d}"] = v
            row[f"tp_dim{d}"] = float(tp.get(d, 0.0))
            l2_vals.append(v)

        # Sum and mean across dimensions.
        row["l2_aggregate"] = float(sum(l2_vals))
        row["l2_average"] = float(sum(l2_vals) / max(len(l2_vals), 1))

        row["h0_total_persistence_euclidean"] = float(payload.get("h0_total_persistence_euclidean", 0.0))
        row["mst_length"] = float(payload.get("mst_length", 0.0))

        # Load evaluation metrics if available
        eval_path = u.paths.eval_metrics(split=split)
        if eval_path.exists():
            try:
                with eval_path.open("r", encoding="utf-8") as f:
                    eval_payload = json.load(f) or {}

                assert eval_payload.get("split") == split, "Split mismatch in eval metrics file."
                
                row["rocauc"] = eval_payload.get("roc_auc")

                # Combined reconstruction error
                recon = eval_payload.get("reconstruction", {}) or {}
                row["recon_n"] = recon.get("n")
                row["recon_mse_mean"] = recon.get("mean")
                row["recon_mse_median"] = recon.get("median")
                row["recon_mse_std"] = recon.get("std")
                row["recon_mse_p95"] = recon.get("p95")
                
                # Benign reconstruction error
                benign = eval_payload.get("benign")
                row["benign_n"] = benign.get("n")
                row["benign_mse_mean"] = benign.get("mean") 
                row["benign_mse_median"] = benign.get("median")
                row["benign_mse_std"] = benign.get("std")
                row["benign_mse_p95"] = benign.get("p95")

                # attack reconstruction error
                attack = eval_payload.get("attack")
                row["attack_n"] = attack.get("n")
                row["attack_mse_mean"] = attack.get("mean")
                row["attack_mse_median"] = attack.get("median")
                row["attack_mse_std"] = attack.get("std")
                row["attack_mse_p95"] = attack.get("p95")


            except Exception as e:
                logger.warning(
                    "[METRICS] Failed to load eval metrics for %s: %s", u.id, e
                )

        rows.append(row)

    if not rows:
        raise RuntimeError("No metrics files found for split = %s.", split)

    return pd.DataFrame(rows)
