from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from preprolamu.config import load_dataset_config
from preprolamu.pipeline.autoencoder import _get_device, load_autoencoder_for_universe
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


def _split_to_path(universe: Universe, split: str):
    if split == "train":
        return universe.preprocessed_train_path()
    if split in {"val", "validation"}:
        return universe.preprocessed_validation_path()
    if split == "test":
        return universe.preprocessed_test_path()
    raise ValueError("split must be 'train', 'val'/'validation', or 'test'")


def _labels_from_df(df: pd.DataFrame, label_col: str) -> Optional[np.ndarray]:
    if label_col not in df.columns:
        return None
    y = df[label_col].astype(str).to_numpy()
    return y


def _feature_matrix_from_df(df: pd.DataFrame, label_col: str) -> np.ndarray:
    cols_to_drop = [label_col]
    if "Label" in df.columns and "Label" not in cols_to_drop:
        cols_to_drop.append("Label")

    df_features = df.drop(columns=cols_to_drop, errors="ignore")
    X = df_features.to_numpy(dtype=np.float32)
    return X


def _recon_error_per_sample(
    model,
    X: np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    device = _get_device()
    model = model.to(device)
    model.eval()

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    errs: list[np.ndarray] = []

    with torch.no_grad():
        for (bx,) in dl:
            bx = bx.to(device)
            xhat = model(bx)

            # per-sample MSE across features (batch,)
            e = torch.mean((xhat - bx) ** 2, dim=1).detach().cpu().numpy()
            errs.append(e)

    return np.concatenate(errs, axis=0) if errs else np.array([], dtype=float)


def _summarize_errors(err: np.ndarray) -> Dict[str, Any]:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]

    if err.size == 0:
        return {
            "n": 0,
            "mse_mean": np.nan,
            "mse_median": np.nan,
            "mse_std": np.nan,
            "mse_min": np.nan,
            "mse_max": np.nan,
            "mse_p90": np.nan,
            "mse_p95": np.nan,
            "mse_p99": np.nan,
        }

    return {
        "n": int(err.size),
        "mse_mean": float(np.mean(err)),
        "mse_median": float(np.median(err)),
        "mse_std": float(np.std(err)),
        "mse_min": float(np.min(err)),
        "mse_max": float(np.max(err)),
        "mse_p90": float(np.quantile(err, 0.90)),
        "mse_p95": float(np.quantile(err, 0.95)),
        "mse_p99": float(np.quantile(err, 0.99)),
    }


def evaluate_autoencoder_reconstruction(
    universe: Universe,
    *,
    split: str = "test",
    batch_size: int = 2048,
    include_stratified: bool = True,
) -> Dict[str, Any]:
    ds_cfg = load_dataset_config(universe.dataset_id)
    label_col = ds_cfg.label_column

    path = _split_to_path(universe, split)
    df = pd.read_parquet(path)

    y = _labels_from_df(df, label_col=label_col)
    X = _feature_matrix_from_df(df, label_col=label_col)

    # Load trained AE
    model = load_autoencoder_for_universe(universe, ds_cfg)

    # Compute per-sample recon MSE
    err = _recon_error_per_sample(model, X, batch_size=batch_size)

    out: Dict[str, Any] = {
        "universe_id": universe.id,
        "dataset_id": universe.dataset_id,
        "split": split,
        "label_column": label_col,
        "recon": _summarize_errors(err),
    }

    if include_stratified and (y is not None) and (len(y) == len(err)):
        benign_mask = y == "Benign"
        attack_mask = ~benign_mask

        out["recon_benign"] = _summarize_errors(err[benign_mask])
        out["recon_attack"] = _summarize_errors(err[attack_mask])

        out["n_benign"] = int(benign_mask.sum())
        out["n_attack"] = int(attack_mask.sum())

    return out


# Single use function that has been integrated in the ae_eval function itself.
# def save_eval_metrics(payload: Dict[str, Any], path) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, indent=2)
