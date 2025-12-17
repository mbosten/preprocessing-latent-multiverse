# src/preprolamu/pipeline/autoencoder.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprolamu.config import DatasetConfig, load_dataset_config
from preprolamu.io.storage import (
    ensure_parent_dir,
    get_ae_model_path,
    get_preprocessed_train_path,
)
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Defines an autoencoder with adjustable architecture that is retrieved from the Universe class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_p = dropout

        layers = []
        prev_dim = input_dim

        # Build encoder
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        dec_layers = []
        prev_dim = latent_dim

        # Build decoder
        for hdim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, hdim))
            dec_layers.append(nn.ReLU())
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# Check for GPU support
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


#
def _get_feature_matrix_for_ae(df: pd.DataFrame, ds_cfg: DatasetConfig) -> np.ndarray:
    df_features = df.copy()
    cols_to_drop = [ds_cfg.label_column]

    if "Attack" in df_features.columns and "Attack" not in cols_to_drop:
        cols_to_drop.append("Attack")

    logger.info(
        f"[AE] Dropping columns for AE feature matrix (if present): {cols_to_drop}"
    )
    df_features = df_features.drop(columns=cols_to_drop, errors="ignore")

    X = df_features.to_numpy(dtype=np.float32)
    logger.info(f"[AE] Feature matrix shape for AE: {X.shape}")

    return X


def get_feature_matrix_from_universe(universe: Universe):
    ds_cfg: DatasetConfig = load_dataset_config(universe.dataset_id)
    preprocessed_train_path = get_preprocessed_train_path(universe)
    logger.info("[AE] Loading preprocessed data from %s", preprocessed_train_path)
    df = pd.read_parquet(preprocessed_train_path)
    logger.info("[AE] Preprocessed data shape: %s", df.shape)
    logger.info("[AE] Data columns: %s", df.columns.tolist())
    X = _get_feature_matrix_for_ae(df, ds_cfg)

    return X, ds_cfg


def train_autoencoder_for_universe(universe: Universe) -> Path:
    """
    Train an autoencoder on the preprocessed data for this universe.
    Also saves a checkpoint to get_ae_model_path(universe) and returns that path.
    """
    logger.info(f"[AE] Training autoencoder for universe = {universe.to_id_string()}")

    X, ds_cfg = get_feature_matrix_from_universe(universe)

    ae_cfg = ds_cfg.autoencoder

    input_dim = X.shape[1]

    device = _get_device()

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=ae_cfg.hidden_dims,
        latent_dim=ae_cfg.latent_dim,
        dropout=ae_cfg.dropout,
    ).to(device)

    tensor_X = torch.from_numpy(X)
    dataset = TensorDataset(tensor_X)
    loader = DataLoader(
        dataset, batch_size=ae_cfg.batch_size, shuffle=True, drop_last=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=ae_cfg.regularization
    )

    logger.info(
        "[AE] Starting training: epochs=%d, batch_size=%d, input_dim=%d, latent_dim=%d, hidden_dims=%s, dropout=%.4f, regularization=%.6f",
        ae_cfg.epochs,
        ae_cfg.batch_size,
        input_dim,
        ae_cfg.latent_dim,
        ae_cfg.hidden_dims,
        ae_cfg.dropout,
        ae_cfg.regularization,
    )

    model.train()
    for epoch in range(1, ae_cfg.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            recon_X = model(batch_X)
            loss = criterion(recon_X, batch_X)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"[AE] Epoch {epoch}/{ae_cfg.epochs}, Loss: {avg_loss:.6f}")

    model_path = get_ae_model_path(universe)
    ensure_parent_dir(model_path)

    checkpoint = {
        "input_dim": input_dim,
        "hidden_dims": list(ae_cfg.hidden_dims),
        "latent_dim": ae_cfg.latent_dim,
        "dropout": ae_cfg.dropout,
        "ae_regularization": ae_cfg.regularization,
        "model_state_dict": model.state_dict(),
        "universe_id": universe.to_id_string(),
    }
    torch.save(checkpoint, model_path)
    logger.info(f"[AE] Saved AE checkpoint to {model_path}")

    return model_path


def load_autoencoder_for_universe(
    universe: Universe, ds_cfg: DatasetConfig
) -> Autoencoder:
    """
    Load the trained autoencoder model for this universe from its checkpoint.
    """
    model_path = get_ae_model_path(universe)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Autoencoder model checkpoint not found at {model_path}"
        )
    ae_cfg = ds_cfg.autoencoder
    checkpoint = torch.load(model_path, map_location=_get_device())
    input_dim = int(checkpoint["input_dim"])
    hidden_dims = tuple(int(h) for h in checkpoint["hidden_dims"])
    latent_dim = int(checkpoint["latent_dim"])
    dropout = float(checkpoint.get("dropout", ae_cfg.dropout))

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "[AE] Loaded AE model from %s (input_dim=%d, latent_dim=%d, hidden_dims=%s, dropout=%.4f)",
        model_path,
        input_dim,
        latent_dim,
        hidden_dims,
        dropout,
    )

    return model
