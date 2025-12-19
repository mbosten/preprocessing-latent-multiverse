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
from preprolamu.io.storage import ensure_parent_dir
from preprolamu.pipeline.universes import Universe
from preprolamu.tests.data_checks import log_feature_stats

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
        logger.info("[TORCH] Resetting peak memory stats.")
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


#
def _get_feature_matrix_for_ae(df: pd.DataFrame, ds_cfg: DatasetConfig) -> np.ndarray:
    df_features = df.copy()
    cols_to_drop = [ds_cfg.label_column]

    # Specific for the three NF datasets
    if "Label" in df_features.columns and "Label" not in cols_to_drop:
        cols_to_drop.append("Label")

    logger.info(
        f"[AE] Dropping columns for AE feature matrix (if present): {cols_to_drop}"
    )
    df_features = df_features.drop(columns=cols_to_drop, errors="ignore")
    feature_names = df_features.columns.tolist()

    X = df_features.to_numpy(dtype=np.float32)
    logger.info(f"[AE] Feature matrix shape for AE: {X.shape}")

    return X, feature_names


def get_feature_matrix_from_universe(
    universe: Universe, split: str = "train"
) -> Tuple[np.ndarray, np.ndarray, DatasetConfig]:

    ds_cfg: DatasetConfig = load_dataset_config(universe.dataset_id)

    if split == "train":
        path = universe.preprocessed_train_path()
    elif split == "val":
        path = universe.preprocessed_validation_path()
    elif split == "test":
        path = universe.preprocessed_test_path()
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    logger.info("[AE] Loading preprocessed %s data from %s", split.upper(), path)

    df = pd.read_parquet(path)
    logger.info("[AE] %s data shape: %s", split.upper(), df.shape)

    X, feature_names = _get_feature_matrix_for_ae(df, ds_cfg)
    return X, feature_names, ds_cfg


def train_autoencoder_for_universe(universe: Universe) -> Path:
    """
    Train an autoencoder on the preprocessed data for this universe.
    Also saves a checkpoint to universe.ae_model_path() and returns that path.
    """
    logger.info(f"[AE] Training autoencoder for universe = {universe.id}")

    X_train, feature_names, ds_cfg = get_feature_matrix_from_universe(
        universe, split="train"
    )

    log_feature_stats(X_train, feature_names, "train", universe)

    X_val, _, _ = get_feature_matrix_from_universe(universe, split="val")

    ae_cfg = ds_cfg.autoencoder

    input_dim = X_train.shape[1]
    device = _get_device()

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=ae_cfg.hidden_dims,
        latent_dim=ae_cfg.latent_dim,
        dropout=ae_cfg.dropout,
    ).to(device)

    tensor_X_train = torch.from_numpy(X_train)
    tensor_X_val = torch.from_numpy(X_val)

    train_dataset = TensorDataset(tensor_X_train)
    val_dataset = TensorDataset(tensor_X_val)

    train_loader = DataLoader(
        train_dataset, batch_size=ae_cfg.batch_size, shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=ae_cfg.batch_size, shuffle=False, drop_last=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=ae_cfg.regularization
    )

    max_epochs = ae_cfg.epochs
    patience = getattr(ae_cfg, "patience", 5)
    min_delta = getattr(ae_cfg, "min_delta", 1e-4)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    logger.info(
        "[AE] Starting training: epochs=%d, batch_size=%d, input_dim=%d, "
        "latent_dim=%d, hidden_dims=%s, dropout=%.4f, regularization=%.6f, "
        "patience=%d, min_delta=%.6f",
        max_epochs,
        ae_cfg.batch_size,
        input_dim,
        ae_cfg.latent_dim,
        ae_cfg.hidden_dims,
        ae_cfg.dropout,
        ae_cfg.regularization,
        patience,
        min_delta,
    )

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for (batch_X,) in train_loader:
            batch_X = batch_X.to(device)

            optimizer.zero_grad()
            recon_X = model(batch_X)
            loss = criterion(recon_X, batch_X)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for (batch_X,) in val_loader:
                batch_X = batch_X.to(device)
                recon_X = model(batch_X)
                loss = criterion(recon_X, batch_X)

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        logger.info(
            "[AE] Epoch %d/%d, Train Loss: %.6f, Val Loss: %.6f",
            epoch,
            max_epochs,
            avg_train_loss,
            avg_val_loss,
        )

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "[AE] Early stopping triggered at epoch %d (best epoch %d, "
                    "best val loss %.6f).",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

    # Restore best model state
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(
            "[AE] Restored best model from epoch %d with val loss %.6f.",
            best_epoch,
            best_val_loss,
        )

    # Save model checkpoint
    model_path = universe.ae_model_path()
    ensure_parent_dir(model_path)

    checkpoint = {
        "input_dim": input_dim,
        "hidden_dims": list(ae_cfg.hidden_dims),
        "latent_dim": ae_cfg.latent_dim,
        "dropout": ae_cfg.dropout,
        "ae_regularization": ae_cfg.regularization,
        "model_state_dict": model.state_dict(),
        "universe_id": universe.id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, model_path)
    logger.info(f"[AE] Saved AE checkpoint to {model_path}")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
        reserved_memory = torch.cuda.max_memory_reserved(device) / (1024**2)
        logger.info(
            f"[TORCH] Peak GPU memory usage during training: {peak_memory:.2f} MB"
        )
        logger.info(
            f"[TORCH] Peak GPU reserved memory during training: {reserved_memory:.2f} MB"
        )

    return model_path


def load_autoencoder_for_universe(
    universe: Universe, ds_cfg: DatasetConfig
) -> Autoencoder:
    """
    Load the trained autoencoder model for this universe from its checkpoint.
    """
    model_path = universe.ae_model_path()
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
