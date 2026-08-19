from __future__ import annotations

import copy
import logging
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprolamu.config import load_dataset_config
from preprolamu.helpers import feature_matrix, load_split
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


SEED = 42  # Default seed for reproducibility

# Autoencoder architecture class
class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = self._network([input_dim, *hidden_dims, latent_dim], dropout)
        self.decoder = self._network([latent_dim, *reversed(hidden_dims), input_dim], dropout)

    @staticmethod
    def _network(dims: list[int], dropout: float) -> nn.Sequential:
        layers = []
        for i, (in_dim, out_dim) in enumerate(pairwise(dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return torch.Generator().manual_seed(seed)


def _loader(
    X: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _run_epoch(
    model: Autoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for (X,) in loader:
            X = X.to(device())

            if training:
                optimizer.zero_grad()

            loss = nn.functional.mse_loss(model(X), X)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def fit_autoencoder(
        universe: Universe,
        X_train: np.ndarray,
        X_val: np.ndarray,
        *,
        epochs: int | None = None,
) -> Autoencoder:
    config = load_dataset_config(universe.dataset_id)
    ae = config["autoencoder"]

    model = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dims=ae["hidden_dims"],
        latent_dim=ae["latent_dim"],
        dropout=ae["dropout"],
    ).to(device())

    generator = _seed_everything(SEED)

    train_loader = _loader(
        X_train,
        ae["batch_size"],
        shuffle=True,
        generator=generator,
    )
    val_loader = _loader(X_val, ae["batch_size"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ae.get("learning_rate", 1e-3),
        weight_decay=ae.get("regularization", 0.0),
    )

    patience = ae.get("patience", 10)
    min_delta = ae.get("min_delta", 1e-5)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    
    for epoch in range(1, (epochs or ae["epochs"]) + 1):
        train_loss = _run_epoch(model, train_loader, optimizer)
        val_loss = _run_epoch(model, val_loader)
    
        logger.info(
            "AE %s | epoch %d | train %.6f | val %.6f",
            universe.id,
            epoch,
            train_loss,
            val_loss,
        )
    
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
    
            if stale_epochs >= patience:
                logger.info(
                    "[AE] Early stopping triggered at epoch %d (best epoch %d, "
                    "best val loss %.6f).",
                    epoch,
                    best_epoch,
                    best_loss,
                )
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)

    return model.eval()


def train_autoencoder(universe: Universe, overwrite: bool = False) -> Path:
    path = universe.paths.ae_model()
    if path.exists() and not overwrite:
        return path

    config = load_dataset_config(universe.dataset_id)
    ae = config["autoencoder"]

    train_df, _ = load_split(universe, "train")
    val_df, _ = load_split(universe, "val")

    X_train = feature_matrix(train_df, config["label_column"])
    X_val = feature_matrix(val_df, config["label_column"])

    model = fit_autoencoder(universe, X_train, X_val)

    torch.save(
            {
                "input_dim": X_train.shape[1],
                "hidden_dims": ae["hidden_dims"],
                "latent_dim": ae["latent_dim"],
                "dropout": ae["dropout"],
                "state_dict": model.state_dict(),
            },
            path,
        )
    
    return path


def load_autoencoder(universe: Universe) -> Autoencoder:
    checkpoint = torch.load(universe.paths.ae_model(), map_location=device(), weights_only=True)

    model = Autoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        latent_dim=checkpoint["latent_dim"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device()).eval()


def encode(model: Autoencoder, X: np.ndarray, batch_size: int = 4096):
    loader = _loader(X, batch_size)
    latent = []

    with torch.no_grad():
        for (batch,) in loader:
            latent.append(model.encoder(batch.to(device())).cpu().numpy())

    return np.concatenate(latent, axis=0)


def reconstruction_error(model: Autoencoder, X: np.ndarray, batch_size: int = 4096):
    loader = _loader(X, batch_size)
    errors = []

    with torch.no_grad():
        for (batch,) in loader:
            recon = model(batch.to(device()))
            errors.append(
                torch.mean((recon - batch) ** 2, dim=1)
                .cpu()
                .numpy()
            )

    return np.concatenate(errors, axis=0)


def create_embedding(
        universe: Universe,
        *,
        split: str = "test",
        retrain: bool = False,
        overwrite: bool = False,
) -> np.ndarray:
    path = universe.paths.embedding(split=split)

    if path.exists() and not overwrite:
        return np.load(path)

    train_autoencoder(universe, overwrite=retrain)

    df, config = load_split(universe, split=split, benign_only=(split == "test"))

    X = feature_matrix(df, config["label_column"])
    latent = encode(load_autoencoder(universe), X)

    np.save(path, latent)

    logger.info("Saved embedding to %s", path)
    return latent
