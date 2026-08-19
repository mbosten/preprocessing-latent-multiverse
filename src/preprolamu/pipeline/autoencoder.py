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
        # self.input_dim = input_dim
        # self.hidden_dims = hidden_dims
        # self.latent_dim = latent_dim
        # self.dropout_p = dropout

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

        # layers = []
        # prev_dim = input_dim

        # # Build encoder
        # for hdim in hidden_dims:
        #     layers.append(nn.Linear(prev_dim, hdim))
        #     layers.append(nn.ReLU())
        #     if dropout > 0:
        #         layers.append(nn.Dropout(dropout))
        #     prev_dim = hdim

        # layers.append(nn.Linear(prev_dim, latent_dim))
        # self.encoder = nn.Sequential(*layers)

        # dec_layers = []
        # prev_dim = latent_dim

        # # Build decoder
        # for hdim in reversed(hidden_dims):
        #     dec_layers.append(nn.Linear(prev_dim, hdim))
        #     dec_layers.append(nn.ReLU())
        #     if dropout > 0:
        #         dec_layers.append(nn.Dropout(dropout))
        #     prev_dim = hdim

        # dec_layers.append(nn.Linear(prev_dim, input_dim))
        # self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    # def encode(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.encoder(x)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Check for GPU support
# def _get_device() -> torch.device:
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
#         logger.info("[TORCH] Resetting peak memory stats.")
#         torch.cuda.reset_peak_memory_stats(device)
#     else:
#         device = torch.device("cpu")
#         logger.info("Using CPU")

#     return device

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


# def _seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#     logger.info("[AE] Seeding worker RNGs with seed=%d", worker_seed)


# def _seed_generator(seed) -> torch.Generator:

#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     g = torch.Generator()
#     g.manual_seed(seed)
#     return g


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

    model = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dims=ae["hidden_dims"],
        latent_dim=ae["latent_dim"],
        dropout=ae["dropout"],
    ).to(device())

    generator = _seed_everything(SEED)

    train_loader = _loader(X_train, ae["batch_size"], shuffle=True, generator=generator)
    val_loader = _loader(X_val, ae["batch_size"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ae.get("learning_rate", 1e-3),
        weight_decay=ae.get("regularization", 0.0),
    )

    # Different from previous iterations to be on the safe side (previous: 5, 1e-4)
    patience = ae.get("patience", 10)
    min_delta = ae.get("min_delta", 1e-5)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, ae["epochs"] + 1):
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

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "input_dim": X_train.shape[1],
            "hidden_dims": ae["hidden_dims"],
            "latent_dim": ae["latent_dim"],
            "dropout": ae["dropout"],
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
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

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, latent)

    logger.info("Saved embedding to %s", path)
    return latent


# def _get_feature_matrix_for_ae(df: pd.DataFrame, ds_cfg: DatasetConfig) -> np.ndarray:
#     df_features = df.copy()
#     cols_to_drop = [ds_cfg["label_column"]]

#     # Specific for the three NF datasets
#     if "Attack" in df_features.columns and "Attack" not in cols_to_drop:
#         cols_to_drop.append("Attack")

#     logger.info(
#         f"[AE] Dropping columns for AE feature matrix (if present): {cols_to_drop}"
#     )
#     df_features = df_features.drop(columns=cols_to_drop, errors="ignore")
#     feature_names = df_features.columns.tolist()

#     X = df_features.to_numpy(dtype=np.float32)
#     logger.info(f"[AE] Feature matrix shape for AE: {X.shape}")

#     return X, feature_names


# def get_feature_matrix_from_universe(
#     universe: Universe, split: str = "train"
# ) -> tuple[np.ndarray, np.ndarray, DatasetConfig]:

#     ds_cfg: DatasetConfig = load_dataset_config(universe.dataset_id)

#     if split == "train":
#         path = universe.paths.preprocessed(split="train")
#     elif split == "val":
#         path = universe.paths.preprocessed(split="val")
#     elif split == "test":
#         path = universe.paths.preprocessed(split="test")
#     else:
#         raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

#     logger.info("[AE] Loading preprocessed %s data from %s", split.upper(), path)

#     df = pd.read_parquet(path)

#     if split == "test":
#         # Only return benign samples for AE encoder
#         # Performance evaluation uses a different function for retrieving the feature matrix.
#         df = df[df[ds_cfg["label_column"]] == ds_cfg["benign_label"]]

#     logger.info("[AE] %s data shape: %s", split.upper(), df.shape)

#     X, feature_names = _get_feature_matrix_for_ae(df, ds_cfg)
#     return X, feature_names, ds_cfg


# def train_autoencoder_for_universe(universe: Universe) -> Path:
#     logger.info(f"[AE] Training autoencoder for universe = {universe.id}")

#     seed = universe.seed

#     g = _seed_generator(seed)

#     X_train, feature_names, ds_cfg = get_feature_matrix_from_universe(
#         universe, split="train"
#     )

#     X_val, _, _ = get_feature_matrix_from_universe(universe, split="val")

#     ae_cfg = ds_cfg["autoencoder"]

#     input_dim = X_train.shape[1]
#     device = _get_device()

#     model = Autoencoder(
#         input_dim=input_dim,
#         hidden_dims=ae_cfg["hidden_dims"],
#         latent_dim=ae_cfg["latent_dim"],
#         dropout=ae_cfg["dropout"],
#     ).to(device)

#     tensor_X_train = torch.from_numpy(X_train)
#     tensor_X_val = torch.from_numpy(X_val)

#     train_dataset = TensorDataset(tensor_X_train)
#     val_dataset = TensorDataset(tensor_X_val)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=ae_cfg["batch_size"],
#         shuffle=True,
#         drop_last=False,
#         generator=g,
#         worker_init_fn=_seed_worker,
#     )

#     val_loader = DataLoader(
#         val_dataset, batch_size=ae_cfg["batch_size"], shuffle=False, drop_last=False
#     )

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=1e-3, weight_decay=ae_cfg["regularization"]
#     )

#     max_epochs = ae_cfg["epochs"]
#     patience = getattr(ae_cfg, "patience", 5)
#     min_delta = getattr(ae_cfg, "min_delta", 1e-4)

#     best_val_loss = float("inf")
#     best_state_dict = None
#     best_epoch = 0
#     epochs_no_improve = 0

#     logger.info(
#         "[AE] Starting training: epochs=%d, batch_size=%d, input_dim=%d, "
#         "latent_dim=%d, hidden_dims=%s, dropout=%.4f, regularization=%.6f, "
#         "patience=%d, min_delta=%.6f",
#         max_epochs,
#         ae_cfg["batch_size"],
#         input_dim,
#         ae_cfg["latent_dim"],
#         ae_cfg["hidden_dims"],
#         ae_cfg["dropout"],
#         ae_cfg["regularization"],
#         patience,
#         min_delta,
#     )

#     for epoch in range(1, max_epochs + 1):
#         # Training phase
#         model.train()
#         train_loss_sum = 0.0
#         train_batches = 0
#         for (batch_X,) in train_loader:
#             batch_X = batch_X.to(device)

#             optimizer.zero_grad()
#             recon_X = model(batch_X)
#             loss = criterion(recon_X, batch_X)
#             loss.backward()
#             optimizer.step()

#             train_loss_sum += loss.item()
#             train_batches += 1

#         avg_train_loss = train_loss_sum / max(train_batches, 1)

#         # Validation phase
#         model.eval()
#         val_loss_sum = 0.0
#         val_batches = 0

#         with torch.no_grad():
#             for (batch_X,) in val_loader:
#                 batch_X = batch_X.to(device)
#                 recon_X = model(batch_X)
#                 loss = criterion(recon_X, batch_X)

#                 val_loss_sum += loss.item()
#                 val_batches += 1

#         avg_val_loss = val_loss_sum / max(val_batches, 1)

#         logger.info(
#             "[AE] Epoch %d/%d, Train Loss: %.6f, Val Loss: %.6f",
#             epoch,
#             max_epochs,
#             avg_train_loss,
#             avg_val_loss,
#         )

#         # Early stopping check
#         if avg_val_loss < best_val_loss - min_delta:
#             best_val_loss = avg_val_loss
#             best_state_dict = model.state_dict()
#             best_epoch = epoch
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 logger.info(
#                     "[AE] Early stopping triggered at epoch %d (best epoch %d, "
#                     "best val loss %.6f).",
#                     epoch,
#                     best_epoch,
#                     best_val_loss,
#                 )
#                 break

#     # Restore best model state
#     if best_state_dict is not None:
#         model.load_state_dict(best_state_dict)
#         logger.info(
#             "[AE] Restored best model from epoch %d with val loss %.6f.",
#             best_epoch,
#             best_val_loss,
#         )

#     # Save model checkpoint
#     model_path = universe.paths.ae_model()

#     checkpoint = {
#         "input_dim": input_dim,
#         "hidden_dims": list(ae_cfg["hidden_dims"]),
#         "latent_dim": ae_cfg["latent_dim"],
#         "dropout": ae_cfg["dropout"],
#         "ae_regularization": ae_cfg["regularization"],
#         "model_state_dict": model.state_dict(),
#         "universe_id": universe.id,
#         "best_epoch": best_epoch,
#         "best_val_loss": best_val_loss,
#     }
#     torch.save(checkpoint, model_path)
#     logger.info(f"[AE] Saved AE checkpoint to {model_path}")

#     if device.type == "cuda":
#         peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
#         reserved_memory = torch.cuda.max_memory_reserved(device) / (1024**2)
#         logger.info(
#             f"[TORCH] Peak GPU memory usage during training: {peak_memory:.2f} MB"
#         )
#         logger.info(
#             f"[TORCH] Peak GPU reserved memory during training: {reserved_memory:.2f} MB"
#         )

#     return model_path


# def load_autoencoder_for_universe(
#     universe: Universe, ds_cfg: DatasetConfig
# ) -> Autoencoder:

#     model_path = universe.paths.ae_model()
#     if not model_path.exists():
#         raise FileNotFoundError(
#             f"Autoencoder model checkpoint not found at {model_path}"
#         )
#     ae_cfg = ds_cfg["autoencoder"]
#     checkpoint = torch.load(model_path, map_location=_get_device())
#     input_dim = int(checkpoint["input_dim"])
#     hidden_dims = tuple(int(h) for h in checkpoint["hidden_dims"])
#     latent_dim = int(checkpoint["latent_dim"])
#     dropout = float(checkpoint.get("dropout", ae_cfg["dropout"]))

#     model = Autoencoder(
#         input_dim=input_dim,
#         hidden_dims=hidden_dims,
#         latent_dim=latent_dim,
#         dropout=dropout,
#     )

#     model.load_state_dict(checkpoint["model_state_dict"])
#     logger.info(
#         "[AE] Loaded AE model from %s (input_dim=%d, latent_dim=%d, hidden_dims=%s, dropout=%.4f)",
#         model_path,
#         input_dim,
#         latent_dim,
#         hidden_dims,
#         dropout,
#     )

#     return model
