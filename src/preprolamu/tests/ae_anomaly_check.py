import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from preprolamu.config import load_dataset_config
from preprolamu.pipeline.autoencoder import Autoencoder, _get_device
from preprolamu.pipeline.universes import get_universe

parser = argparse.ArgumentParser(description="AE anomaly inspection")

parser.add_argument(
    "--universe-index",
    dest="uid",
    default=0,
    type=int,
)

# Parse inputs
args = parser.parse_args()
u = get_universe(args.uid)


BASE_DATA_DIR = Path("data")
OUTDIR = BASE_DATA_DIR / "tests"
OUTDIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTDIR / f"ae_debug_{u.id}.txt"

N_DEBUG = 4096
# Sample batch size for layer activation test (same as pipeline)
if u.dataset_id == "NF-UNSW-NB15-v3":
    N_ACTIVATION = 256
else:
    N_ACTIVATION = 512
N_DISTANCE = 64
SAVE_LATENT_SAMPLE = True
RECON_SUMMARY_N = 50000
BATCH_SIZE = 8192

device = _get_device()


# =========================================================
# Helpers for writing
# =========================================================
class ReportWriter:
    """Collects lines of text and writes one final report to disk."""

    def __init__(self, path: Path):
        self.path = path
        self.lines = []

    def add(self, text=""):
        self.lines.append(str(text))

    def add_header(self, text):
        self.lines.append("\n" + "=" * 100)
        self.lines.append(str(text))
        self.lines.append("=" * 100)

    def add_dict(self, d):
        for k, v in d.items():
            self.lines.append(f"{k}: {v}")

    def add_dataframe(self, df: pd.DataFrame, max_rows=200):
        if len(df) > max_rows:
            self.lines.append(df.head(max_rows).to_string(index=False))
            self.lines.append(
                f"... truncated, showing first {max_rows} rows of {len(df)}"
            )
        else:
            self.lines.append(df.to_string(index=False))

    def save(self):
        self.path.write_text("\n".join(self.lines), encoding="utf-8")


def load_model(model_path, device):
    """Load checkpoint dict, rebuild Autoencoder, load weights, move to device."""
    obj = torch.load(model_path, map_location=device)

    model = Autoencoder(
        input_dim=obj["input_dim"],
        hidden_dims=tuple(obj["hidden_dims"]),
        latent_dim=obj["latent_dim"],
        dropout=obj["dropout"],
    )
    model.load_state_dict(obj["model_state_dict"])
    model.to(device)
    model.eval()

    return obj, model


def df_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Convert a dataframe to contiguous float32 NumPy array on CPU."""
    return np.ascontiguousarray(df.to_numpy(dtype=np.float32))


def summarize_numpy_array(x: np.ndarray):
    """Return global summary stats for a 2D NumPy array."""
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "nans": int(np.isnan(x).sum()),
    }


def per_column_stats_df(x: np.ndarray) -> pd.DataFrame:
    """Return per-column mean/std/min/max as a dataframe."""
    return pd.DataFrame(
        {
            "dim": np.arange(x.shape[1]),
            "mean": np.mean(x, axis=0),
            "std": np.std(x, axis=0),
            "min": np.min(x, axis=0),
            "max": np.max(x, axis=0),
        }
    )


def get_tensor_feature_matrix(u, split, device):
    path = u.paths.preprocessed(split=split)
    df = pd.read_parquet(path)
    cfg = load_dataset_config(u.dataset_id)
    cols_to_drop = [cfg.label_column]
    if "Label" in df.columns and "Label" not in cols_to_drop:
        cols_to_drop.append("Label")
    df = df.drop(columns=cols_to_drop, errors="ignore")
    X = df.to_numpy(dtype=np.float32)
    return np.ascontiguousarray(X)


def iter_numpy_batches(x: np.ndarray, batch_size: int):
    """Yield consecutive NumPy batches from a full NumPy array."""
    for start in range(0, len(x), batch_size):
        yield x[start : start + batch_size]


def numpy_batch_to_tensor(xb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Move one NumPy batch to the target device as a float32 tensor."""
    return torch.from_numpy(xb).to(device, non_blocking=True)


@torch.no_grad()
def encode_dataset(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """Encode a full NumPy dataset in batches and return latent array on CPU."""
    zs = []
    for xb in iter_numpy_batches(x, batch_size):
        xt = numpy_batch_to_tensor(xb, device)
        z = model.encode(xt)
        zs.append(z.detach().cpu().numpy())
    return np.concatenate(zs, axis=0)


@torch.no_grad()
def reconstruct_dataset(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """Reconstruct a full NumPy dataset in batches and return output array on CPU."""
    ys = []
    for xb in iter_numpy_batches(x, batch_size):
        xt = numpy_batch_to_tensor(xb, device)
        y = model(xt)
        ys.append(y.detach().cpu().numpy())
    return np.concatenate(ys, axis=0)


@torch.no_grad()
def decode_latents(
    model: nn.Module, z: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """Decode a full latent array in batches and return reconstructions on CPU."""
    ys = []
    for zb in iter_numpy_batches(z, batch_size):
        zt = numpy_batch_to_tensor(zb, device)
        y = model.decoder(zt)
        ys.append(y.detach().cpu().numpy())
    return np.concatenate(ys, axis=0)


def summarize_latents(z: np.ndarray):
    """Return global and per-dimension latent summaries."""
    summary = {
        "shape": list(z.shape),
        "dtype": str(z.dtype),
        "abs_mean": float(np.mean(np.abs(z))),
        "abs_max": float(np.max(np.abs(z))),
        "global_min": float(np.min(z)),
        "global_max": float(np.max(z)),
        "nans": int(np.isnan(z).sum()),
        "dead_dims_std_lt_1e_8": int(np.sum(np.std(z, axis=0) < 1e-8)),
    }

    per_dim = pd.DataFrame(
        {
            "dim": np.arange(z.shape[1]),
            "mean": np.mean(z, axis=0),
            "std": np.std(z, axis=0),
            "min": np.min(z, axis=0),
            "max": np.max(z, axis=0),
        }
    )

    return summary, per_dim


def summarize_pairwise_latent_distances(z: np.ndarray):
    """Compute pairwise distances for a small latent sample."""
    zt = torch.from_numpy(z)
    d = torch.cdist(zt, zt).numpy()
    summary = {
        "shape": list(d.shape),
        "mean_distance": float(np.mean(d)),
        "median_distance": float(np.median(d)),
        "min_distance": float(np.min(d)),
        "max_distance": float(np.max(d)),
    }
    preview = pd.DataFrame(d[:10, :10])
    return summary, preview


def rowwise_mse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute mean squared error per row."""
    return np.mean((x - y) ** 2, axis=1)


def percentile_summary(arr_1d: np.ndarray):
    """Summarize a 1D array with common percentiles."""
    q = np.percentile(arr_1d, [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        "n": int(len(arr_1d)),
        "mean": float(arr_1d.mean()),
        "std": float(arr_1d.std()),
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(q[7]),
    }


def decoder_sensitivity_report(
    model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int
):
    """Test whether reconstructions change when latent codes are altered."""
    z = encode_dataset(model, x, device, batch_size)

    # Decode as is
    recon_real = decode_latents(model, z, device, batch_size)

    # Decode zeros
    recon_zero = decode_latents(model, np.zeros_like(z), device, batch_size)

    # Decode shuffled input
    rng = np.random.default_rng(seed=42)
    perm_idx = rng.permutation(len(z))
    recon_shuf = decode_latents(model, z[perm_idx], device, batch_size)

    # Decode with random noise
    recon_noise = decode_latents(
        model,
        z + 0.1 * rng.standard_normal(*z.shape, dtype=np.float32),
        device,
        batch_size,
    )
    # Fist 3 rows show relative effect wrt default.
    return {
        "decoder_output_mse_real_vs_zero": float(
            np.mean((recon_real - recon_zero) ** 2)
        ),
        "decoder_output_mse_real_vs_shuffled": float(
            np.mean((recon_real - recon_shuf) ** 2)
        ),
        "decoder_output_mse_real_vs_noisy": float(
            np.mean((recon_real - recon_noise) ** 2)
        ),
        "reconstruction_mse_real": float(np.mean((x - recon_real) ** 2)),
        "reconstruction_mse_zero": float(np.mean((x - recon_zero) ** 2)),
        "reconstruction_mse_shuffled": float(np.mean((x - recon_shuf) ** 2)),
        "reconstruction_mse_noisy": float(np.mean((x - recon_noise) ** 2)),
    }


def register_activation_hooks(model: nn.Module):
    """Register forward hooks on leaf modules and collect activations."""
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()

        return hook

    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(name)))
    return activations, handles


def remove_hooks(handles):
    """Remove registered forward hooks."""
    for h in handles:
        h.remove()


@torch.no_grad()
def activation_summary(model: nn.Module, x: np.ndarray, device: torch.device):
    """Run one small batch through the model and summarize each layer output."""
    xb = numpy_batch_to_tensor(x, device)
    activations, handles = register_activation_hooks(model)
    _ = model(xb)
    remove_hooks(handles)

    rows = []
    for name, a in activations.items():
        rows.append(
            {
                "layer": name,
                "shape": list(a.shape),
                "mean": float(a.mean().item()),
                "std": float(a.std().item()),
                "min": float(a.min().item()),
                "max": float(a.max().item()),
                "frac_zero": float((a == 0).float().mean().item()),
            }
        )

    return pd.DataFrame(rows)


def parameter_summary(model: nn.Module) -> pd.DataFrame:
    """Summarize magnitude and spread of all trainable parameters."""
    rows = []
    for name, p in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": list(p.shape),
                "mean_abs": float(p.abs().mean().item()),
                "max_abs": float(p.abs().max().item()),
                "mean": float(p.mean().item()),
                "std": float(p.std().item()),
                "min": float(p.min().item()),
                "max": float(p.max().item()),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
device = _get_device()
report = ReportWriter(REPORT_PATH)

model_path = u.paths.ae_model()
obj, model = load_model(model_path, device)

report.add_header("Checkpoint / Model Info")
report.add(f"model_path: {model_path}")
report.add(f"device: {device}")
report.add(f"checkpoint keys: {list(obj.keys())}")
report.add(f"input_dim: {obj['input_dim']}")
report.add(f"hidden_dims: {obj['hidden_dims']}")
report.add(f"latent_dim: {obj['latent_dim']}")
report.add(f"dropout: {obj['dropout']}")
report.add(f"best_epoch: {obj.get('best_epoch')}")
report.add(f"best_val_loss: {obj.get('best_val_loss')}")
report.add("")
report.add("Model:")
report.add(model)

train_X = get_tensor_feature_matrix(u, split="train", device=device)
val_X = get_tensor_feature_matrix(u, split="val", device=device)
test_X = get_tensor_feature_matrix(u, split="test", device=device)

report.add_header("Dataset Shapes")
report.add(f"train: {train_X.shape} {train_X.dtype}")
report.add(f"val  : {val_X.shape} {val_X.dtype}")
report.add(f"test : {test_X.shape} {test_X.dtype}")

for split_name, x in [("train", train_X), ("val", val_X), ("test", test_X)]:
    report.add_header(f"Input Summary: {split_name}")
    report.add_dict(summarize_numpy_array(x))
    report.add("")
    report.add("Per-column stats:")
    report.add_dataframe(per_column_stats_df(x), max_rows=200)

x_debug = test_X[: min(N_DEBUG, len(test_X))]
z_debug = encode_dataset(model, x_debug, device, BATCH_SIZE)

latent_summary, latent_df = summarize_latents(z_debug)
report.add_header("Latent Summary: test_subset")
report.add_dict(latent_summary)
report.add("")
report.add("Per-latent-dimension stats:")
report.add_dataframe(latent_df, max_rows=200)

z_dist = z_debug[: min(N_DISTANCE, len(z_debug))]
dist_summary, dist_preview = summarize_pairwise_latent_distances(z_dist)
report.add_header("Latent Distance Summary: test_subset")
report.add_dict(dist_summary)
report.add("")
report.add("Top-left of distance matrix:")
report.add_dataframe(dist_preview, max_rows=20)

# Gauge decoder sensitivity under systematic latent variation.
report.add_header("Decoder Sensitivity: test_subset")
report.add_dict(decoder_sensitivity_report(model, x_debug, device, BATCH_SIZE))

report.add_header("Layer Activations: test_subset")
report.add_dataframe(
    activation_summary(model, test_X[: min(N_ACTIVATION, len(test_X))], device),
    max_rows=200,
)

# Inspect parameter distributions
report.add_header("Parameter Magnitudes")
report.add_dataframe(parameter_summary(model), max_rows=200)

for split_name, x in [
    ("train_subset", train_X[: min(RECON_SUMMARY_N, len(train_X))]),
    ("val_subset", val_X[: min(RECON_SUMMARY_N, len(val_X))]),
    ("test_subset", test_X[: min(RECON_SUMMARY_N, len(test_X))]),
]:
    recon = reconstruct_dataset(model, x, device, BATCH_SIZE)
    mse = rowwise_mse(x, recon)

    report.add_header(f"Reconstruction Summary: {split_name}")
    report.add_dict(percentile_summary(mse))
    report.add("")
    report.add("First 20 per-row MSE values:")
    report.add(np.array2string(mse[:20], precision=6, separator=", "))

if device.type == "cuda":
    report.add_header("CUDA Memory Summary")
    report.add(
        f"max_memory_allocated_MB: {torch.cuda.max_memory_allocated(device) / (1024**2):.2f}"
    )
    report.add(
        f"max_memory_reserved_MB : {torch.cuda.max_memory_reserved(device) / (1024**2):.2f}"
    )

report.save()
print(f"Done. Report written to: {REPORT_PATH.resolve()}")
