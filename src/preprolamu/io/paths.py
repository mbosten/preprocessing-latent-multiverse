from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from preprolamu.pipeline.universes import Universe

Split = Literal["train", "val", "test"]


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class UniversePaths:
    u: Universe

    def clean_data(self) -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir / "raw" / f"{self.u.dataset_id}_clean.parquet"
        )

    def embedding(self, split: Split = "test") -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "interim"
            / "embeddings"
            / f"{self.u.id}_latent_{split}.npy"
        )

    def projected(self, split: Split = "test", normalized: bool = False) -> Path:
        tag = "" if normalized else "_raw"
        return ensure_parent_dir(
            self.u.base_data_dir
            / "interim"
            / "projections"
            / f"{self.u.id}_projected_{split}{tag}.npy"
        )

    def persistence(self, split: Split = "test") -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "interim"
            / "persistence"
            / f"{self.u.id}_persistence_{split}.npz"
        )

    def landscapes(self, split: Split = "test") -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "interim"
            / "landscapes"
            / f"{self.u.id}_landscapes_{split}.npz"
        )

    def metrics(self, split: Split = "test") -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "processed"
            / "metrics"
            / f"{self.u.id}_metrics_{split}.json"
        )

    def ae_model(self) -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir / "interim" / "autoencoder" / f"{self.u.id}_ae.pt"
        )

    def eval_metrics(self, split: Split = "test") -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "processed"
            / "eval_metrics"
            / f"{self.u.id}_eval_{split}.json"
        )

    def preprocessed(self, split: Split) -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "processed"
            / split
            / f"{self.u.id}_preprocessed_{split}.parquet"
        )

    def preprocessing_status(self) -> Path:
        return ensure_parent_dir(
            self.u.base_data_dir
            / "interim"
            / "preprocessing_status"
            / f"{self.u.id}.status"
        )
