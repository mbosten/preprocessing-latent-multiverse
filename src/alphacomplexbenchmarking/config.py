# src/alphacomplexbenchmarking/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml


@dataclass
class DatasetConfig:
    dataset_id: str
    raw_path: Path
    non_numerical_columns: List[str]
    features_to_exclude: List[str]
    label_column: Optional[str] = None
    label_classes: Optional[List[str]] = None


def load_dataset_config(dataset_id: str) -> DatasetConfig:
    """
    Load dataset-specific config from config/datasets/{dataset_id}.yml
    """
    config_path = Path("config") / "datasets" / f"{dataset_id}.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {config_path}. "
            f"Create it to specify dataset-specific settings for {dataset_id}."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    cfg = DatasetConfig(
        dataset_id=raw_cfg.get("dataset_id", dataset_id),
        raw_path=Path(raw_cfg["raw_path"]),
        non_numerical_columns=list(raw_cfg.get("non_numerical_columns", [])),
        features_to_exclude=list(raw_cfg.get("features_to_exclude", [])),
        label_column=raw_cfg.get("label_column"),
        label_classes=list(raw_cfg.get("label_classes", [])) or None,
    )
    return cfg