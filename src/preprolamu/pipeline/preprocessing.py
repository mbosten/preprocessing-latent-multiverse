from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Generator

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    )

from preprolamu.config import load_dataset_config
from preprolamu.pipeline.universes import Universe

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocess dataset according to a universe configuration."""

    def __init__(self, universe: Universe):
        self.universe = universe
        self.config = load_dataset_config(universe.dataset_id)

        self.df:    pd.DataFrame | None = None
        self.train: pd.DataFrame | None = None
        self.val:   pd.DataFrame | None = None
        self.test:  pd.DataFrame | None = None

        self.transformer: TransformerMixin | None = None
        self.scaler: TransformerMixin | None = None

    def process(self):
        self.load()
        self.feature_subset()
        self.duplicates()
        self.split()
        self.missingness()
        self.transform()
        self.scale()

        return self._require_splits()

    def run(self, overwrite=False):
        logger.info("Preprocessing universe=%s", self.universe.id)

        if not overwrite and all(path.exists() for path in self.paths):
            logger.info("Preprocessed files already exist. Skipping.")
            self._clear_status()
            return self.paths

        with self._preprocessing_status():
            self.process()
            self.save()

        return self.paths

    # ──────────────────────────────────────────────────────────
    # Dataset                                                 
    # ──────────────────────────────────────────────────────────

    def load(self):
        self.df = pd.read_parquet(self.universe.paths.clean_data())
        logger.info("Loaded %d rows x %d columns.", *self.df.shape)


    def split(self, train_frac=0.6, val_frac=0.2):
            """Create benign-only training data and stratified validation/test data."""
            df = self._require_df()
    
            label_col = self.config["label_column"]
            benign_label = self.config["benign_label"]
    
            if label_col not in df.columns:
                raise ValueError(f"Label column {label_col!r} not found.")
    
            train, remainder = train_test_split(
                df,
                train_size=train_frac,
                random_state=self.universe.seed,
                stratify=df[label_col],
            )
    
            val, test = train_test_split(
                remainder,
                train_size=val_frac / (1 - train_frac),
                random_state=self.universe.seed,
                stratify=remainder[label_col],
            )
    
            self.train = (train[train[label_col] == benign_label].reset_index(drop=True))
            self.val = val.reset_index(drop=True)
            self.test = test.reset_index(drop=True)
            self.df = None
    
            logger.info(
                "Data split: train=%d, validation=%d, test=%d",
                len(self.train),
                len(self.val),
                len(self.test)
            )   

    # ──────────────────────────────────────────────────────────
    # Preprocessing                                                 
    # ──────────────────────────────────────────────────────────
    def feature_subset(self):
        if self.universe.feature_subset == "all":
            return

        if self.universe.feature_subset != "without_confounders":
            raise ValueError(
                f"Unknown feature subset: {self.universe.feature_subset!r}"
            )

        confounders = self.config.get("confounders", [])
        self.df = self._require_df().drop(columns=confounders,errors="ignore")

        logger.info("Dropped confounders: %s", confounders)


    def duplicates(self):
        if self.universe.duplicate_handling == "keep":
            return

        df = self._require_df()
        before = len(df)

        self.df = df.drop_duplicates()

        logger.info("Dropped %d duplicate rows.", before - len(self.df))


    def missingness(self):
        cols = self.feature_cols

        if not cols:
            return

        def replace_inf(df: pd.DataFrame):
            df = df.copy()
            df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
            return df

        self._map_splits(replace_inf)

        if self.universe.missingness == "drop_rows":
            self._map_splits(
                lambda df: df.dropna(subset=cols).reset_index(drop=True)
            )
            return


        if self.universe.missingness == "imp_med":
            medians = self._require_train()[cols].median()

            def impute(df: pd.DataFrame):
                df = df.copy()
                df[cols] = df[cols].fillna(medians)
                return df

            self._map_splits(impute)
            return

        raise ValueError(f"Unknown missingness strategy: {self.universe.missingness!r}")


    def transform(self):
        method = self.universe.transform

        if method == "none":
            return

        cols = self.feature_cols
        if not cols:
            return

        if method == "log1p":
            func = lambda x: np.sign(x) * np.log1p(np.abs(x))

        elif method == "asinh":
            func = np.arcsinh

        elif method == "yeo_johnson":
            self.transformer = PowerTransformer(
                method="yeo-johnson", 
                standardize=False,
            ).fit(self._require_train()[cols])
            
            func = self.transformer.transform

        else:
            raise ValueError(f"Unknown transformation: {method!r}")

        def transform_df(df):
            df = df.copy()
            df[cols] = func(df[cols].astype(float))
            return df

        self._map_splits(transform_df)
        logger.info("Applied %s transformation.", method)


    def scale(self):
        method = self.universe.scaling

        if method == "none":
            return
        
        cols = self.feature_cols
        if not cols:
            return

        train = self._require_train()[cols]

        if method == "robust_mad":
            median = train.median()
            mad = (train - median).abs().median().replace(0, 1.0)
            func = lambda x: (x - median) / mad

        elif method == "robust_iqr":
            scaler = RobustScaler(
                quantile_range=(25, 75),
            ).fit(train)
            self.scaler = scaler
            func = scaler.transform

        elif method in {"quantile_uniform", "quantile_normal"}:
            scaler = QuantileTransformer(
                output_distribution=method.removeprefix("quantile_"),
                random_state=self.universe.seed,
            ).fit(train)
            self.scaler = scaler
            func = scaler.transform

        else:  
            raise ValueError(f"Unknown Scaling method: {self.universe.scaling!r}")

        def transform_df(df: pd.DataFrame):
            df = df.copy()
            df[cols] = func(df[cols])
            return df

        self._map_splits(transform_df)

        logger.info(
            "Applied %s scaling to %d features.",
            self.universe.scaling,
            len(cols),
        )

    # ──────────────────────────────────────────────────────────
    # IO                                                 
    # ──────────────────────────────────────────────────────────
    def save(self):
        for split, path in zip(
            self._require_splits(),
            self.paths,
            strict=True,
        ):
            split.to_parquet(path)
            logger.info("Saved preprocessed data to %s", path)

    # ──────────────────────────────────────────────────────────
    # Properties                                                 
    # ──────────────────────────────────────────────────────────
    @property
    def feature_cols(self):
        train = self._require_train()
        label_col = self.config["label_column"]

        return [
            col
            for col in train.select_dtypes(include="number").columns
            if col != label_col
        ]

    @property
    def paths(self):
        return tuple(
            self.universe.paths.preprocessed(split=split)
            for split in ("train", "val", "test")
        )

    @property
    def _status_path(self):
        return self.universe.paths.preprocessing_status()

    # ──────────────────────────────────────────────────────────
    # Helpers                                                 
    # ──────────────────────────────────────────────────────────
    def _map_splits(self, func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.train, self.val, self.test = map(
            func,
            self._require_splits(),
        )

    def _require_df(self):
        if self.df is None:
            raise RuntimeError("Raw dataframe is not Loaded.")
        return self.df

    def _require_train(self):
        if self.train is None:
            raise RuntimeError("Dataset has not been split.")
        return self.train

    def _require_splits(self):
        if self.train is None or self.val is None or self.test is None:
            raise RuntimeError("Dataset has not been split.")
        return self.train, self.val, self.test

    def _clear_status(self):
        try:
            self._status_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove preprocessing status file %s.",
                self._status_path,
                exc_info=True,
            )

    @contextmanager
    def _preprocessing_status(self) -> Generator[None]:
        self._status_path.write_text("IN_PROGRESS\n", encoding="utf-8")

        try:
            yield
        except Exception:
            logger.exception(
                "Preprocessing failed for universe=%s",
                self.universe.id,
            )

            try:
                self._status_path.write_text("FAILED\n", encoding="utf-8")
            except OSError:
                logger.warning(
                    "Could not write preprocessing status file %s.",
                    self._status_path,
                    exc_info=True,
                )
            raise
        else:
            self._clear_status()

