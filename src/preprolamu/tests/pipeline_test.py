from __future__ import annotations

import logging
import numpy as np
import time

from preprolamu.helpers import feature_matrix, labels
from preprolamu.pipeline.autoencoder import encode, fit_autoencoder
from preprolamu.pipeline.create_tda import prepare_point_cloud
from preprolamu.pipeline.embedding_metrics import embedding_metrics
from preprolamu.pipeline.evaluation import evaluate_model
from preprolamu.pipeline.persistence import Persistence
from preprolamu.pipeline.preprocessing import Preprocessor
from preprolamu.pipeline.universes import generate_multiverse, get_universe


logger = logging.getLogger(__name__)


def test_pipeline(
        universe_index: int | None = None,
        epochs: int = 2,
):
    universe = (
        get_universe(universe_index)
        if universe_index is not None
        else np.random.default_rng().choice(generate_multiverse())
    )

    logger.info("Testing pipeline for universe %s", universe.id)

    start1 = time.perf_counter()
    # Preprocessing
    preprocessor = Preprocessor(universe)
    train, val, test = preprocessor.process()
    logger.debug("Post-preprocess train/val/test shapes: %s/%s/%s", len(train), len(val), len(test))
    logger.debug("[TIME] Preprocessing took %.2f seconds", time.perf_counter() - start1)
    config = preprocessor.config
    label_col = config["label_column"]

    X_train = feature_matrix(train, label_col)
    X_val = feature_matrix(val, label_col)
    X_test = feature_matrix(test, label_col)
    y_test = labels(test, label_col)

    # AE training
    logger.info("Training autoencoder for %d epochs", epochs)
    start2 = time.perf_counter()
    logger.debug("pre-train data shapes: X_train=%s, X_val=%s, X_test=%s", X_train.shape, X_val.shape, X_test.shape)
    model = fit_autoencoder(universe, X_train, X_val, epochs=epochs)

    # AE evaluation
    logger.info("Evaluating model on test set")
    evaluation = evaluate_model(model, X_test, y_test, config["benign_label"])
    logger.debug("Post-evaluation test shapes: X_test=%s, y_test=%s", X_test.shape, y_test.shape)
    benign = y_test == config["benign_label"]
    logger.debug("[TIME] Training and evaluation took %.2f seconds", time.perf_counter() - start2)
    # Get latent space
    logger.info("Encoding test set to latent space")
    latent = encode(model, X_test[benign])
    logger.debug("Latent space shape: %s", latent.shape)
    # Embedding quality metrics
    start3 = time.perf_counter()
    quality = embedding_metrics(latent)

    logger.debug("[TIME] Embedding quality metrics took %.2f seconds", time.perf_counter() - start3)
    # Compute TDA metrics
    logger.info("Computing TDA metrics for test set. Latent shape: %s", latent.shape)
    start4 = time.perf_counter()
    point_cloud = prepare_point_cloud(universe, latent)
    logger.debug("Point cloud shape: %s", point_cloud.latent_space.shape)
    point_cloud.sample(target_size=universe.tda_config.subsample_size)
    logger.debug("[TIME] Point cloud prep took %.2f seconds", time.perf_counter() - start4)
    logger.debug("Sampled point cloud shape: %s", point_cloud.latent_space.shape)
    start5 = time.perf_counter()
    tda = Persistence(universe=universe, points=point_cloud.latent_space)
    tda.compute_intervals()
    tda.compute_landscapes()
    tda_metrics = tda.metrics()
    logger.debug("[TIME] TDA metrics took %.2f seconds", time.perf_counter() - start5)
    return {
            "universe": universe.id,
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape,
            "embedding_shape": latent.shape,
            "roc_auc": evaluation["roc_auc"],
            "tda_metrics": tda_metrics,
            "embedding_metrics": quality,
        }