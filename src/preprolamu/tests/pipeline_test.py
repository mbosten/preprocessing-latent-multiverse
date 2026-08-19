from __future__ import annotations

import logging
import numpy as np

from preprolamu.helpers import feature_matrix, labels
from preprolamu.pipeline.autoencoder import encode, fit_autoencoder
from preprolamu.pipeline.create_tda import compute_tda_for_test
from preprolamu.pipeline.evaluation import evaluate_model
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

    preprocessor = Preprocessor(universe)
    train, val, test = preprocessor.process()

    config = preprocessor.config
    label_col = config["label_column"]

    X_train = feature_matrix(train, label_col)
    X_val = feature_matrix(val, label_col)
    X_test = feature_matrix(test, label_col)
    y_test = labels(test, label_col)

    logger.info("Training autoencoder for %d epochs", epochs)
    model = fit_autoencoder(universe, X_train, X_val, epochs=epochs)

    logger.info("Evaluating model on test set")
    evaluation = evaluate_model(model, X_test, y_test, config["benign_label"])

    benign = y_test == config["benign_label"]

    logger.info("Computing TDA metrics for test set")
    latent = encode(model, X_test[benign])
    tda_metrics = compute_tda_for_test(universe, latent)

    return {
            "universe": universe.id,
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape,
            "embedding_shape": latent.shape,
            "roc_auc": evaluation["roc_auc"],
            "tda_metrics": tda_metrics,
        }