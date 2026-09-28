from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_THRESHOLDS = 101


def threshold_summary(y_true: np.ndarray, scores: np.ndarray):
    """Compute classification statistics across score thresholds for cross-dataset generalization scores."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape.")

    # Remove non-finite data
    valid = np.isfinite(scores)
    n_invalid = np.count_nonzero(~valid)

    y_true = y_true[valid]
    scores = scores[valid]

    if scores.size == 0:
        return [], n_invalid

    # Label-independent threshold grid.
    thresholds = np.unique(
        np.quantile(scores, np.linspace(0.0, 1.0, N_THRESHOLDS))
    )

    positive_scores = np.sort(scores[y_true])
    negative_scores = np.sort(scores[~y_true])

    n_positive = positive_scores.size
    n_negative = negative_scores.size

    # A sample is classified as an attack when score >= threshold.
    tp = n_positive - np.searchsorted(
        positive_scores, thresholds, side="left"
    )
    fp = n_negative - np.searchsorted(
        negative_scores, thresholds, side="left"
    )

    fn = n_positive - tp
    tn = n_negative - fp

    def divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    accuracy = divide(tp + tn, n_positive + n_negative)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    specificity = divide(tn, tn + fp)

    return zip(
        thresholds,
        tp,
        fp,
        fn,
        tn,
        accuracy,
        precision,
        recall,
        f1,
        specificity,
    ), n_invalid


def summarize_folder() -> None:
    """Summarize all cross-evaluation NPZ files"""
    output_path = Path("data/processed/analysis") / "cross_eval_summary.csv.gz"

    folder = Path("data/processed/cross_eval_scores")
    npz_files = sorted(folder.rglob("*.npz"))

    fieldnames = [
        "source_file",
        "target_id",
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "n_invalid",
    ]

    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for file_index, path in enumerate(npz_files, start=1):
            logger.info(f"[{file_index}/{len(npz_files)}] {path}")

            with np.load(path, allow_pickle=False) as data:
                y_true_keys = [
                    key for key in data.files if key.endswith("__y_true")
                ]

                for y_true_key in y_true_keys:
                    target_id = y_true_key.removesuffix("__y_true")
                    scores_key = f"{target_id}__scores"

                    if scores_key not in data:
                        raise KeyError(
                            f"Missing {scores_key!r} in {path}"
                        )

                    y_true = data[y_true_key]
                    scores = data[scores_key]

                    for row in threshold_summary(y_true, scores):
                        (
                            threshold,
                            tp,
                            fp,
                            fn,
                            tn,
                            accuracy,
                            precision,
                            recall,
                            f1,
                            specificity,
                            n_invalid
                        ) = row

                        writer.writerow(
                            [
                                path.relative_to(folder),
                                target_id,
                                float(threshold),
                                int(tp),
                                int(fp),
                                int(fn),
                                int(tn),
                                float(accuracy),
                                float(precision),
                                float(recall),
                                float(f1),
                                float(specificity),
                                int(n_invalid),
                            ]
                        )

    logger.info(f"Saved summary to {output_path}")


if __name__ == "__main__":
    summarize_folder()