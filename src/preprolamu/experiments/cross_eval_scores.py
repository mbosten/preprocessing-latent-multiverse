from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/processed/cross_eval_scores")
OUTPUT_DIR = Path("data/processed/cross_eval_score_agg")
N_THRESHOLDS = 101


def threshold_summary(y_true: np.ndarray, scores: np.ndarray):
    """Compute classification statistics across score thresholds for cross-dataset generalization scores."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape.")

    # Remove non-finite data
    valid = np.isfinite(scores)
    n_total = scores.size
    n_invalid = np.count_nonzero(~valid)

    y_true = y_true[valid]
    scores = scores[valid]

    if scores.size == 0:
        return [], n_total, n_invalid

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

    rows =  zip(
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
    )

    return rows, n_total, n_invalid


def summarize_file(path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{path.stem}.csv"

    with np.load(path, allow_pickle=False) as data, output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
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
                "n_total",
                "n_invalid",
            ]
        )

        for y_true_key in data.files:
            if not y_true_key.endswith("__y_true"):
                continue

            target_id = y_true_key.removesuffix("__y_true")
            scores_key = f"{target_id}__scores"

            rows, n_total, n_invalid = threshold_summary(
                data[y_true_key],
                data[scores_key],
            )

            writer.writerows(
                [
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
                    n_total,
                    n_invalid,
                ]
                for (
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
                ) in rows
            )


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob("*.npz"))
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    summarize_file(files[index])