from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def plot_numeric_distributions(
    df: pd.DataFrame,
    target: str = "Label",
    bins: int = 50,
    low_cardinality_threshold: int = 20,
    skew_threshold: float = 10.0,
) -> None:
    """
    Plot integer-variable distributions grouped by target.

    - Low cardinality -> grouped bar chart
    - High cardinality -> overlaid histograms
    - Highly right-skewed, non-negative -> log1p transform
    - Uses raw counts and all observations
    """

    numeric_cols = (
        df.select_dtypes(include="integer")
        .columns
        .drop(target, errors="ignore")
    )

    groups = df[target].dropna().unique()

    for col in numeric_cols:
        data = df[[col, target]].dropna()

        if data.empty:
            continue

        values = data[col]
        n_unique = values.nunique()

        plt.figure(figsize=(8, 4))

        # Low-cardinality variables
        if n_unique <= low_cardinality_threshold:
            counts = pd.crosstab(
                data[col],
                data[target],
            )

            counts.plot(
                kind="bar",
                ax=plt.gca(),
            )

            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"{col} | {n_unique} unique values")
            plt.xticks(rotation=45)

        # High-cardinality variables
        else:
            skew = values.skew()

            use_log = (
                skew > skew_threshold
                and values.min() >= 0
            )

            # Establish common bins
            if use_log:
                all_values = np.log1p(values.to_numpy())
                xlabel = f"log1p({col})"
            else:
                all_values = values.to_numpy()
                xlabel = col

            _, edges = np.histogram(
                all_values,
                bins=bins,
            )

            # Count observations per bin for each Attack group
            for group in groups:
                group_values = data.loc[
                    data[target] == group,
                    col,
                ].to_numpy()

                if use_log:
                    group_values = np.log1p(group_values)

                counts, _ = np.histogram(
                    group_values,
                    bins=edges,
                )

                plt.stairs(
                    counts,
                    edges,
                    label=f"{target}={group}",
                    fill=True,
                    alpha=0.4,
                )

            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(
                f"{col} | skew={skew:.2f}"
                + (" | log1p" if use_log else "")
            )

        plt.legend(title=target)
        plt.tight_layout()
        plt.show()