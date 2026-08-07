from contextlib import contextmanager
from pathlib import Path

import csv
import logging
import time

import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class Experiment:
    """Base class for experiments."""

    def __init__(
        self,
        name,
        stem,
        parameter_name="Parameter",
        figure_dir="data/figures",
        results_dir="data/experiments"
    ):
        self.name = name
        self.stem = stem
        self.parameter_name = parameter_name

        self.figure_dir = Path(figure_dir) / name
        self.results_dir = Path(results_dir) / name

        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.timings = {}

    @contextmanager
    def timer(self, name, key):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings.setdefault(name, {})[key] = (
                time.perf_counter() - start
            )

    def figure_path(self, name):
        return self.figure_dir / f"{name}_{self.stem}.png"

    def results_path(self, name):
        return self.results_dir / f"{name}_{self.stem}.csv"

    def plot_timings(self, name):
        values = self.timings[name]
        out_path = self.figure_path(f"{name}_time")

        grouped = self._group(values)
        values = {k: np.mean(v) for k, v in grouped.items()}

        fig, ax  = plt.subplots(figsize=(8, 6), dpi=300)
        ax.plot(values.keys(), values.values())
        ax.set(
            xlabel=self.parameter_name,
            ylabel="Computation time (s)",
        )
        fig.tight_layout(pad=1.5)
        fig.savefig(out_path)
        plt.close(fig)

    def plot_metric(self, key, ylabel, out_path):
        grouped = self._group(self.results)
        parameters = sorted(grouped)

        hom_dims = sorted({
            dim
            for runs in grouped.values()
            for result in runs
            for dim in result[key]
        })

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        for dim in hom_dims:
            ax.plot(
                parameters,
                [
                    np.nanmean([
                        result[key].get(dim, np.nan)
                        for result in grouped[param]
                    ])
                    for param in parameters
                ],
                label=f"H{dim}",
            )

        ax.set_xlabel(self.parameter_name, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=18)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def save_results(self, out_path, fields, rows):
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)

        logger.info("Wrote metric table to %s.", out_path)

    @staticmethod
    def outputs_exist(*paths):
        return all(path.exists() for path in paths)

    @staticmethod
    def _group(values):
        if not isinstance(next(iter(values)), tuple):
            return {k: [v] for k, v in values.items()}

        return {
            k: [v for (_, param), v in values.items() if param == k]
            for k in sorted({k for _, k in values})
        }