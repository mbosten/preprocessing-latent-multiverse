import logging

import numpy as np
import gudhi as gd
from gudhi.representations import Landscape
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class Persistence:
    """Compute, analyze, and plot persistence intervals and landscapes."""

    def __init__(self, universe=None, points=None, intervals=None, landscapes=None):
        self.universe = universe
        self.points = points
        self.intervals = intervals
        self.landscapes = landscapes
        self.num_landscapes = None
        self.resolution = None
        self.summed_persistence = None
        self.norms = None

        config = universe.tda_config if universe is not None else None
        self.num_landscapes = getattr(config, "num_landscapes", None)
        self.resolution = getattr(config, "resolution", None)
    
    # ──────────────────────────────────────────────────────────
    # Persistence                                                 
    # ──────────────────────────────────────────────────────────

    def compute_intervals(self, precision="exact", hom_dims: tuple[int, ...]|None=None):
        """
        Computes Alpha complex persistence intervals on points in case intervals required but not set.
        If hom_dims is None, it will use the homology dimensions from the universe's TDA config, or default to (0, 1, 2) if not available.
        """
        if self.points is None:
            raise RuntimeError("Points are not set. Cannot compute intervals.")

        hom_dims = hom_dims or (
            self.universe.tda_config.homology_dimensions
            if self.universe is not None
            else (0, 1, 2)
        )

        st = gd.AlphaComplex(points=self.points, precision=precision).create_simplex_tree()
        st.compute_persistence(homology_coeff_field=2)
        self.intervals = {dim: self._finite(st.persistence_intervals_in_dimension(dim)) for dim in hom_dims}
        return self.intervals

        
    def compute_landscapes(self, num_landscapes=None, resolution=None, hom_dims: tuple[int, ...]|None=None):

        intervals = self._require_intervals()

        num_landscapes = num_landscapes or self.num_landscapes or 5
        resolution = resolution or self.resolution or 1000
        hom_dims = hom_dims or (
            self.universe.tda_config.homology_dimensions
            if self.universe is not None
            else (0, 1, 2)
    )

        self.num_landscapes = num_landscapes
        self.resolution = resolution

        LS = Landscape(resolution=resolution, keep_endpoints=False, num_landscapes=num_landscapes)

        self.landscapes = {}

        for dim in hom_dims:
            pairs = intervals.get(dim)
            self.landscapes[dim] = (LS.fit_transform([pairs]) if pairs is not None and len(pairs) else None)

        return self.landscapes

    # ──────────────────────────────────────────────────────────
    # Metrics                                                 
    # ──────────────────────────────────────────────────────────

    def total_persistence(self):
        self.summed_persistence =  {
            dim: float(np.sum(diagram[:, 1] - diagram[:, 0]))
            for dim, diagram in self._require_intervals().items()
        }
        return self.summed_persistence


    def landscape_norms(self):
        self.norms =  {
            dim: None if landscape is None else float(np.linalg.norm(landscape))
            for dim, landscape in self._require_landscapes().items()
        }
        return self.norms


    def metrics(self):
        return {
            "total_persistence": self.total_persistence(),
            "landscape_norms": self.landscape_norms(),
        }

    
    # ──────────────────────────────────────────────────────────
    # Plotting                                                 
    # ──────────────────────────────────────────────────────────

    # implement saving functionality
    def plot_persistence_diagram(self):
        intervals = self._require_intervals()
        persistence = [(dim, tuple(interval)) for dim, diagram in intervals.items() for interval in diagram]
        ax = gd.plot_persistence_diagram(sorted(persistence, reverse=True), legend=True)
        return ax.figure, ax

    def plot_persistence_barcode(self, max_intervals=50):
        intervals = self._require_intervals()
        hom_dims = sorted(intervals)

        fig, axes = plt.subplots(1, len(hom_dims), figsize=(8 * len(hom_dims), 5), squeeze=False)
        axes = axes.ravel()

        for ax, dim in zip(axes, hom_dims):
            gd.plot_persistence_diagram(intervals[dim], max_intervals=max_intervals, axes=ax)
            ax.set_title(f"H{dim} persistence barcode")

        fig.tight_layout()
        return fig, axes
    
    def plot_landscape(self, num_landscapes=None, resolution=None):
        landscapes = [
            (dim, landscape)
            for dim, landscape in sorted(self._require_landscapes().items())
            if landscape is not None
        ]

        if not landscapes:
            raise RuntimeError("No persistence landscapes available.")

        num_landscapes = self.num_landscapes or num_landscapes
        resolution = self.resolution or resolution

        if None in (num_landscapes, resolution):
            raise ValueError("provide num_landscapes and resolution, or run compute_landscapes() first.")
        fig, axes = plt.subplots(len(landscapes), 1, figsize=(10, 3 * len(landscapes)), squeeze=False)

        for ax, (dim, landscape) in zip(axes.flat, landscapes):
            if landscape.size != num_landscapes * resolution:
                raise ValueError(f"Landscape H_{dim} has size {landscape.size}, expected {num_landscapes * resolution}.")
            curves = landscape.reshape(num_landscapes, resolution)

            for i, curve in enumerate(curves, start=1):
                ax.plot(curve, label=f"lambda {i}")

            ax.set_title(fr"$H_{dim}$ persistence landscape")
            ax.legend()

        fig.tight_layout()
        return fig, axes


    def _require_intervals(self):
        if self.intervals is None:
            logger.info("Intervals are not set. Computing intervals now.")
            self.compute_intervals()
        return self.intervals

    def _require_landscapes(self):
        if self.landscapes is None:
            logger.info("Landscapes are not set. Computing landscapes now.")
            self.compute_landscapes()
        return self.landscapes

    @staticmethod
    def _finite(intervals):
        return intervals[np.isfinite(intervals[:, 1])]