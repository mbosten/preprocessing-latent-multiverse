from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class Embedding:
    def __init__(self, latent_space=None, universe=None):
        self.universe = universe
        self.operations = {}
        self.scale = None
        self._latent_space = None
        self.latent_space = latent_space

    def _reset_operations(self):
        self.operations = {
            "normalized": None,
            "pca": None,
            "sampled": None,
        }

    def _get_seed(self, seed=None):
        if seed is not None:
            return seed
        if self.universe is not None and hasattr(self.universe, "seed"):
            return self.universe.seed
        return 42

    @property
    def latent_space(self):
        if self._latent_space is None:
            raise RuntimeError("Latent space is not set.")
        return self._latent_space

    @latent_space.setter
    def latent_space(self, value):
        self._latent_space = None if value is None else value.copy()
        self.scale = None
        self._reset_operations()


    def normalize(self, method="diameter", seed=None, inplace=True, **kwargs):
        """
        Normalize the latent space using either the diameter or a quantile-based scale.
        Seed=None uses the Universe's seed if available, otherwise defaults to 42.

        Supported methods:
        1. Diameter normalization: maximum distance between points is 1. Requires 'iterations' keyword argument (default 1000).
        2. Quantile normalization: specified quantile of pairwise distances is 1. Requires 'q' keyword argument (default 0.999).
        """
        
        seed = self._get_seed(seed)
        params = (method, seed, tuple(sorted(kwargs.items())))
        rng = np.random.default_rng(seed)
        X = self.latent_space
        eps = 1e-8

        logger.info("[Embedding] parameter: %s", params)
        logger.info("[Embedding] Normalizing.")

        if inplace:
            if self.operations["normalized"] == params:
                logger.warning(f"[Embedding] Latent space is already normalized with parameters {params}. Skipping normalization.")
                return

            if self.operations["normalized"] is not None:
                raise RuntimeError("Normalization was already applied with different parameters.")

        if method == "diameter":
            iterations = kwargs.get("iterations", 1000)

            subset = [rng.choice(len(X))]
            for _ in range(iterations - 1):
                distances = cdist([X[subset[-1]]], X).ravel()
                new_point = np.argmax(distances)
                subset.append(new_point)
            
                pairwise_distances = cdist(X[subset], X[subset])
                scale = np.max(pairwise_distances)        

        elif method == "quantile":
            q = kwargs.get("q", 0.999)
            n_pairs = 2_000_000
            batch_size = 250_000

            distances = np.empty(n_pairs, dtype=np.float64)
            
            start = 0
            # Compute over batches to avoid memory issues.
            while start < n_pairs:
                stop = min(start + batch_size, n_pairs)
                m = stop - start
            
                i = rng.integers(len(self.latent_space), size=m)
                j = rng.integers(len(self.latent_space) - 1, size=m)
                j += j >= i  # ensure i != j
            
                # Compute differences
                diff = X[i] - X[j]
            
                # Squared distances
                distances[start:stop] = np.einsum("ij,ij->i", diff, diff)
            
                start = stop
            
            k = min(int(np.ceil(q * n_pairs)) - 1, n_pairs - 1)
            
            # partition data on the index of the q-value instead of full sorting
            scale = float(np.sqrt(np.partition(distances, k)[k]))
            
        else:
            raise ValueError(f"Unknown normalization method: {method}. Supported methods are 'diameter' and 'quantile'.")

        if not np.isfinite(scale) or scale < eps:
            logger.info(
                "[EMB] Computed scale/diameter is non-finite or too small; defaulting to 1 (scale=%s).",
                scale,
            )
            scale = 1.0       

        self.scale = scale
        
        if inplace:
            self._latent_space /= self.scale
            self.operations["normalized"] = params

        return self.scale.copy()
        

    def project_PCA(self, n_components=3, seed=None, inplace=True):
        """
        Project the latent space to a lower-dimensional space using PCA.
        Leave seed=None to use the Universe's seed if available, otherwise defaults to 42.
        If inplace=True, _latent_space is modified, the operation is recorded and the scale parameter is updated.)
        """
        
        seed = self._get_seed(seed)
        params = (n_components, seed)

        logger.info("[Embedding] Projecting.")

        if inplace:
            if self.operations["pca"] == params:
                logger.warning("[Embedding] Skipping PCA because projection with identical params already exists.")
                return

            if self.operations["pca"] is not None:
                raise RuntimeError("PCA projection was already applied with different parameters.")
        
        pca = PCA(n_components=n_components, random_state=seed)
        projected = pca.fit_transform(self.latent_space)

        if inplace:
            self._latent_space = projected
            self.operations["pca"] = params
            logger.info(f"[EMB] PCA projection shape: {self._latent_space.shape}")

        return projected.copy()


    def sample(self, target_size=1000, seed=None, inplace=True):
        """
        Sample the latent space, optionally without modifying the object.
        Leave seed=None to use the Universe's seed if available, otherwise defaults to 42.
        """
        
        seed = self._get_seed(seed)
        params = (target_size, seed)

        logger.info("[Embedding] Sampling.")

        if inplace:
            if self.operations["sampled"] == params:
                logger.warning(f"[Embedding] Latent space is already sampled to {target_size} points. Skipping sampling.")
                return

            if self.operations["sampled"] is not None:
                raise RuntimeError("Sampling was already applied with different parameters.")

        target_size = min(target_size, len(self.latent_space))

        if target_size < len(self.latent_space):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.latent_space), size=target_size, replace=False)
            sampled = self.latent_space[indices]
        else:
            logger.info("[Embedding] Target size is greater than or equal to the latent space size.")
            sampled = self.latent_space

        if inplace:
            self._latent_space = sampled
            self.operations["sampled"] = params

        return sampled.copy()


    def save(self, split="test"):
        logger.info("[Embedding] Saving.")
        self.universe.io.save_projected(
            split=split,
            normalized=True if self.operations["normalized"] is not None else False,
            arr=self.latent_space.astype(np.float32, copy=False),
        )