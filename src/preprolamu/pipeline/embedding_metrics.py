"""
Unsupervised embedding-quality metrics.

Adapted from:
https://github.com/google-research/google-research/tree/master/graph_embedding/metrics
"""
from __future__ import annotations

import json
import logging
import numpy as np
from IsoScore import IsoScore

logger = logging.getLogger(__name__)

EMBEDDING_METRIC_SAMPLE_SIZE = 400_000
EMBEDDING_METRIC_SEED = 42

# NOTE: RankMe is predictive of OOD performance for Joint-Embedding self-supervised learning tasks.
# Perhaps it is therefore also informative for cross-dataset generalization in the current study.

# More intuitive version of the original function that computes all metrics at once.
def embedding_metrics(X: np.ndarray, *, sample_size: int = EMBEDDING_METRIC_SAMPLE_SIZE) -> dict[str, float]:
    X = sample_embedding(X, size=sample_size)
    
    """Compute unsupervised embedding-quality metrics for a given embedding matrix."""
    u, s, _ = np.linalg.svd(X, compute_uv=True, full_matrices=False)

    return {
        "rankme": rankme(X, s=s),
        "rankme_modified": rankme_modified(X, s=s),
        "coherence": coherence(X, u=u),
        "pseudo_condition_number": pseudo_condition_number(X, s=s),
        "alpha_req": alpha_req(X, s=s),
        "stable_rank": stable_rank(X, s=s),
        "ne_sum": ne_sum(X),
        # "self_clustering": self_clustering(X),  # Disabled due to difficulty with large arrays.
        "isoscore": isoscore(X),
    }


def save_embedding_metrics(
    universe,
    *,
    split: str = "test",
    overwrite: bool = False,
):
    path = universe.paths.embedding_metrics(split=split)

    if path.exists() and not overwrite:
        logger.info(f"Embedding metrics already exist at {path}. Skipping.")
        return

    latent = universe.io.load_embedding(split=split)
    metrics = embedding_metrics(latent)

    path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")


# Original aggregating function that is deprecated in favor of the above function.
def report_all_metrics(tensor):
    """Computes all metric values given a tensor and its SVD.

    Args:
      tensor (dense matrix): Input embeddings.

    Returns:
      Mapping[str, float]: All metric values.
    """
    # Pre-compute SVD for metric computations.
    u, s, _ = np.linalg.svd(tensor, compute_uv=True, full_matrices=False)
    fns = [
        rankme,
        coherence,
        pseudo_condition_number,
        alpha_req,
        stable_rank,
        ne_sum,
        # self_clustering,  # Disabled due to difficulty with large arrays.
        isoscore,
    ]
    return dict((fn.__name__, fn(tensor, u=u, s=s)) for fn in fns)


def sample_embedding(
    X: np.ndarray, 
    *, 
    size: int = EMBEDDING_METRIC_SAMPLE_SIZE, 
    seed: int = EMBEDDING_METRIC_SEED
):
    if len(X) < size:
        raise ValueError(f"Cannot sample {size} points from embedding of size {len(X)}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=size, replace=False)
    return X[indices]


def pseudo_condition_number(tensor, s=None, epsilon=1e-12, **_):
    """Implementation of the pseudo-condition number metric.
    Interpretation: Smallest vs largest singular value

    Args:
      tensor (dense matrix): Input embeddings.
      s (optional, dense vector): Singular values of `tensor`.
      epsilon (float): Numerical epsilon.

    Returns:
      float: Pseudo-condition number metric value.
    """
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)
    return s[-1] / (s[0] + epsilon)


def coherence(tensor, u=None, **_):
    """Implementation of the coherence metric.

    Args:
      tensor (dense matrix): Input embeddings.
      u (optional, dense matrix): Left singular vectors of `tensor`.

    Returns:
      float: Coherence metric value.
    """
    if u is None:
        u, _, _ = np.linalg.svd(tensor, compute_uv=True, full_matrices=False)
    maxu = np.linalg.norm(u, axis=1).max() ** 2
    return maxu * u.shape[0] / u.shape[1]


def stable_rank(tensor, s=None, epsilon=1e-12, **_):
    """Implementation of the stable rank metric.

    Args:
      tensor (dense matrix): Input embeddings.
      s (optional, dense vector): Singular values of `tensor`.
      epsilon (float): Numerical epsilon.

    Returns:
      float: Stable rank metric value.
    """
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)
    trace = np.square(tensor).sum()
    denominator = s[0] * s[0] + epsilon
    return trace / denominator


def self_clustering(tensor, epsilon=1e-12, **_):
    """Implementation of the SelfCluster metric.

    Args:
      tensor (dense matrix): Input embeddings.
      epsilon (float): Numerical epsilon.

    Returns:
      float: SelfCluster metric value.
    """
    tensor = tensor + epsilon
    tensor /= np.linalg.norm(tensor, axis=1)[:, np.newaxis]
    n, d = tensor.shape
    expected = n + n * (n - 1) / d
    actual = np.sum(np.square(tensor @ tensor.T))
    return (actual - expected) / (n * n - expected)


def rankme(tensor, s=None, epsilon=1e-12, **_):
    """Implementation of the RankMe metric.
    Interpretation: effective dimensionality from entropy of singular values.

    This metric is defined in "RankMe: Assessing the Downstream Performance of
    Pretrained Self-Supervised Representations by Their Rank". Garrido et al.
    arXiv:2210.02885.

    Args:
      tensor (dense matrix): Input embeddings.
      s (optional, dense vector): Singular values of `tensor`.
      epsilon (float): Numerical epsilon.

    Returns:
      float: RankMe metric value.
    """
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)

    # Thought: Shouldn't this be: p_ks = s / np.sum(s) + epsilon? See modified version below
    p_ks = s / np.sum(s + epsilon) + epsilon
    return np.exp(-np.sum(p_ks * np.log(p_ks)))


def rankme_modified(tensor, s=None, epsilon=1e-12, **_):
    """Modified implementation of the RankMe metric.
    The google source seems to add epsilon twice while the arxiv paper seems to add it only once. This version adds it only once.
    """
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)

    # Modified here.
    p_ks = s / np.sum(s) + epsilon
    return np.exp(-np.sum(p_ks * np.log(p_ks)))


def ne_sum(tensor, epsilon=1e-12, **_):
    """Implementation of the NESum metric.

    This metric is defined in "Exploring the Gap between Collapsed & Whitened
    Features in Self-Supervised Learning". He & Ozay, ICML 2022. See Definition
    4.1 from the paper for more details.

    Args:
      tensor (dense matrix): Input embeddings.
      epsilon (float): Numerical epsilon.

    Returns:
      float: NESum metric value.
    """
    cov_t = np.cov(tensor.T)
    ei_t = np.linalg.eigvalsh(cov_t) + epsilon
    return (ei_t / ei_t[-1]).sum()


def alpha_req(tensor, s=None, epsilon=1e-12, **_):
    """Implementation of the Alpha-ReQ metric.

    This metric is defined in "α-ReQ: Assessing representation quality in
    self-supervised learning by measuring eigenspectrum decay". Agrawal et al.,
    NeurIPS 2022.

    Args:
      tensor (dense matrix): Input embeddings.
      s (optional, dense vector): Singular values of `tensor`.
      epsilon (float): Numerical epsilon.

    Returns:
      float: Alpha-ReQ metric value.
    """
    if s is None:
        s = np.linalg.svd(tensor, compute_uv=False)
    n = s.shape[0]
    s = s + epsilon
    features = np.vstack([np.linspace(1, 0, n), np.ones(n)]).T
    a, _, _, _ = np.linalg.lstsq(features, np.log(s), rcond=None)
    return a[0]


def isoscore(points, **_):
    """Implementation wrapper for the IsoScore metric.
    Interpretation: Uniformity of variance across dimensions

    This metric is defined in "IsoScore: Measuring the Uniformity
    of Embedding Space Utilization". Rudman et al., ACL 2022.

    Args:
      points (dense matrix): Input embeddings.

    Returns:
      float: IsoScore metric value.
    """
    return IsoScore.IsoScore(points)
