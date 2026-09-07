"""
Unsupervised embedding-quality metrics.

Adapted from:
https://github.com/google-research/google-research/tree/master/graph_embedding/metrics
"""
from __future__ import annotations

import json
import logging
import mlpack
import numpy as np
from IsoScore import IsoScore
from scipy import sparse
from time import perf_counter


logger = logging.getLogger(__name__)

EMBEDDING_METRIC_SAMPLE_SIZE = 400_000
EMBEDDING_METRIC_SEED = 42

# NOTE: RankMe is predictive of OOD performance for Joint-Embedding self-supervised learning tasks.
# Perhaps it is therefore also informative for cross-dataset generalization in the current study.

# More intuitive version of the original function that computes all metrics at once.
def embedding_metrics(
        X: np.ndarray, 
        *, 
        sample_size: int = EMBEDDING_METRIC_SAMPLE_SIZE,
        mst_dimension_alpha: float = 2.0,
        mst_local_dimension_hops: tuple[int, ...] | None = (1, 2, 4, 8, 16, 32),
        mst_local_dimension_sample_size: int = 1024,
        return_local_details: bool = False,
    ) -> dict[str, float]:

    logger.debug("Sampling embedding at size %d for metric computation.", sample_size)
    X = sample_embedding(X, size=sample_size)
    
    """Compute unsupervised embedding-quality metrics for a given embedding matrix."""
    u, s, v = np.linalg.svd(X, compute_uv=True, full_matrices=False)

    edges = mst(X)

    if mst_local_dimension_hops is not None:
        local_mst = mst_local_dimension_utilization(
            X,
            edges,
            hops=mst_local_dimension_hops,
            alpha=mst_dimension_alpha,
            sample_size=mst_local_dimension_sample_size,
            seed=EMBEDDING_METRIC_SEED,
            return_details=return_local_details,
        )

    return {
        "rankme": rankme(X, s=s),
        "rankme_modified": rankme_modified(X, s=s),
        "coherence": coherence(X, u=u),
        "coherence_modified": coherence_modified(X, u=u, v=v),
        "pseudo_condition_number": pseudo_condition_number(X, s=s),
        "alpha_req": alpha_req(X, s=s),
        "stable_rank": stable_rank(X, s=s),
        "ne_sum": ne_sum(X),
        # "self_clustering": self_clustering(X),  # Disabled due to memory issues with large arrays.
        "isoscore": isoscore(X),
        "MST_length": mst_length(edges),
        "MST_dimension_utilization": mst_dimension_utilization(X, edges, alpha=mst_dimension_alpha),
        "MST_local_dimension_utilization": None if mst_local_dimension_hops is None else local_mst,
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



def coherence_modified(tensor, u=None, v=None, **_):
    """Modified implementation of the COMPLETE coherence metric as defined in the paper.

    It is unclear to me why only the above function has been implemented since it seems that we need both the U and V parts.
    """
    # compute SVD
    if u is None or v is None:
        u, _, v = np.linalg.svd(tensor, compute_uv=True, full_matrices=False)


    # find the rank of the matrix 
    r = np.linalg.matrix_rank(tensor)

    if r == 0:
        raise ValueError("mu_0 incoherence is undefined for the zero matrix.")

    # Compact SVD
    u = u[:, :r]
    v = v[:r, :].T

    mu_u = u.shape[0] / r * np.linalg.norm(u, axis=1).max() ** 2
    mu_v = v.shape[0] / r * np.linalg.norm(v, axis=1).max() ** 2

    return max(mu_u, mu_v)


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
    Measures the uniformity of variance across dimensions.
    output interpretation: proportion of Rn dimensions utilized. 

    This metric is defined in "IsoScore: Measuring the Uniformity
    of Embedding Space Utilization". Rudman et al., ACL 2022.

    Args:
      points (dense matrix): Input embeddings.

    Returns:
      float: IsoScore metric value.
    """
    return IsoScore.IsoScore(points)


def mst(X: np.ndarray) -> np.ndarray:
    """Compute the Euclidean minimum spanning tree."""
    return mlpack.emst(
        input_=X,
        leaf_size=1,
        naive=False,
    )["output"]


def mst_length(edges: np.ndarray) -> float:
        return float(np.sum(edges[:, 2]))


def _entropy_effective_rank(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Compute the entropy effective rank of a matrix."""

    # normalized eigenvalues.
    eig = np.clip(np.linalg.eigvalsh(matrix), 0, None)
    total = eig.sum(axis=-1, keepdims=True)
    p = eig / np.maximum(total, epsilon)

    # entropy in nats
    log_p = np.zeros_like(p)
    np.log(p, out=log_p, where=p > 0)

    # exponent to set rank on the same scale as the number of dimensions.
    rank = np.exp(-np.sum(p * log_p, axis=-1))

    return np.where(total[..., 0] > epsilon, rank, 0.0)


def _mst_features(X, edges, alpha):
    """Computes the following features:
    - ij:       Indices of the nodes connected by each edge
    - length:   Length of each edge
    - d:        Displacement vector for each edge
    - q:        Unit direction (displacement divided by length)

    In q we scale length by alpha/2 to give more or less weight to longer edges.
    """
    ij = edges[:, :2].astype(np.intp)
    length = edges[:, 2]

    # displacement vector for each edge
    d = X[ij[:, 1]] - X[ij[:, 0]]

    # unit direction scaled by length^(alpha/2) to give more or less weight to longer edges.
    q = d / np.maximum(length[:, None], 1e-12) * length[:, None] ** (alpha / 2)
    return ij, q


def _knee_point(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xn  = (x - x[0]) / (x[-1] - x[0])
    yn  = (y - y[0]) / (y[-1] - y[0])
    return np.argmax(yn - xn)


def mst_dimension_utilization(X, edges, alpha, **_):
    """Effective number of dimensions utilized by the MST."""
    _, q = _mst_features(X, edges, alpha)
    # outer product ensures that undirected edges do not cancel each other.
    return float(_entropy_effective_rank(q.T @ q))


def mst_local_dimension_utilization(
        X,
        edges,
        *,
        hops: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        alpha: float = 1.0,
        sample_size: int = 1024,
        seed: int = EMBEDDING_METRIC_SEED,
        return_details: bool = False,
) -> dict[int, np.ndarray]:
    """
    Distribution of MST dimension utilization across hop scales. For each hop scale, we compute the effective number of dimensions utilized by the MST edges that are within that hop distance from each node.
    The result is a dictionary mapping hop distances to arrays of effective dimensions for each node in the graph.
    It provides a local view of dimension utilization, complementary to the global MST dimension utilization metric.
    """
    ij, q = _mst_features(X, edges, alpha)
    n = len(X)

    # adjacency matrix
    A = sparse.csr_matrix(
        (
            np.ones(2 * len(ij), dtype=bool),
            (
                ij[:, [0, 1]].ravel(),
                ij[:, [1, 0]].ravel(),
            ),
        ),
        shape=(n, n),
    )

    rng = np.random.default_rng(seed)
    centers = rng.choice(n, size=min(sample_size, n), replace=False)

    # Points reachable from each center.
    R = sparse.csr_matrix(
        (
            np.ones(len(centers), dtype=bool),
            (np.arange(len(centers)), centers),
        ),
        shape=(len(centers), n),
    )

    hops = tuple(sorted(set(hops)))
    medians = {}
    details = {}


    for hop in range(1, max(hops) + 1):
        
        # Expand every center's neighborhood by exactly one MST hop (multiplying by the adjancency matrix).
        R = (R + R @ A).astype(bool)

        if hop not in hops:
            continue
        
        logger.debug("Computing MST utilization at hop %d", hop)

        # Since R is a boolean matrix with the number of centers as rows and the number of nodes as columns,
        # multiplication tells us whether both nodes of an edge are reachable from the same center.
        E = R[:, ij[:, 0]].multiply(R[:, ij[:, 1]]).tocsr()


        # Take outer product of q subset corresponding to edges reachable from each center.
        # Resulting array is N x D x D.
        matrices = np.stack([
            q[idx].T @ q[idx]
            for idx in np.split(E.indices, E.indptr[1:-1])
        ])

        dimensions = _entropy_effective_rank(matrices)
        medians[hop] = float(np.median(dimensions))

        if return_details:
            details[hop] = {
                "dimension": dimensions,
                "median": medians[hop],
                "edge_coverage": float(np.mean(E.getnnz(axis=0) > 0)),
                "edges": E.getnnz(axis=1),
                "matrices": matrices,
                "centers": centers,
            }

    hop_values = np.asarray(list(medians))
    median_values = np.asarray(list(medians.values()))

    i = _knee_point(hop_values, median_values)
    knee_hop = int(hop_values[i])
    estimate = float(median_values[i])

    if return_details:
        return {
            "dimension": estimate,
            "knee_hop": knee_hop,
            "hops": details,
        }

    return estimate