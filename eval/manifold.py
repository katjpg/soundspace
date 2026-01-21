from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances


FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]

L2_EPS = 1e-12
DEFAULT_SUBSAMPLE_N = 5000


@dataclass(frozen=True, slots=True)
class ProjectionQuality:
    """
    UMAP projection fidelity metrics.

    Args
    ----
        trustworthiness (float) : fraction of local neighborhoods preserved.
        continuity      (float) : fraction of similar items that stay close.
        shepard_rho     (float) : Spearman correlation of pairwise distances.
        k                 (int) : neighborhood size used for evaluation.
        n_samples         (int) : number of samples evaluated.
    """

    trustworthiness: float
    continuity: float
    shepard_rho: float
    k: int
    n_samples: int


def score_projection_quality(
    X_high: FloatArray,
    X_low: FloatArray,
    *,
    k: int = 15,
    metric_high: str = "cosine",
    metric_low: str = "euclidean",
    subsample_n: int | None = DEFAULT_SUBSAMPLE_N,
    seed: int = 42,
) -> ProjectionQuality:
    """
    Measure neighborhood preservation from high-D to low-D projection.

    Args
    ----
        X_high       (FloatArray) : high-D array of shape (n, d_high).
        X_low        (FloatArray) : low-D array of shape (n, d_low).
        k                  (int)  : neighborhood size for trustworthiness.
                                    (Default is 15).
        metric_high        (str)  : distance metric for high-D space.
                                    (Default is "cosine").
        metric_low         (str)  : distance metric for low-D space.
                                    (Default is "euclidean").
        subsample_n (int | None)  : if set, subsample to this many points
                                    to avoid O(n^2) memory.
                                    (Default is 5000).
        seed               (int)  : RNG seed for subsampling.
                                    (Default is 42).

    Returns
    -------
        (ProjectionQuality) : trustworthiness, continuity, and Shepard correlation.
    """
    if X_high.ndim != 2 or X_low.ndim != 2:
        raise ValueError("X_high and X_low must be 2D arrays")

    n_full = len(X_high)
    if n_full != len(X_low):
        raise ValueError(
            f"shape mismatch: X_high has {n_full} rows, X_low has {len(X_low)} rows"
        )

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if n_full < 2:
        return ProjectionQuality(
            trustworthiness=0.0,
            continuity=0.0,
            shepard_rho=0.0,
            k=k,
            n_samples=n_full,
        )

    # subsample if needed
    X_high_eval, X_low_eval = _subsample(
        X_high, X_low, subsample_n=subsample_n, seed=seed
    )
    n = len(X_high_eval)

    if n <= k:
        return ProjectionQuality(
            trustworthiness=0.0,
            continuity=0.0,
            shepard_rho=0.0,
            k=k,
            n_samples=n,
        )

    # trustworthiness requires k < n/2 for proper scaling
    k_eval = min(k, (n // 2) - 1)
    if k_eval < 1:
        k_eval = 1

    # compute pairwise distances
    dist_high = pairwise_distances(X_high_eval, metric=metric_high).astype(
        np.float64, copy=False
    )
    dist_low = pairwise_distances(X_low_eval, metric=metric_low).astype(
        np.float64, copy=False
    )

    # compute kNN indices
    nn_high = _knn_from_distances(dist_high, k=k_eval)
    nn_low = _knn_from_distances(dist_low, k=k_eval)

    # compute inverse ranks for penalty calculation
    inv_rank_high = _inverse_ranks(dist_high)
    inv_rank_low = _inverse_ranks(dist_low)

    # compute trustworthiness and continuity
    trustworthiness, continuity = _trustworthiness_continuity(
        nn_high, nn_low, inv_rank_high, inv_rank_low, n=n, k=k_eval
    )

    # compute Shepard correlation
    shepard_rho = _shepard_correlation(dist_high, dist_low, seed=seed)

    return ProjectionQuality(
        trustworthiness=trustworthiness,
        continuity=continuity,
        shepard_rho=shepard_rho,
        k=k_eval,
        n_samples=n,
    )


def _subsample(
    X_high: FloatArray,
    X_low: FloatArray,
    *,
    subsample_n: int | None,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Subsample both arrays with same indices."""
    n = len(X_high)
    if subsample_n is None or subsample_n >= n:
        return (
            np.asarray(X_high, dtype=np.float64),
            np.asarray(X_low, dtype=np.float64),
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=subsample_n, replace=False)
    return (
        np.asarray(X_high[idx], dtype=np.float64),
        np.asarray(X_low[idx], dtype=np.float64),
    )


def _knn_from_distances(dist: FloatArray, *, k: int) -> IntArray:
    """Compute kNN indices from distance matrix, excluding self."""
    n = len(dist)
    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)

    # use argpartition for efficiency
    idx = np.argpartition(masked, kth=k - 1, axis=1)[:, :k]

    # sort within the k neighbors by distance
    row = np.arange(n)[:, None]
    row_d = masked[row, idx]
    order = np.argsort(row_d, axis=1)

    return idx[row, order].astype(np.int64, copy=False)


def _inverse_ranks(dist: FloatArray) -> IntArray:
    """Compute inverse rank mapping for each row."""
    n = len(dist)
    masked = dist.copy()
    np.fill_diagonal(masked, -1.0)  # self gets rank 0

    order = np.argsort(masked, axis=1)
    inv = np.empty_like(order, dtype=np.int64)
    row = np.arange(n)[:, None]
    inv[row, order] = np.arange(n)[None, :]

    return inv


def _trustworthiness_continuity(
    nn_high: IntArray,
    nn_low: IntArray,
    inv_rank_high: IntArray,
    inv_rank_low: IntArray,
    *,
    n: int,
    k: int,
) -> tuple[float, float]:
    """Compute trustworthiness and continuity from kNN structures."""
    # denominator for normalization
    denom = float(n * k * (2 * n - 3 * k - 1))
    if denom <= 0:
        return 0.0, 0.0

    trust_sum = 0.0
    cont_sum = 0.0

    for i in range(n):
        high_set = set(int(x) for x in nn_high[i].tolist())
        low_set = set(int(x) for x in nn_low[i].tolist())

        # false neighbors in low-D (trustworthiness penalty)
        for j in nn_low[i]:
            if int(j) not in high_set:
                trust_sum += float(int(inv_rank_high[i, int(j)]) - k)

        # missing neighbors in low-D (continuity penalty)
        for j in nn_high[i]:
            if int(j) not in low_set:
                cont_sum += float(int(inv_rank_low[i, int(j)]) - k)

    factor = 2.0 / denom
    trustworthiness = float(1.0 - factor * trust_sum)
    continuity = float(1.0 - factor * cont_sum)

    return trustworthiness, continuity


def _shepard_correlation(
    dist_high: FloatArray,
    dist_low: FloatArray,
    *,
    seed: int,
    n_pairs: int = 5000,
) -> float:
    """Compute Spearman correlation between sampled pairwise distances."""
    n = len(dist_high)
    if n < 2:
        return 0.0

    max_pairs = (n * (n - 1)) // 2
    n_pairs_eff = min(n_pairs, max_pairs)

    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=n_pairs_eff)
    idx_j = rng.integers(0, n, size=n_pairs_eff)

    # ensure i != j
    same = idx_i == idx_j
    while np.any(same):
        idx_j[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = idx_i == idx_j

    d_high = dist_high[idx_i, idx_j]
    d_low = dist_low[idx_i, idx_j]

    result = spearmanr(d_high, d_low)
    rho = float(result[0]) if hasattr(result, "__getitem__") else float(result.statistic)  # type: ignore[union-attr]
    return 0.0 if np.isnan(rho) else rho
