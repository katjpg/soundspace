from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True, slots=True)
class Trustworthiness:
    """Projection fidelity metrics."""

    score: float
    continuity: float
    lcmc: float
    diagnostics: dict[str, float] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Shepard:
    """Rank correlation between high-D and low-D pairwise distances."""

    spearman: float
    kendall: float
    diagnostics: dict[str, float] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_trustworthiness(X_high: np.ndarray, X_low: np.ndarray, k: int = 15) -> Trustworthiness:
    """
    Measure neighborhood preservation from high-D to low-D.

    Args:
        X_high (np.ndarray) : high-D array (n, d).
        X_low  (np.ndarray) : low-D array (n, 2 or 3).
        k      (int)        : neighborhood size.

    Returns:
        (Trustworthiness) : trustworthiness, continuity, and overlap fraction.

    Notes:
        - Uses cosine in high-D and euclidean in low-D.
        - O(n^2) time; intended for diagnostics, not hot paths.

    Example:
        >>> report = score_trustworthiness(X_high, X_low, k=15)
    """
    if X_high.ndim != 2 or X_low.ndim != 2:
        raise ValueError("X_high and X_low must be 2D arrays")

    n = int(X_high.shape[0])
    if n != int(X_low.shape[0]):
        raise ValueError(f"shape mismatch: high-D has {X_high.shape[0]}, low-D has {X_low.shape[0]}")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    anomalies: list[str] = []
    if n < 2:
        anomalies.append("insufficient samples (n < 2)")
        return Trustworthiness(
            score=0.0,
            continuity=0.0,
            lcmc=0.0,
            diagnostics={"n": float(n), "k": float(k)},
            anomalies=anomalies,
        )

    if n <= k:
        anomalies.append(f"n ({n}) must be > k ({k})")
        return Trustworthiness(
            score=0.0,
            continuity=0.0,
            lcmc=0.0,
            diagnostics={"n": float(n), "k": float(k)},
            anomalies=anomalies,
        )

    denom = float(n * k * (2 * n - 3 * k - 1))
    if denom <= 0.0:
        anomalies.append("invalid denominator for trustworthiness formula (n and k too close)")
        return Trustworthiness(
            score=0.0,
            continuity=0.0,
            lcmc=0.0,
            diagnostics={"n": float(n), "k": float(k)},
            anomalies=anomalies,
        )

    nn_high = _knn_indices(X_high, k=k, metric="cosine")
    nn_low = _knn_indices(X_low, k=k, metric="euclidean")

    trust_sum = 0.0
    cont_sum = 0.0
    overlap_sum = 0.0

    for i in range(n):
        high_set = set(nn_high[i].tolist())
        low_set = set(nn_low[i].tolist())
        overlap_sum += float(len(high_set & low_set))

        false_low = [int(j) for j in nn_low[i] if int(j) not in high_set]
        if false_low:
            inv_rank_high = _inverse_ranks(X_high, row=i, metric="cosine")
            trust_sum += _rank_penalty(inv_rank_high, false_low, k=k)

        false_high = [int(j) for j in nn_high[i] if int(j) not in low_set]
        if false_high:
            inv_rank_low = _inverse_ranks(X_low, row=i, metric="euclidean")
            cont_sum += _rank_penalty(inv_rank_low, false_high, k=k)

    factor = 2.0 / denom
    trustworthiness = float(1.0 - factor * trust_sum)
    continuity = float(1.0 - factor * cont_sum)
    lcmc = float(overlap_sum / float(n * k))

    diagnostics = {
        "n": float(n),
        "k": float(k),
        "high_d_dims": float(X_high.shape[1]),
        "low_d_dims": float(X_low.shape[1]),
    }

    return Trustworthiness(
        score=trustworthiness,
        continuity=continuity,
        lcmc=lcmc,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def score_shepard(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_pairs: int = 5000,
    seed: int = 42,
) -> Shepard:
    """
    Measure distance-rank preservation via Spearman and Kendall correlation.

    Args:
        X_high  (np.ndarray) : high-D array (n, d).
        X_low   (np.ndarray) : low-D array (n, 2 or 3).
        n_pairs (int)        : number of random pairs to sample.
        seed    (int)        : RNG seed.

    Returns:
        (Shepard) : rank correlations between sampled pairwise distances.

    Notes:
        - High-D uses cosine; low-D uses euclidean.
        - Subsampling pairs keeps runtime near O(n_pairs).

    Example:
        >>> report = score_shepard(X_high, X_low, n_pairs=5000)
    """
    if X_high.ndim != 2 or X_low.ndim != 2:
        raise ValueError("X_high and X_low must be 2D arrays")

    n = int(X_high.shape[0])
    if n != int(X_low.shape[0]):
        raise ValueError(f"shape mismatch: high-D has {X_high.shape[0]}, low-D has {X_low.shape[0]}")

    anomalies: list[str] = []
    if n < 2:
        anomalies.append("insufficient samples (n < 2)")
        return Shepard(
            spearman=0.0,
            kendall=0.0,
            diagnostics={"n": float(n), "n_pairs": 0.0, "seed": float(seed)},
            anomalies=anomalies,
        )

    max_pairs = (n * (n - 1)) // 2
    n_pairs_eff = int(min(max(n_pairs, 1), max_pairs))

    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, n, size=n_pairs_eff, endpoint=False)
    j_idx = rng.integers(0, n, size=n_pairs_eff, endpoint=False)

    same = i_idx == j_idx
    while bool(np.any(same)):
        j_idx[same] = rng.integers(0, n, size=int(np.sum(same)), endpoint=False)
        same = i_idx == j_idx

    dist_high = _cosine_distance_pairs(X_high, i_idx=i_idx, j_idx=j_idx)
    dist_low = _euclidean_distance_pairs(X_low, i_idx=i_idx, j_idx=j_idx)

    spearman_out = spearmanr(dist_high, dist_low)
    kendall_out = kendalltau(dist_high, dist_low)

    rho, _ = cast(tuple[float, float], spearman_out)
    tau, _ = cast(tuple[float, float], kendall_out)

    diagnostics = {
        "n": float(n),
        "n_pairs": float(n_pairs_eff),
        "seed": float(seed),
    }

    return Shepard(
        spearman=float(rho),
        kendall=float(tau),
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def _knn_indices(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Compute kNN indices excluding self."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, nn = nbrs.kneighbors(X)
    return nn[:, 1:]


def _inverse_ranks(X: np.ndarray, row: int, metric: str) -> np.ndarray:
    """Compute inverse rank array for one row (self rank 0)."""
    d = pairwise_distances(X[row : row + 1], X, metric=metric).reshape(-1)
    d[row] = -1.0
    order = np.argsort(d, kind="quicksort")
    inv_rank = np.empty_like(order)
    inv_rank[order] = np.arange(order.size)
    return inv_rank


def _rank_penalty(inv_rank: np.ndarray, indices: list[int], k: int) -> float:
    """Sum (rank - k) for indices using an inverse-rank lookup."""
    penalty = 0.0
    for j in indices:
        penalty += float(int(inv_rank[int(j)]) - k)
    return penalty


def _cosine_distance_pairs(X: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """Compute cosine distances for index pairs."""
    a = X[i_idx]
    b = X[j_idx]
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.where(norms > 0.0, dots / norms, 0.0)
    return 1.0 - cos


def _euclidean_distance_pairs(X: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """Compute euclidean distances for index pairs."""
    diff = X[i_idx] - X[j_idx]
    return np.linalg.norm(diff, axis=1)
