from dataclasses import dataclass, field
import inspect
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import pairwise_distances


FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]
Diagnostics: TypeAlias = dict[str, float]

L2_EPS = 1e-12
MAX_RANK_SAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class Trustworthiness:
    """Projection fidelity metrics."""

    score: float
    continuity: float
    lcmc: float
    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Shepard:
    """Rank correlation between high-D and low-D pairwise distances."""

    spearman: float
    kendall: float
    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_trustworthiness(
    X_high: FloatArray,
    X_low: FloatArray,
    *,
    k: int = 15,
    metric_high: str = "cosine",
    metric_low: str = "euclidean",
    reference: bool = False,
    subsample_n: int | None = None,
    subsample_seed: int = 42,
) -> Trustworthiness:
    """
    Measure neighborhood preservation from high-D to low-D.

    Args
    ----
        X_high         (FloatArray) : high-D array of shape (n, d).
        X_low          (FloatArray) : low-D array of shape (n, 2 or 3).
        k              (int)        : neighborhood size.
                                      (Default is 15).
        metric_high    (str)        : distance metric for high-D.
                                      (Default is "cosine").
        metric_low     (str)        : distance metric for low-D.
                                      (Default is "euclidean").
        reference      (bool)       : if True, attempt to compute trustworthiness using
                                      sklearn.manifold.trustworthiness for cross-checking.
                                      (Default is False).
        subsample_n    (int | None) : if set and < n, evaluate on a random subset of rows
                                      to avoid O(n^2) memory/time.
                                      (Default is None).
        subsample_seed (int)        : RNG seed for subsampling.
                                      (Default is 42).

    Returns
    -------
        (Trustworthiness) : trustworthiness, continuity, and LCMC(k).

    Notes
    -----
        - LCMC is computed as: Q_NX(k) - k/(n-1), where Q_NX is mean kNN overlap fraction.
        - this implementation materializes full pairwise distance matrices, which is O(n^2) memory.
    """
    if X_high.ndim != 2 or X_low.ndim != 2:
        raise ValueError("X_high and X_low must be 2D arrays")

    n_full = int(X_high.shape[0])
    if n_full != int(X_low.shape[0]):
        raise ValueError(
            f"shape mismatch: high-D has {X_high.shape[0]}, low-D has {X_low.shape[0]}"
        )

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    anomalies: list[str] = []

    if n_full < 2:
        anomalies.append("insufficient samples (n < 2)")
        return Trustworthiness(
            score=0.0,
            continuity=0.0,
            lcmc=0.0,
            diagnostics={"n": float(n_full), "k": float(k)},
            anomalies=anomalies,
        )

    if k >= (n_full / 2.0):
        raise ValueError(
            f"k must be < n/2 to keep trustworthiness in [0, 1], got k={k}, n={n_full}"
        )

    if subsample_n is not None and subsample_n <= 0:
        raise ValueError(
            f"subsample_n must be positive when provided, got {subsample_n}"
        )

    if subsample_n is None and n_full > MAX_RANK_SAMPLES:
        raise ValueError(
            f"n ({n_full}) exceeds MAX_RANK_SAMPLES ({MAX_RANK_SAMPLES}); "
            f"set subsample_n to bound O(n^2) memory/time"
        )

    selected_idx = _choose_subsample_indices(
        n=n_full, subsample_n=subsample_n, seed=subsample_seed
    )
    if selected_idx is not None:
        anomalies.append(
            f"subsampled evaluation to n={int(selected_idx.size)} (seed={subsample_seed})"
        )
        X_high_eval = np.asarray(X_high, dtype=np.float64)[selected_idx]
        X_low_eval = np.asarray(X_low, dtype=np.float64)[selected_idx]
    else:
        X_high_eval = np.asarray(X_high, dtype=np.float64)
        X_low_eval = np.asarray(X_low, dtype=np.float64)

    n = int(X_high_eval.shape[0])
    if n <= k:
        anomalies.append(f"n ({n}) must be > k ({k}) after subsampling")
        return Trustworthiness(
            score=0.0,
            continuity=0.0,
            lcmc=0.0,
            diagnostics={"n": float(n), "k": float(k), "n_full": float(n_full)},
            anomalies=anomalies,
        )

    if k >= (n / 2.0):
        raise ValueError(
            f"k must be < n/2 after subsampling to keep trustworthiness in [0, 1], got k={k}, n={n}"
        )

    denom = float(n * k * (2 * n - 3 * k - 1))
    if denom <= 0.0:
        raise ValueError(
            "invalid denominator for trustworthiness formula (n and k too close)"
        )

    dist_high = pairwise_distances(X_high_eval, metric=metric_high).astype(
        np.float64, copy=False
    )
    dist_low = pairwise_distances(X_low_eval, metric=metric_low).astype(
        np.float64, copy=False
    )

    nn_high = _knn_from_distance_matrix(dist_high, k=k)
    nn_low = _knn_from_distance_matrix(dist_low, k=k)

    inv_rank_high = _inverse_ranks_from_distance_matrix(dist_high)
    inv_rank_low = _inverse_ranks_from_distance_matrix(dist_low)

    trust_sum = 0.0
    cont_sum = 0.0
    overlap_sum = 0.0

    for i in range(n):
        high_row = nn_high[i]
        low_row = nn_low[i]

        high_set = set(int(x) for x in high_row.tolist())
        low_set = set(int(x) for x in low_row.tolist())

        overlap_sum += float(len(high_set & low_set))

        false_low = [int(j) for j in low_row if int(j) not in high_set]
        if false_low:
            trust_sum += _rank_penalty(inv_rank_high[i], false_low, k=k)

        false_high = [int(j) for j in high_row if int(j) not in low_set]
        if false_high:
            cont_sum += _rank_penalty(inv_rank_low[i], false_high, k=k)

    factor = 2.0 / denom

    trustworthiness_explicit = float(1.0 - factor * trust_sum)
    continuity = float(1.0 - factor * cont_sum)

    qnn = float(overlap_sum / float(n * k))
    expected_random = float(k / float(n - 1))
    lcmc = float(qnn - expected_random)

    trustworthiness = trustworthiness_explicit
    reference_ok = 0.0
    reference_used_metric = 0.0

    if reference and metric_low != "euclidean":
        anomalies.append(
            "reference trustworthiness uses euclidean in embedded space; skipping due to metric_low mismatch"
        )
    elif reference:
        try:
            trustworthiness = float(
                _sklearn_trustworthiness(
                    X_high_eval, X_low_eval, k=k, metric=metric_high
                )
            )
            reference_ok = 1.0
            reference_used_metric = 1.0
        except TypeError:
            try:
                trustworthiness = float(
                    _sklearn_trustworthiness(X_high_eval, X_low_eval, k=k, metric=None)
                )
                reference_ok = 1.0
                reference_used_metric = 0.0
                anomalies.append(
                    "sklearn trustworthiness did not accept metric; ran without metric"
                )
            except Exception as e:
                anomalies.append(
                    f"reference trustworthiness failed: {type(e).__name__}"
                )
        except Exception as e:
            anomalies.append(f"reference trustworthiness failed: {type(e).__name__}")

    diagnostics: Diagnostics = {
        "n_full": float(n_full),
        "n": float(n),
        "k": float(k),
        "high_d_dims": float(X_high_eval.shape[1]),
        "low_d_dims": float(X_low_eval.shape[1]),
        "metric_high_is_cosine": float(metric_high == "cosine"),
        "metric_low_is_euclidean": float(metric_low == "euclidean"),
        "qnn": float(qnn),
        "expected_random_overlap": float(expected_random),
        "trustworthiness_explicit": float(trustworthiness_explicit),
        "reference_enabled": float(reference),
        "reference_ok": float(reference_ok),
        "reference_used_metric": float(reference_used_metric),
        "subsample_enabled": float(selected_idx is not None),
        "subsample_seed": float(subsample_seed),
    }

    return Trustworthiness(
        score=trustworthiness,
        continuity=continuity,
        lcmc=lcmc,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def score_shepard(
    X_high: FloatArray,
    X_low: FloatArray,
    *,
    n_pairs: int = 5000,
    seed: int = 42,
) -> Shepard:
    """
    Measure distance-rank preservation via Spearman and Kendall correlation.

    Args
    ----
        X_high  (FloatArray) : high-D array of shape (n, d).
        X_low   (FloatArray) : low-D array of shape (n, 2 or 3).
        n_pairs (int)        : number of random pairs to sample.
                               (Default is 5000).
        seed    (int)        : RNG seed.
                               (Default is 42).

    Returns
    -------
        (Shepard) : rank correlations between sampled pairwise distances.
    """
    if X_high.ndim != 2 or X_low.ndim != 2:
        raise ValueError("X_high and X_low must be 2D arrays")

    n = int(X_high.shape[0])
    if n != int(X_low.shape[0]):
        raise ValueError(
            f"shape mismatch: high-D has {X_high.shape[0]}, low-D has {X_low.shape[0]}"
        )

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
    n_pairs_eff = int(min(max(int(n_pairs), 1), max_pairs))

    rng = np.random.default_rng(int(seed))
    idx_i = rng.integers(0, n, size=n_pairs_eff, endpoint=False).astype(
        np.int64, copy=False
    )
    idx_j = rng.integers(0, n, size=n_pairs_eff, endpoint=False).astype(
        np.int64, copy=False
    )

    same_mask = idx_i == idx_j
    while bool(np.any(same_mask)):
        replacement = rng.integers(
            0, n, size=int(np.sum(same_mask)), endpoint=False
        ).astype(np.int64, copy=False)
        idx_j[same_mask] = replacement
        same_mask = idx_i == idx_j

    Xh = np.asarray(X_high, dtype=np.float64)
    Xl = np.asarray(X_low, dtype=np.float64)

    dist_high = _cosine_distance_pairs(Xh, i_idx=idx_i, j_idx=idx_j, eps=L2_EPS)
    dist_low = _euclidean_distance_pairs(Xl, i_idx=idx_i, j_idx=idx_j)

    spearman_out = spearmanr(dist_high, dist_low)
    kendall_out = kendalltau(dist_high, dist_low)

    rho, _ = cast(tuple[float, float], spearman_out)
    tau, _ = cast(tuple[float, float], kendall_out)

    if np.isnan(rho):
        anomalies.append(
            "spearman correlation is NaN (near-constant distances); treating as 0.0"
        )
        rho = 0.0

    if np.isnan(tau):
        anomalies.append(
            "kendall correlation is NaN (near-constant distances); treating as 0.0"
        )
        tau = 0.0

    diagnostics: Diagnostics = {
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


def _choose_subsample_indices(
    *, n: int, subsample_n: int | None, seed: int
) -> IntArray | None:
    """Choose a deterministic subsample of row indices."""
    if subsample_n is None or subsample_n >= n:
        return None

    rng = np.random.default_rng(int(seed))
    return rng.choice(n, size=int(subsample_n), replace=False).astype(
        np.int64, copy=False
    )


def _knn_from_distance_matrix(dist: FloatArray, *, k: int) -> IntArray:
    """Compute kNN indices from a distance matrix (self excluded by diagonal masking)."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square, got shape {dist.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    n = int(dist.shape[0])
    if n <= k:
        raise ValueError(f"n ({n}) must be > k ({k})")

    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)

    idx = np.argpartition(masked, kth=int(k - 1), axis=1)[:, :k]
    row = np.arange(n)[:, None]
    row_d = masked[row, idx]
    order = np.argsort(row_d, axis=1, kind="quicksort")
    return idx[row, order].astype(np.int64, copy=False)


def _inverse_ranks_from_distance_matrix(dist: FloatArray) -> IntArray:
    """Compute inverse ranks for all rows (self has rank 0 by diagonal forcing)."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square, got shape {dist.shape}")

    masked = dist.copy()
    np.fill_diagonal(masked, -1.0)

    order = np.argsort(masked, axis=1, kind="quicksort")
    inv = np.empty_like(order, dtype=np.int64)
    row = np.arange(order.shape[0])[:, None]
    inv[row, order] = np.arange(order.shape[1])[None, :]
    return inv


def _rank_penalty(inv_rank_row: IntArray, indices: list[int], *, k: int) -> float:
    """Sum (rank - k) for indices using an inverse-rank lookup."""
    penalty = 0.0
    for j in indices:
        penalty += float(int(inv_rank_row[int(j)]) - int(k))
    return penalty


def _cosine_distance_pairs(
    X: FloatArray,
    *,
    i_idx: IntArray,
    j_idx: IntArray,
    eps: float,
) -> FloatArray:
    """Compute cosine distances for index pairs."""
    a = X[i_idx]
    b = X[j_idx]
    dots = np.sum(a * b, axis=1)
    norms = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + float(eps)

    cos = np.clip(dots / norms, -1.0, 1.0)
    return (1.0 - cos).astype(np.float64, copy=False)


def _euclidean_distance_pairs(
    X: FloatArray, *, i_idx: IntArray, j_idx: IntArray
) -> FloatArray:
    """Compute euclidean distances for index pairs."""
    diff = X[i_idx] - X[j_idx]
    return np.linalg.norm(diff, axis=1).astype(np.float64, copy=False)


def _sklearn_trustworthiness(
    X_high: FloatArray,
    X_low: FloatArray,
    *,
    k: int,
    metric: str | None,
) -> float:
    """
    Call sklearn.manifold.trustworthiness with signature adaptation.

    This helper avoids pinning to a specific scikit-learn version.
    """
    from sklearn.manifold import trustworthiness as sklearn_trustworthiness

    sig = inspect.signature(sklearn_trustworthiness)
    kwargs: dict[str, Any] = {}

    if "n_neighbors" in sig.parameters:
        kwargs["n_neighbors"] = int(k)

    if metric is not None and "metric" in sig.parameters:
        kwargs["metric"] = metric

    return float(sklearn_trustworthiness(X_high, X_low, **kwargs))
