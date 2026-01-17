from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import stats

SimilarityStats: TypeAlias = dict[str, float]
SpectrumData: TypeAlias = dict[str, Any]
Diagnostics: TypeAlias = dict[str, float]

FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]

KNN_NEIGHBORS = 15
PAIRWISE_SAMPLE_SIZE = 10_000
PROBE_SAMPLE_SIZE = 200
RANDOM_SEED = 42

ALIGNMENT_ALPHA = 2.0
UNIFORMITY_TEMP = 2.0

L2_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Isotropy:
    """Centered spectral isotropy report."""

    score: float
    singular_values: FloatArray
    effective_rank: float
    entropy: float
    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Hubness:
    """Hubness report from reverse neighbor (k-occurrence) statistics."""

    skewness: float
    hub_rate: float
    hub_occurrence_rate: float
    antihub_rate: float
    hubs_idx: IntArray
    antihubs_idx: IntArray
    gini: float
    k_occurrence: IntArray
    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EmbeddingQuality:
    """Comprehensive embedding quality assessment."""

    validity: SimilarityStats
    pairwise_cosine: SimilarityStats
    dimension_spread: SimilarityStats
    neighbor_gap: SimilarityStats
    degree_stats: SimilarityStats
    effective_rank: dict[str, float]
    information_abundance: dict[str, float]
    spectrum: SpectrumData
    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_isotropy(
    embeddings: FloatArray,
    *,
    center: bool = True,
    l2_normalize: bool = False,
    eps: float = L2_EPS,
) -> Isotropy:
    """
    Measure how uniformly variance is distributed across directions.

    This is a centered spectral-uniformity proxy based on the singular values of the
    centered embedding matrix.

    Args
    ----
        embeddings    (FloatArray) : embedding matrix of shape (n_samples, n_dims).
        center              (bool) : subtract mean before SVD.
                                     (Default is True).
        l2_normalize        (bool) : L2-normalize rows before SVD.
                                     Use this when QA should reflect cosine retrieval space.
                                     (Default is False).
        eps                (float) : numerical stability constant.
                                     (Default is 1e-12).

    Returns
    -------
        (Isotropy) : isotropy ratio in [~1/d, 1], spectral entropy, and effective rank.

    Notes
    -----
        - score uses singular values σ directly (sum σ)^2 / (d * sum σ^2).
        - effective_rank uses energy weights σ^2 via exp(H(p)).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

    X = np.asarray(embeddings, dtype=np.float64)
    n_samples, n_dims = int(X.shape[0]), int(X.shape[1])
    anomalies: list[str] = []

    if n_samples < 2:
        anomalies.append("insufficient samples (n < 2)")
        sv_empty = np.zeros(0, dtype=np.float64)
        return Isotropy(
            score=0.0,
            singular_values=sv_empty,
            effective_rank=0.0,
            entropy=0.0,
            diagnostics={
                "n_samples": float(n_samples),
                "n_dims": float(n_dims),
                "center": float(center),
                "l2_normalize": float(l2_normalize),
            },
            anomalies=anomalies,
        )

    if center:
        X = X - np.mean(X, axis=0)

    if l2_normalize:
        X = _l2_normalize_rows(X, eps=eps)

    sv = np.linalg.svd(X, full_matrices=False, compute_uv=False).astype(
        np.float64, copy=False
    )
    sv_sq = sv * sv
    sv_sum = float(np.sum(sv))
    sv_sq_sum = float(np.sum(sv_sq))

    if sv_sq_sum <= float(eps):
        anomalies.append(
            "degenerate spectrum (sum of squared singular values near zero)"
        )
        return Isotropy(
            score=0.0,
            singular_values=sv,
            effective_rank=0.0,
            entropy=0.0,
            diagnostics={
                "n_samples": float(n_samples),
                "n_dims": float(n_dims),
                "n_singular_values": float(sv.size),
                "center": float(center),
                "l2_normalize": float(l2_normalize),
            },
            anomalies=anomalies,
        )

    score = float((sv_sum * sv_sum) / (float(n_dims) * sv_sq_sum))

    p = sv_sq / sv_sq_sum
    entropy = float(-np.sum(p * np.log(p + float(eps))))
    effective_rank = float(np.exp(entropy))

    min_sv = float(np.min(sv)) if sv.size > 0 else 0.0
    max_sv = float(np.max(sv)) if sv.size > 0 else 0.0
    condition_number = (
        float(max_sv / (min_sv + float(eps))) if max_sv > 0.0 else float("inf")
    )

    diagnostics: Diagnostics = {
        "n_samples": float(n_samples),
        "n_dims": float(n_dims),
        "n_singular_values": float(sv.size),
        "center": float(center),
        "l2_normalize": float(l2_normalize),
        "score_min_theoretical": float(1.0 / float(n_dims)) if n_dims > 0 else 0.0,
        "score_max_theoretical": 1.0,
        "participation_ratio_sigma": float((sv_sum * sv_sum) / sv_sq_sum),
        "min_singular_value": float(min_sv),
        "max_singular_value": float(max_sv),
        "condition_number_truncated": float(condition_number),
    }

    if n_samples < n_dims:
        anomalies.append(
            "underdetermined system (n_samples < n_dims); spectrum is truncated to min(n_samples, n_dims)"
        )

    if n_samples < 100:
        anomalies.append("sample size below recommended minimum (100)")

    return Isotropy(
        score=score,
        singular_values=sv,
        effective_rank=effective_rank,
        entropy=entropy,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def calibrate_isotropy(
    embeddings: FloatArray,
    *,
    method: str = "whitening",
    eps: float = 1e-5,
) -> FloatArray:
    """
    Transform embeddings to improve isotropy via whitening or centering.

    Args
    ----
        embeddings (FloatArray) : embedding matrix of shape (n_samples, n_dims).
        method          (str)   : "whitening" or "centering".
                                  (Default is "whitening").
        eps            (float)  : covariance regularization for whitening.
                                  (Default is 1e-5).

    Returns
    -------
        (FloatArray) : transformed embeddings with same shape.

    """
    X = np.asarray(embeddings, dtype=np.float64)

    if method == "centering":
        return (X - np.mean(X, axis=0)).astype(np.float64, copy=False)

    if method != "whitening":
        raise ValueError(f"unknown method '{method}'; use 'whitening' or 'centering'")

    mean = np.mean(X, axis=0)
    centered = X - mean
    cov = np.cov(centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals + float(eps)
    whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    return (centered @ whitening_matrix).astype(np.float64, copy=False)


def reduce_hubness(
    embeddings: FloatArray,
    *,
    k: int = 10,
    eps: float = L2_EPS,
) -> FloatArray:
    """
    Mitigate hubness in cosine space via local centering.

    Computes cosine kNN neighborhoods on L2-normalized vectors
    then subtracts each point's neighborhood mean direction 
    and re-normalizes.

    Args
    ----
        embeddings (FloatArray) : embedding matrix of shape (n_samples, n_dims).
        k                (int)  : neighborhood size for local centering.
                                  (Default is 10).
        eps             (float) : numerical stability constant.
                                  (Default is 1e-12).

    Returns
    -------
        (FloatArray) : transformed L2-normalized embeddings with same shape.
    """
    from sklearn.neighbors import NearestNeighbors

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    X = np.asarray(embeddings, dtype=np.float64)
    Z = _l2_normalize_rows(X, eps=eps)

    n_samples = int(Z.shape[0])
    if n_samples <= k + 1:
        return Z

    nbrs = NearestNeighbors(n_neighbors=int(k) + 1, metric="cosine").fit(Z)
    _, indices = nbrs.kneighbors(Z)

    nn_idx = indices[:, 1:]
    nn_mean = np.mean(Z[nn_idx], axis=1)

    centered = Z - nn_mean
    norms = np.linalg.norm(centered, axis=1)

    collapsed = norms <= float(eps)
    if bool(np.any(collapsed)):
        centered[collapsed] = Z[collapsed]

    return _l2_normalize_rows(centered, eps=eps)


def score_hubness(
    knn_indices: IntArray,
    *,
    beta: float = 3.0,
    antihub_max_occurrence: int = 0,
    drop_self: bool = True,
) -> Hubness:
    """
    Quantify hubness via the k-occurrence (reverse neighbor count) distribution.

    Args
    ----
        knn_indices (IntArray) : k-NN indices of shape (n_samples, k).
        beta          (float)  : hub threshold multiplier for O_k(x) > beta*k.
                                 (Default is 3.0).
        antihub_max_occurrence (int) : maximum k-occurrence treated as an antihub.
                                      (Default is 0).
        drop_self     (bool)   : drop i from row i if present.
                                 (Default is True).

    Returns
    -------
        (Hubness) : hubness scores, distribution summaries, and index sets.

    """
    if knn_indices.ndim != 2:
        raise ValueError(f"knn_indices must be 2D, got shape {knn_indices.shape}")

    n_samples, k = int(knn_indices.shape[0]), int(knn_indices.shape[1])
    anomalies: list[str] = []

    if n_samples < 1 or k < 1:
        empty = np.zeros(max(n_samples, 0), dtype=np.int64)
        return Hubness(
            skewness=0.0,
            hub_rate=0.0,
            hub_occurrence_rate=0.0,
            antihub_rate=0.0,
            hubs_idx=empty,
            antihubs_idx=empty,
            gini=0.0,
            k_occurrence=empty,
            diagnostics={"n_samples": float(n_samples), "k": float(k)},
            anomalies=["insufficient shape for hubness scoring"],
        )

    idx_min = int(np.min(knn_indices))
    idx_max = int(np.max(knn_indices))
    if idx_min < 0 or idx_max >= n_samples:
        raise ValueError(
            f"knn_indices out of range: min={idx_min}, max={idx_max}, expected within [0, {n_samples - 1}]"
        )

    sorted_rows = np.sort(knn_indices, axis=1)
    has_duplicates = bool(np.any(sorted_rows[:, 1:] == sorted_rows[:, :-1]))
    if has_duplicates:
        anomalies.append("duplicate neighbor indices detected within at least one row")

    row_ids = np.arange(n_samples, dtype=knn_indices.dtype)[:, None]
    self_mask = knn_indices == row_ids
    has_self = bool(np.any(self_mask))

    if drop_self and has_self:
        keep_mask = ~self_mask
        k_by_row = keep_mask.sum(axis=1).astype(np.int64, copy=False)

        if int(np.min(k_by_row)) <= 0:
            anomalies.append("at least one row has no neighbors after dropping self")
        if bool(np.any(k_by_row != k_by_row[0])):
            anomalies.append(
                "effective neighbor count varies across rows after dropping self"
            )

        flattened = knn_indices[keep_mask].astype(np.int64, copy=False)
        k_effective_mean = float(np.mean(k_by_row))
    else:
        flattened = knn_indices.astype(np.int64, copy=False).reshape(-1)
        k_effective_mean = float(k)

    total_neighbor_slots = int(flattened.size)
    if total_neighbor_slots == 0:
        empty_occ = np.zeros(n_samples, dtype=np.int64)
        return Hubness(
            skewness=0.0,
            hub_rate=0.0,
            hub_occurrence_rate=0.0,
            antihub_rate=0.0,
            hubs_idx=np.array([], dtype=np.int64),
            antihubs_idx=np.array([], dtype=np.int64),
            gini=0.0,
            k_occurrence=empty_occ,
            diagnostics={
                "n_samples": float(n_samples),
                "k": float(k),
                "k_effective_mean": float(k_effective_mean),
                "beta": float(beta),
                "drop_self": float(drop_self),
                "has_self_neighbors": float(has_self),
            },
            anomalies=anomalies + ["empty neighbor list after self handling"],
        )

    k_occurrence = np.bincount(flattened, minlength=n_samples).astype(
        np.int64, copy=False
    )
    hub_threshold = float(beta) * float(k_effective_mean)

    hubs_mask = k_occurrence > hub_threshold
    hubs_idx = np.flatnonzero(hubs_mask).astype(np.int64, copy=False)

    antihubs_mask = k_occurrence <= int(antihub_max_occurrence)
    antihubs_idx = np.flatnonzero(antihubs_mask).astype(np.int64, copy=False)

    raw_skew = float(stats.skew(k_occurrence)) if k_occurrence.size > 0 else 0.0
    skewness = 0.0 if np.isnan(raw_skew) else raw_skew
    if np.isnan(raw_skew):
        anomalies.append("skewness returned NaN; treating as 0.0")

    hub_rate = float(np.mean(hubs_mask))
    antihub_rate = float(np.mean(antihubs_mask))
    hub_occurrence_rate = float(
        np.sum(k_occurrence[hubs_mask]) / float(total_neighbor_slots)
    )
    gini = _gini_index(k_occurrence)

    if beta <= 1.0:
        anomalies.append(
            f"beta ({beta}) is <= 1.0; hub threshold may be too permissive"
        )
    if drop_self and has_self and abs(k_effective_mean - float(k - 1)) > 0.25:
        anomalies.append(
            "unexpected effective k after dropping self; check knn_indices construction"
        )

    diagnostics: Diagnostics = {
        "n_samples": float(n_samples),
        "k": float(k),
        "k_effective_mean": float(k_effective_mean),
        "beta": float(beta),
        "hub_threshold": float(hub_threshold),
        "total_neighbor_slots": float(total_neighbor_slots),
        "has_self_neighbors": float(has_self),
        "dropped_self_neighbors": float(bool(drop_self and has_self)),
        "has_duplicate_neighbors": float(has_duplicates),
        "mean_occurrence": float(np.mean(k_occurrence)),
        "std_occurrence": float(np.std(k_occurrence)),
        "min_occurrence": float(np.min(k_occurrence)),
        "max_occurrence": float(np.max(k_occurrence)),
        "q50_occurrence": float(np.percentile(k_occurrence, 50)),
        "q90_occurrence": float(np.percentile(k_occurrence, 90)),
        "q95_occurrence": float(np.percentile(k_occurrence, 95)),
        "q99_occurrence": float(np.percentile(k_occurrence, 99)),
        "max_over_k_effective": float(np.max(k_occurrence) / float(k_effective_mean))
        if k_effective_mean > 0
        else 0.0,
        "hubs_count": float(hubs_idx.size),
        "antihubs_count": float(antihubs_idx.size),
    }

    return Hubness(
        skewness=skewness,
        hub_rate=hub_rate,
        hub_occurrence_rate=hub_occurrence_rate,
        antihub_rate=antihub_rate,
        hubs_idx=hubs_idx,
        antihubs_idx=antihubs_idx,
        gini=gini,
        k_occurrence=k_occurrence,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def score_embedding_quality(
    embeddings: FloatArray,
    *,
    k: int = KNN_NEIGHBORS,
    seed: int = RANDOM_SEED,
    eps: float = L2_EPS,
) -> EmbeddingQuality:
    """
    Validate embedding quality across multiple metrics.

    Args
    ----
        embeddings (FloatArray) : embedding matrix of shape (n_samples, n_dims).
        k                (int)  : num neighbors for cosine kNN diagnostics.
                                  (Default is 15).
        seed             (int)  : random seed for sampling-based metrics.
                                  (Default is 42).
        eps              (float): numerical stability constant for normalization.
                                  (Default is 1e-12).

    Returns
    -------
        (EmbeddingQuality) : metric bundle for QA and debugging.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

    X = np.asarray(embeddings, dtype=np.float64)
    n_samples, n_dims = int(X.shape[0]), int(X.shape[1])
    anomalies: list[str] = []

    if n_samples < 2:
        anomalies.append("insufficient samples (n < 2)")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if k >= n_samples:
        anomalies.append(f"k ({k}) must be < n_samples ({n_samples})")

    Z = _l2_normalize_rows(X, eps=eps)

    validity = _check_validity(X, eps=eps)
    pairwise_cosine = _sample_pairwise_cosine(Z, seed=seed)
    dimension_spread = _measure_dimension_spread(X)
    neighbor_gap = _measure_neighbor_separation(Z, k=k, seed=seed)
    degree_stats = _measure_graph_degrees(Z, k=k)

    erank = _compute_effective_rank(Z, eps=eps)
    ia = _compute_information_abundance(Z, eps=eps)
    spectrum = _compute_singular_spectrum(Z, eps=eps)

    effective_rank = {"erank": float(erank), "max_rank": float(n_dims)}
    information_abundance = {"ia": float(ia), "max_ia": float(n_dims)}
    diagnostics = {
        "n_samples": float(n_samples),
        "n_dims": float(n_dims),
        "k": float(k),
        "eps": float(eps),
    }

    return EmbeddingQuality(
        validity=validity,
        pairwise_cosine=pairwise_cosine,
        dimension_spread=dimension_spread,
        neighbor_gap=neighbor_gap,
        degree_stats=degree_stats,
        effective_rank=effective_rank,
        information_abundance=information_abundance,
        spectrum=spectrum,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def compute_alignment_uniformity(
    embeddings: FloatArray,
    positive_pairs: list[tuple[int, int]],
    *,
    l2_normalize: bool = True,
    alpha: float = ALIGNMENT_ALPHA,
    t: float = UNIFORMITY_TEMP,
    max_pairs: int = 250_000,
    seed: int = RANDOM_SEED,
    eps: float = L2_EPS,
) -> SimilarityStats:
    """
    Compute alignment and uniformity for contrastive learning diagnostics.

    Args
    ----
        embeddings (FloatArray) : embedding matrix (n_samples, n_dims).
        positive_pairs (list[tuple[int, int]]) : index pairs expected to be similar.
        l2_normalize (bool) : if True, compute both metrics on unit-norm embeddings.
                              This matches cosine-space retrieval semantics.
                              (Default is True).
        alpha        (float)    : exponent for alignment distance.
                                  (Default is 2.0).
        t            (float)    : temperature for uniformity.
                                  (Default is 2.0).
        max_pairs    (int)      : cap on random pair count for uniformity.
                                  (Default is 250000).
        seed         (int)      : random seed for uniformity sampling.
                                  (Default is 42).
        eps          (float)    : numerical stability constant for normalization.
                                  (Default is 1e-12).

    Returns
    -------
        (dict[str, float]) : alignment and uniformity values.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    Z = _l2_normalize_rows(X, eps=eps) if l2_normalize else X

    if len(positive_pairs) == 0:
        return {"alignment": 0.0, "uniformity": 0.0}

    n = int(Z.shape[0])
    if n < 1:
        return {"alignment": 0.0, "uniformity": 0.0}

    for idx, (i_raw, j_raw) in enumerate(positive_pairs):
        i = int(i_raw)
        j = int(j_raw)
        if i < 0 or i >= n or j < 0 or j >= n:
            raise ValueError(
                f"positive_pairs[{idx}] out of bounds: ({i}, {j}) not in [0, {n - 1}]"
            )

    alignment_terms: list[float] = []
    for i, j in positive_pairs:
        dist = float(np.linalg.norm(Z[int(i)] - Z[int(j)]))
        alignment_terms.append(dist ** float(alpha))

    alignment = float(np.mean(alignment_terms))

    if n < 2:
        return {"alignment": alignment, "uniformity": 0.0}

    max_n = min(n, 1000)
    rng = np.random.default_rng(int(seed))

    pair_count = min(int(max_pairs), (max_n * (max_n - 1)) // 2)
    idx_i = rng.integers(0, max_n, size=pair_count, endpoint=False)
    idx_j = rng.integers(0, max_n, size=pair_count, endpoint=False)

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    diffs = Z[idx_i] - Z[idx_j]
    sq_dists = np.sum(diffs * diffs, axis=1)

    if sq_dists.size == 0:
        uniformity = 0.0
    else:
        uniformity = float(np.log(np.mean(np.exp(-float(t) * sq_dists))))

    return {"alignment": alignment, "uniformity": uniformity}


def compare_neighborhoods(
    embeddings_before: FloatArray,
    embeddings_after: FloatArray,
    *,
    k: int = KNN_NEIGHBORS,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    eps: float = L2_EPS,
) -> SimilarityStats:
    """
    Measure Jaccard overlap of cosine kNN neighborhoods before vs after transformation.

    Args
    ----
        embeddings_before (FloatArray) : embedding matrix before transformation.
        embeddings_after  (FloatArray) : embedding matrix after transformation.
        k           (int) : num neighbors for kNN.
                            (Default is 15).
        n_probe     (int) : num probe points.
                            (Default is 200).
        seed        (int) : random seed.
                            (Default is 42).
        eps        (float): numerical stability constant for normalization.
                            (Default is 1e-12).

    Returns
    -------
        (dict[str, float]) : overlap summary statistics.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    X0 = _l2_normalize_rows(np.asarray(embeddings_before, dtype=np.float64), eps=eps)
    X1 = _l2_normalize_rows(np.asarray(embeddings_after, dtype=np.float64), eps=eps)

    n0 = int(X0.shape[0])
    n1 = int(X1.shape[0])
    if n0 != n1:
        raise ValueError(f"shape mismatch: before has {n0} rows, after has {n1} rows")

    n = n0
    if n <= k + 1:
        return {
            "mean_overlap": 0.0,
            "median_overlap": 0.0,
            "min_overlap": 0.0,
            "max_overlap": 0.0,
        }

    rng = np.random.default_rng(int(seed))
    probes = rng.choice(n, size=min(int(n_probe), n), replace=False)

    overlaps: list[float] = []
    for i in probes:
        sims_before = X0 @ X0[int(i)]
        sims_after = X1 @ X1[int(i)]

        sims_before[int(i)] = -np.inf
        sims_after[int(i)] = -np.inf

        nn_before = set(
            np.argpartition(-sims_before, kth=int(k - 1))[: int(k)].tolist()
        )
        nn_after = set(np.argpartition(-sims_after, kth=int(k - 1))[: int(k)].tolist())

        denom = float(len(nn_before | nn_after))
        overlaps.append(
            float(len(nn_before & nn_after)) / denom if denom > 0.0 else 0.0
        )

    arr = np.asarray(overlaps, dtype=np.float64)
    return {
        "mean_overlap": float(np.mean(arr)),
        "median_overlap": float(np.median(arr)),
        "min_overlap": float(np.min(arr)),
        "max_overlap": float(np.max(arr)),
    }


def _l2_normalize_rows(X: FloatArray, *, eps: float = L2_EPS) -> FloatArray:
    """L2-normalize rows with epsilon for numerical stability."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / (norms + float(eps))).astype(np.float64, copy=False)


def _check_validity(X: FloatArray, *, eps: float = L2_EPS) -> SimilarityStats:
    """Check for NaN, inf, near-zero norms, and duplicate rows."""
    nan_count = int(np.isnan(X).any(axis=1).sum())
    inf_count = int(np.isinf(X).any(axis=1).sum())

    norms = np.linalg.norm(X, axis=1)
    zero_count = int(np.sum(norms <= float(eps)))

    unique_rows = np.unique(X, axis=0)
    duplicate_count = int(X.shape[0] - unique_rows.shape[0])

    return {
        "total": float(X.shape[0]),
        "nan_count": float(nan_count),
        "inf_count": float(inf_count),
        "zero_count": float(zero_count),
        "duplicate_count": float(duplicate_count),
    }


def _sample_pairwise_cosine(
    Z: FloatArray,
    *,
    n_samples: int = PAIRWISE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """Sample random pairs and compute cosine similarity stats."""
    rng = np.random.default_rng(int(seed))
    n = int(Z.shape[0])

    if n < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
        }

    n_actual = int(min(int(n_samples), (n * (n - 1)) // 2))
    sims = np.zeros(n_actual, dtype=np.float64)

    for idx in range(n_actual):
        i, j = rng.choice(n, size=2, replace=False)
        sims[idx] = float(Z[int(i)] @ Z[int(j)])

    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "q1": float(np.percentile(sims, 25)),
        "median": float(np.median(sims)),
        "q3": float(np.percentile(sims, 75)),
        "max": float(np.max(sims)),
    }


def _measure_dimension_spread(X: FloatArray) -> SimilarityStats:
    """Measure per-dimension standard deviation spread."""
    d = int(X.shape[1])
    stds = np.std(X, axis=0).astype(np.float64, copy=False)
    target = float(1.0 / np.sqrt(float(d))) if d > 0 else 0.0

    return {
        "dim": float(d),
        "target_isotropic": float(target),
        "mean_std": float(np.mean(stds)) if stds.size > 0 else 0.0,
        "min_std": float(np.min(stds)) if stds.size > 0 else 0.0,
        "max_std": float(np.max(stds)) if stds.size > 0 else 0.0,
    }


def _measure_neighbor_separation(
    Z: FloatArray,
    *,
    k: int,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """Measure gap between mean cosine similarity to kNN vs random points."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    rng = np.random.default_rng(int(seed))
    n = int(Z.shape[0])

    if n <= int(k) + 1:
        return {"mean_nn_sim": 0.0, "mean_rand_sim": 0.0, "gap": 0.0}

    probes = rng.choice(n, size=min(int(n_probe), n), replace=False)

    nn_sims: list[float] = []
    rand_sims: list[float] = []

    for i in probes:
        sims = Z @ Z[int(i)]
        sims[int(i)] = -np.inf

        nn_idx = np.argpartition(-sims, kth=int(k - 1))[: int(k)]
        nn_sims.append(float(np.mean(sims[nn_idx])))

        j = int(rng.integers(0, n))
        while j == int(i):
            j = int(rng.integers(0, n))
        rand_sims.append(float(sims[j]))

    mean_nn = float(np.mean(nn_sims))
    mean_rand = float(np.mean(rand_sims))
    return {
        "mean_nn_sim": mean_nn,
        "mean_rand_sim": mean_rand,
        "gap": mean_nn - mean_rand,
    }


def _measure_graph_degrees(Z: FloatArray, *, k: int) -> SimilarityStats:
    """Compute undirected cosine-kNN graph degree distribution (O(n²) memory)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    n = int(Z.shape[0])

    if n <= int(k) + 1:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
            "isolated_count": 0.0,
        }

    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)

    A = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        idx = np.argpartition(-S[i], kth=int(k - 1))[: int(k)]
        A[i, idx] = True

    A_sym = np.logical_or(A, A.T)
    deg = np.sum(A_sym, axis=1).astype(np.float64, copy=False)

    return {
        "mean": float(np.mean(deg)),
        "std": float(np.std(deg)),
        "min": float(np.min(deg)),
        "q1": float(np.percentile(deg, 25)),
        "median": float(np.median(deg)),
        "q3": float(np.percentile(deg, 75)),
        "max": float(np.max(deg)),
        "isolated_count": float(np.sum(deg == 0.0)),
    }


def _compute_effective_rank(Z: FloatArray, *, eps: float = L2_EPS) -> float:
    """Compute entropy-based effective rank from singular-value energy."""
    sv = np.linalg.svd(Z, full_matrices=False, compute_uv=False).astype(
        np.float64, copy=False
    )
    energy = sv * sv
    total = float(np.sum(energy))

    if total <= float(eps):
        return 0.0

    p = energy / total
    entropy = float(-np.sum(p * np.log(p + float(eps))))
    return float(np.exp(entropy))


def _compute_information_abundance(Z: FloatArray, *, eps: float = L2_EPS) -> float:
    """Compute information abundance as sum(sv)/max(sv)."""
    sv = np.linalg.svd(Z, full_matrices=False, compute_uv=False).astype(
        np.float64, copy=False
    )
    max_sv = float(np.max(sv)) if sv.size > 0 else 0.0

    if max_sv <= float(eps):
        return 0.0

    return float(np.sum(sv) / max_sv)


def _compute_singular_spectrum(Z: FloatArray, *, eps: float = L2_EPS) -> SpectrumData:
    """Compute singular spectrum of the covariance of normalized embeddings."""
    n_samples = int(Z.shape[0])
    n_dims = int(Z.shape[1]) if Z.ndim == 2 else 0

    if n_samples < 2 or n_dims == 0:
        sv_empty = np.zeros(0, dtype=np.float64)
        return {
            "singular_values": sv_empty,
            "singular_values_log": sv_empty,
            "singular_values_normalized": sv_empty,
            "n_dims": n_dims,
            "n_nonzero": 0,
        }

    cov = np.cov(Z.T)
    sv = np.linalg.svd(cov, compute_uv=False).astype(np.float64, copy=False)

    sv_max = float(np.max(sv)) if sv.size > 0 else 0.0
    sv_norm = (sv / sv_max) if sv_max > float(eps) else np.zeros_like(sv)

    return {
        "singular_values": sv,
        "singular_values_log": np.log(sv + float(eps)),
        "singular_values_normalized": sv_norm,
        "n_dims": int(sv.size),
        "n_nonzero": int(np.sum(sv > float(1e-9))),
    }


def _gini_index(values: IntArray) -> float:
    """Compute Gini coefficient for a non-negative 1D count vector."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)

    if x.size == 0:
        return 0.0

    if float(np.min(x)) < 0.0:
        raise ValueError("gini input must be non-negative")

    total = float(np.sum(x))
    if total <= 0.0:
        return 0.0

    x_sorted = np.sort(x, kind="quicksort")
    n = float(x_sorted.size)
    index = np.arange(1.0, n + 1.0, dtype=np.float64)

    return float((2.0 * np.sum(index * x_sorted) / (n * total)) - ((n + 1.0) / n))
