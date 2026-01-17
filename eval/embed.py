from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from scipy import stats

SimilarityStats: TypeAlias = dict[str, float]
SpectrumData: TypeAlias = dict[str, Any]

KNN_NEIGHBORS = 15
PAIRWISE_SAMPLE_SIZE = 10000
PROBE_SAMPLE_SIZE = 200
RANDOM_SEED = 42
ALIGNMENT_ALPHA = 2.0
UNIFORMITY_TEMP = 2.0


@dataclass(frozen=True, slots=True)
class Isotropy:
    """Isotropy measurement results."""

    score: float
    singular_values: np.ndarray
    effective_rank: float
    entropy: float
    diagnostics: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Hubness:
    """Hubness measurement results."""

    skewness: float
    hub_rate: float
    antihub_rate: float
    k_occurrence: np.ndarray
    diagnostics: dict = field(default_factory=dict)
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
    diagnostics: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_isotropy(embeddings: np.ndarray) -> Isotropy:
    """
    Measure uniformity of variance distribution across embedding dimensions.

    IsoScore quantifies how evenly embedding space is utilized. Low scores
    indicate concentration in narrow subspace (anisotropy), which degrades
    cosine similarity discrimination and neighbor quality.

    Args:
        embeddings (np.ndarray) : embedding matrix of shape (n_samples, n_dims).

    Returns:
        (Isotropy) : continuous metrics with diagnostics and anomalies.

    Notes:
        - score near 1/d = severe collapse; near 1 = good isotropy
        - requires n_samples >= 100 for reliable estimates

    Example:
        >>> report = score_isotropy(embeddings)
        >>> print(f"IsoScore: {report.score:.3f}")
    """
    n_samples, n_dims = embeddings.shape
    anomalies = []

    if n_samples < 100:
        anomalies.append(
            f"sample size ({n_samples}) below recommended minimum (100)"
        )
    if n_samples < n_dims:
        anomalies.append(
            f"underdetermined system: {n_samples} samples < {n_dims} dims"
        )

    _, singular_values, _ = np.linalg.svd(embeddings, full_matrices=False)

    sigma_sum = np.sum(singular_values)
    sigma_sq_sum = np.sum(singular_values**2)

    if sigma_sq_sum == 0:
        anomalies.append("zero singular values; degenerate embedding matrix")
        iso_score = 0.0
        effective_rank = 0.0
        entropy = 0.0
    else:
        iso_score = float((sigma_sum**2) / (n_dims * sigma_sq_sum))
        effective_rank = float((sigma_sum**2) / sigma_sq_sum)

        sigma_normalized = singular_values / sigma_sum
        sigma_normalized = sigma_normalized + 1e-12
        entropy = float(-np.sum(sigma_normalized * np.log(sigma_normalized)))

    diagnostics = {
        "n_samples": n_samples,
        "n_dims": n_dims,
        "min_singular_value": float(singular_values[-1]),
        "max_singular_value": float(singular_values[0]),
        "condition_number": (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0
            else np.inf
        ),
    }

    return Isotropy(
        score=iso_score,
        singular_values=singular_values,
        effective_rank=effective_rank,
        entropy=entropy,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def score_hubness(knn_indices: np.ndarray) -> Hubness:
    """
    Quantify hubness phenomenon via k-occurrence distribution analysis.

    Hubness emerges in high dimensions when certain points (hubs) appear
    disproportionately in k-NN lists. This violates symmetric neighbor
    assumptions and degrades graph-based algorithms.

    Args:
        knn_indices (np.ndarray) : k-NN indices of shape (n_samples, k).

    Returns:
        (Hubness) : continuous metrics with diagnostics and anomalies.

    Notes:
        - skewness > 1.5 indicates severe hubness requiring mitigation
        - hub rate = fraction with occurrence > 3k (3× expected)

    Example:
        >>> report = score_hubness(knn_indices)
        >>> print(f"Skewness: {report.skewness:.2f}")
    """
    n_samples, k = knn_indices.shape
    anomalies = []

    k_occurrence = np.zeros(n_samples, dtype=int)
    for neighbors in knn_indices:
        for neighbor_idx in neighbors:
            k_occurrence[neighbor_idx] += 1

    expected_occurrence = k
    skewness = float(stats.skew(k_occurrence)) if len(k_occurrence) > 0 else 0.0

    hub_threshold = 3 * expected_occurrence
    hub_rate = float(np.sum(k_occurrence > hub_threshold) / n_samples)
    antihub_rate = float(np.sum(k_occurrence == 0) / n_samples)

    if skewness > 2.0:
        anomalies.append(
            f"extreme skewness ({skewness:.2f}); severe hubness detected"
        )
    if hub_rate > 0.10:
        anomalies.append(
            f"high hub rate ({hub_rate:.1%}); distance concentration"
        )
    if antihub_rate > 0.10:
        anomalies.append(
            f"high anti-hub rate ({antihub_rate:.1%}); many isolated points"
        )

    diagnostics = {
        "n_samples": n_samples,
        "k": k,
        "expected_occurrence": expected_occurrence,
        "mean_occurrence": float(np.mean(k_occurrence)),
        "std_occurrence": float(np.std(k_occurrence)),
        "min_occurrence": int(np.min(k_occurrence)),
        "max_occurrence": int(np.max(k_occurrence)),
        "hub_threshold": hub_threshold,
    }

    return Hubness(
        skewness=skewness,
        hub_rate=hub_rate,
        antihub_rate=antihub_rate,
        k_occurrence=k_occurrence,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def calibrate_isotropy(
    embeddings: np.ndarray, method: str = "whitening"
) -> np.ndarray:
    """
    Transform embeddings to improve isotropy via whitening or centering.

    Args:
        embeddings (np.ndarray) : embedding matrix of shape (n_samples, n_dims).
        method          (str)   : "whitening" or "centering".
                                  (Default is "whitening").

    Returns:
        (np.ndarray) : transformed embeddings with same shape.

    Notes:
        - whitening: ZCA via SVD; makes covariance identity
        - centering: mean-center per dimension; simpler but less effective

    Example:
        >>> calibrated = calibrate_isotropy(embeddings)
    """
    if method == "centering":
        return embeddings - np.mean(embeddings, axis=0)
    elif method == "whitening":
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        cov = np.cov(centered, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals + 1e-5
        whitening_matrix = (
            eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        )

        return centered @ whitening_matrix
    else:
        raise ValueError(
            f"unknown method '{method}'; use 'whitening' or 'centering'"
        )


def reduce_hubness(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Transform embeddings to mitigate hubness via local scaling.

    Args:
        embeddings (np.ndarray) : embedding matrix of shape (n_samples, n_dims).
        k               (int)   : neighborhood size for local scaling.
                                  (Default is 10).

    Returns:
        (np.ndarray) : transformed embeddings with same shape.

    Notes:
        - normalizes distances by local density estimate
        - requires k-NN computation; O(n²) for brute force

    Example:
        >>> reduced = reduce_hubness(embeddings, k=10)
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    local_density = np.mean(distances[:, 1:], axis=1)
    density_weights = 1.0 / (local_density + 1e-8)
    density_weights = density_weights / np.mean(density_weights)

    return embeddings * density_weights[:, np.newaxis]


def score_embedding_quality(
    embeddings: np.ndarray,
    *,
    k: int = KNN_NEIGHBORS,
) -> EmbeddingQuality:
    """
    Validate embedding quality across multiple metrics.

    Checks: validity (NaN/inf/zeros), pairwise sim, dim spread,
    neighbor separation, graph connectivity, effective rank,
    information abundance (IA), singular spectrum.

    Args:
        embeddings (np.ndarray) : embedding matrix of shape (n_samples, n_dims).
        k               (int)   : num neighbors for kNN graph.
                                  (Default is 15).

    Returns:
        (EmbeddingQuality) : nested dict w/ metric categories + values.

    Example:
        >>> quality = score_embedding_quality(embeddings, k=15)
        >>> print(f"Valid: {quality.validity['nan_count']}")
    """
    n_samples, n_dims = embeddings.shape
    anomalies: list[str] = []

    if n_samples < 2:
        anomalies.append("insufficient samples (n < 2)")
    if k >= n_samples:
        anomalies.append(f"k ({k}) must be < n_samples ({n_samples})")

    E = embeddings.astype(np.float32)

    validity = _check_validity(E)
    pairwise_cosine = _sample_pairwise_cosine(E)
    dimension_spread = _measure_dimension_spread(E)
    neighbor_gap = _measure_neighbor_separation(E, k=k)
    degree_stats = _measure_graph_degrees(E, k=k)

    erank = _compute_effective_rank(E)
    ia = _compute_information_abundance(E)
    spectrum = _compute_singular_spectrum(E)

    effective_rank = {
        "erank": erank,
        "max_rank": float(n_dims),
    }
    information_abundance = {
        "ia": ia,
        "max_ia": float(n_dims),
    }

    diagnostics = {
        "n_samples": n_samples,
        "n_dims": n_dims,
        "k": k,
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


def _check_validity(E: np.ndarray) -> SimilarityStats:
    """Check for NaN, inf, zero-norm, duplicate vectors."""
    nan_count = int(np.isnan(E).any(axis=1).sum())
    inf_count = int(np.isinf(E).any(axis=1).sum())

    norms = np.linalg.norm(E, axis=1)
    zero_count = int(np.sum(norms < 1e-9))

    unique_rows = np.unique(E, axis=0)
    duplicate_count = len(E) - len(unique_rows)

    return {
        "total": float(len(E)),
        "nan_count": float(nan_count),
        "inf_count": float(inf_count),
        "zero_count": float(zero_count),
        "duplicate_count": float(duplicate_count),
    }


def _sample_pairwise_cosine(
    E: np.ndarray,
    n_samples: int = PAIRWISE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """Sample random pairs and compute cosine sim stats."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]

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

    n_actual = min(n_samples, (n * (n - 1)) // 2)
    sims = np.zeros(n_actual, dtype=np.float32)

    for idx in range(n_actual):
        i, j = rng.choice(n, size=2, replace=False)
        sims[idx] = float(E[i] @ E[j])

    return {
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "q1": float(np.percentile(sims, 25)),
        "median": float(np.median(sims)),
        "q3": float(np.percentile(sims, 75)),
        "max": float(sims.max()),
    }


def _measure_dimension_spread(E: np.ndarray) -> SimilarityStats:
    """Check if embedding dims have similar variance (isotropy)."""
    d = E.shape[1]
    target = 1.0 / np.sqrt(float(d))
    stds = E.std(axis=0).astype(np.float32)

    return {
        "dim": float(d),
        "target_isotropic": float(target),
        "mean_std": float(stds.mean()),
        "min_std": float(stds.min()),
        "max_std": float(stds.max()),
    }


def _measure_neighbor_separation(
    E: np.ndarray,
    k: int = KNN_NEIGHBORS,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """Measure gap between kNN sim and random-pair sim."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]

    if n <= k + 1:
        return {
            "mean_nn_sim": 0.0,
            "mean_rand_sim": 0.0,
            "gap": 0.0,
        }

    probes = rng.choice(n, size=min(n_probe, n), replace=False)
    nn_sims = []
    rand_sims = []

    for i in probes:
        sims = E @ E[int(i)]
        sims[int(i)] = -np.inf

        nn_idx = np.argpartition(-sims, kth=k - 1)[:k]
        nn_sims.append(float(np.mean(sims[nn_idx])))

        j = int(rng.choice(n))
        while j == int(i):
            j = int(rng.choice(n))
        rand_sims.append(float(sims[j]))

    mean_nn = float(np.mean(nn_sims))
    mean_rand = float(np.mean(rand_sims))

    return {
        "mean_nn_sim": mean_nn,
        "mean_rand_sim": mean_rand,
        "gap": mean_nn - mean_rand,
    }


def _measure_graph_degrees(
    E: np.ndarray,
    k: int = KNN_NEIGHBORS,
) -> SimilarityStats:
    """Compute kNN graph degree distribution (O(n²) memory)."""
    n = E.shape[0]

    if n <= k + 1:
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

    S = E @ E.T
    np.fill_diagonal(S, -np.inf)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argpartition(-S[i], kth=k - 1)[:k]
        A[i, idx] = 1.0

    A_sym = np.maximum(A, A.T)
    deg = (A_sym != 0).sum(axis=1).astype(np.float32)

    return {
        "mean": float(deg.mean()),
        "std": float(deg.std()),
        "min": float(deg.min()),
        "q1": float(np.percentile(deg, 25)),
        "median": float(np.median(deg)),
        "q3": float(np.percentile(deg, 75)),
        "max": float(deg.max()),
        "isolated_count": float(np.sum(deg == 0)),
    }


def _compute_effective_rank(E: np.ndarray) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values."""
    singular_values = np.linalg.svd(E, full_matrices=False, compute_uv=False)
    singular_values_sq = singular_values**2
    total = singular_values_sq.sum()

    if total < 1e-12:
        return 1.0

    p = singular_values_sq / total
    entropy = -p @ np.nan_to_num(np.log(p), neginf=0.0)
    erank = float(np.exp(entropy))

    return erank


def _compute_information_abundance(E: np.ndarray) -> float:
    """Compute information abundance (IA): sum / max."""
    singular_values = np.linalg.svd(E, full_matrices=False, compute_uv=False)

    if singular_values.max() < 1e-12:
        return 1.0

    ia = float(singular_values.sum() / singular_values.max())
    return ia


def _compute_singular_spectrum(E: np.ndarray) -> SpectrumData:
    """Compute singular value spectrum of normalized embedding covariance."""
    z = E / np.linalg.norm(E, axis=1, keepdims=True)
    C = np.cov(z.T)

    singular_values = np.linalg.svd(C, compute_uv=False)
    singular_values_log = np.log(singular_values + 1e-12)

    sv_max = singular_values.max()
    if sv_max < 1e-12:
        singular_values_normalized = np.zeros_like(singular_values)
    else:
        singular_values_normalized = singular_values / sv_max

    n_nonzero = int(np.sum(singular_values > 1e-9))

    return {
        "singular_values": singular_values,
        "singular_values_log": singular_values_log,
        "singular_values_normalized": singular_values_normalized,
        "n_dims": len(singular_values),
        "n_nonzero": n_nonzero,
    }


def compute_alignment_uniformity(
    E: np.ndarray,
    positive_pairs: list[tuple[int, int]],
    alpha: float = ALIGNMENT_ALPHA,
    t: float = UNIFORMITY_TEMP,
) -> SimilarityStats:
    """
    Compute alignment + uniformity for contrastive learning.

    Args:
        E              (np.ndarray) : embedding matrix (n_samples, n_dims).
        positive_pairs (list[tuple]) : list of (i, j) index pairs that should be similar.
        alpha              (float)   : exponent for alignment loss.
                                       (Default is 2.0).
        t                  (float)   : temperature param for uniformity.
                                       (Default is 2.0).

    Returns:
        (dict[str, float]) : alignment, uniformity scores.
    """
    if len(positive_pairs) == 0:
        return {"alignment": 0.0, "uniformity": 0.0}

    alignment_dists = []
    for i, j in positive_pairs:
        dist = np.linalg.norm(E[i] - E[j])
        alignment_dists.append(float(dist**alpha))
    alignment = float(np.mean(alignment_dists))

    n = E.shape[0]
    if n < 2:
        return {"alignment": alignment, "uniformity": 0.0}

    pairwise_sq_dists = []
    for i in range(min(n, 1000)):
        for j in range(i + 1, min(n, 1000)):
            sq_dist = float(np.sum((E[i] - E[j]) ** 2))
            pairwise_sq_dists.append(sq_dist)

    if len(pairwise_sq_dists) == 0:
        uniformity = 0.0
    else:
        exp_terms = np.exp(-t * np.array(pairwise_sq_dists))
        uniformity = float(np.log(np.mean(exp_terms)))

    return {"alignment": alignment, "uniformity": uniformity}


def compare_neighborhoods(
    E_before: np.ndarray,
    E_after: np.ndarray,
    k: int = KNN_NEIGHBORS,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """
    Measure Jaccard overlap of kNN neighborhoods before vs after transformation.

    Args:
        E_before (np.ndarray) : embedding matrix before transformation.
        E_after  (np.ndarray) : embedding matrix after transformation.
        k             (int)   : num neighbors for kNN.
                                (Default is 15).
        n_probe       (int)   : num probe points.
                                (Default is 200).
        seed          (int)   : random seed.
                                (Default is 42).

    Returns:
        (dict[str, float]) : mean/median/min/max overlap.
    """
    rng = np.random.default_rng(seed)
    n = E_before.shape[0]

    if n <= k + 1:
        return {"mean_overlap": 0.0, "median_overlap": 0.0}

    probes = rng.choice(n, size=min(n_probe, n), replace=False)
    overlaps = []

    for i in probes:
        sims_before = E_before @ E_before[int(i)]
        sims_before[int(i)] = -np.inf
        nn_before = set(np.argpartition(-sims_before, kth=k - 1)[:k])

        sims_after = E_after @ E_after[int(i)]
        sims_after[int(i)] = -np.inf
        nn_after = set(np.argpartition(-sims_after, kth=k - 1)[:k])

        jaccard = len(nn_before & nn_after) / float(len(nn_before | nn_after))
        overlaps.append(jaccard)

    overlaps_arr = np.array(overlaps, dtype=np.float32)

    return {
        "mean_overlap": float(overlaps_arr.mean()),
        "median_overlap": float(np.median(overlaps_arr)),
        "min_overlap": float(overlaps_arr.min()),
        "max_overlap": float(overlaps_arr.max()),
    }
