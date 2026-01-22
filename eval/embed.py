from dataclasses import dataclass

import numpy as np

from dtypes import FloatArray

L2_EPS = 1e-12
PAIRWISE_SAMPLE_SIZE = 10_000
RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class EmbeddingSanity:
    """
    One-time embedding validation results.

    Args
    ----
        n_samples              (int) : number of embedding vectors.
        n_dims                 (int) : embedding dimensionality.
        has_nan               (bool) : True if any NaN values present.
        has_inf               (bool) : True if any infinite values present.
        has_zero_norm         (bool) : True if any zero-norm vectors present.
        mean_pairwise_cosine (float) : mean cosine similarity of sampled pairs.
        std_pairwise_cosine  (float) : std of cosine similarity of sampled pairs.
    """

    n_samples: int
    n_dims: int
    has_nan: bool
    has_inf: bool
    has_zero_norm: bool
    mean_pairwise_cosine: float
    std_pairwise_cosine: float


def check_embedding_sanity(
    embeddings: FloatArray,
    *,
    n_pairs: int = PAIRWISE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    eps: float = L2_EPS,
) -> EmbeddingSanity:
    """
    Validate embedding matrix for common pathologies.

    Args
    ----
        embeddings (FloatArray) : embedding matrix of shape (n_samples, n_dims).
        n_pairs          (int)  : number of random pairs to sample for cosine stats.
                                  (Default is 10000).
        seed             (int)  : random seed for sampling.
                                  (Default is 42).
        eps            (float)  : threshold for zero-norm detection.
                                  (Default is 1e-12).

    Returns
    -------
        (EmbeddingSanity) : validation results including cosine statistics.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

    X = np.asarray(embeddings, dtype=np.float64)
    n_samples, n_dims = X.shape

    if n_samples == 0:
        return EmbeddingSanity(
            n_samples=0,
            n_dims=n_dims,
            has_nan=False,
            has_inf=False,
            has_zero_norm=False,
            mean_pairwise_cosine=0.0,
            std_pairwise_cosine=0.0,
        )

    # check for NaN and inf
    has_nan = bool(np.any(np.isnan(X)))
    has_inf = bool(np.any(np.isinf(X)))

    # check for zero-norm vectors
    norms = np.linalg.norm(X, axis=1)
    has_zero_norm = bool(np.any(norms <= eps))

    # compute pairwise cosine statistics
    mean_cos, std_cos = _sample_pairwise_cosine(X, n_pairs=n_pairs, seed=seed, eps=eps)

    return EmbeddingSanity(
        n_samples=n_samples,
        n_dims=n_dims,
        has_nan=has_nan,
        has_inf=has_inf,
        has_zero_norm=has_zero_norm,
        mean_pairwise_cosine=mean_cos,
        std_pairwise_cosine=std_cos,
    )


def _sample_pairwise_cosine(
    X: FloatArray,
    *,
    n_pairs: int,
    seed: int,
    eps: float,
) -> tuple[float, float]:
    """Sample random pairs and compute cosine similarity statistics."""
    n = len(X)
    if n < 2:
        return 0.0, 0.0

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Z = X / (norms + eps)

    # sample pairs
    max_pairs = (n * (n - 1)) // 2
    n_actual = min(n_pairs, max_pairs)

    rng = np.random.default_rng(seed)
    sims = np.zeros(n_actual, dtype=np.float64)

    for idx in range(n_actual):
        i, j = rng.choice(n, size=2, replace=False)
        sims[idx] = float(Z[i] @ Z[j])

    return float(np.mean(sims)), float(np.std(sims))
