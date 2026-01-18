from dataclasses import dataclass

import numpy as np

from eval.manifold import (
    FloatArray,
    Shepard,
    Trustworthiness,
    score_shepard,
    score_trustworthiness,
)


@dataclass(frozen=True, slots=True)
class UMAPConfig:
    """
    Configure UMAP layout computation.

    Attributes
    ----------
    n_neighbors : int
        UMAP neighborhood size controlling local vs global structure.
    min_dist : float
        Minimum distance between points in the embedded space (cluster compactness).
    seed : int | None
        Random seed for deterministic layouts. If None, UMAP is nondeterministic.
    metric : str
        Distance metric in high-D. SoundSpace defaults to cosine for CLAP-style embeddings.
    compute_quality : bool
        If True, compute trustworthiness/continuity/LCMC and Shepard correlations.
    quality_k : int | None
        Neighborhood size for trustworthiness/continuity evaluation. If None, uses
        min(n_neighbors, 15).
    quality_subsample_n : int | None
        If set, evaluate quality metrics on a deterministic subsample to bound O(n^2).
    quality_seed : int
        Seed for diagnostic subsampling and pair sampling.
    quality_reference : bool
        If True, attempt an sklearn trustworthiness cross-check when available.
    shepard_n_pairs : int
        Number of random pairs for Shepard rank correlation.
    """

    n_neighbors: int = 15
    min_dist: float = 0.10
    seed: int | None = 42
    metric: str = "cosine"

    compute_quality: bool = False
    quality_k: int | None = None
    quality_subsample_n: int | None = None
    quality_seed: int = 42
    quality_reference: bool = False
    shepard_n_pairs: int = 5000


@dataclass(frozen=True, slots=True)
class UMAPDiagnostics:
    """Diagnostics computed from inputs and the resulting 2D layout."""

    n_samples: int
    n_dims: int
    n_neighbors: int
    min_dist: float
    metric: str

    coord_min_x: float
    coord_max_x: float
    coord_min_y: float
    coord_max_y: float
    coord_mean_norm: float
    coord_std_norm: float


@dataclass(frozen=True, slots=True)
class UMAPResult:
    """UMAP outputs for downstream analysis and UI rendering."""

    coords: FloatArray
    diagnostics: UMAPDiagnostics
    anomalies: tuple[str, ...]

    trustworthiness: Trustworthiness | None
    shepard: Shepard | None


def umap_layout(embeddings: FloatArray, config: UMAPConfig | None = None) -> UMAPResult:
    """
    Compute a 2D UMAP layout from an embedding matrix.

    Args
    ----
    embeddings (FloatArray) : Embedding matrix of shape (n_samples, n_dims).
    config (UMAPConfig | None) : UMAP configuration. Default is None.

    Returns
    -------
    UMAPResult
        2D coordinates with validation diagnostics and optional quality metrics.

    Raises
    ------
    ValueError
        If embeddings are invalid or configuration parameters are out of range.
    ImportError
        If the optional dependency `umap-learn` is not installed.
    """
    cfg = config or UMAPConfig()
    X, anomalies = _validate_embeddings_matrix(embeddings=embeddings, cfg=cfg)

    coords = _fit_umap_2d(embeddings=X, cfg=cfg)
    diagnostics = _compute_layout_diagnostics(embeddings=X, coords=coords, cfg=cfg)

    trust: Trustworthiness | None = None
    shep: Shepard | None = None
    if cfg.compute_quality:
        trust, shep, quality_anomalies = _compute_quality_metrics(
            embeddings=X, coords=coords, cfg=cfg
        )
        anomalies = tuple([*anomalies, *quality_anomalies])

    return UMAPResult(
        coords=coords,
        diagnostics=diagnostics,
        anomalies=anomalies,
        trustworthiness=trust,
        shepard=shep,
    )


def _validate_embeddings_matrix(
    embeddings: FloatArray,
    cfg: UMAPConfig,
) -> tuple[FloatArray, tuple[str, ...]]:
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"embeddings must be a numpy array, got {type(embeddings)!r}.")

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}.")

    n_samples = int(embeddings.shape[0])
    n_dims = int(embeddings.shape[1])

    if n_samples < 2:
        raise ValueError("embeddings must contain at least 2 rows.")
    if n_dims < 1:
        raise ValueError("embeddings must have at least 1 column.")

    if cfg.metric != "cosine":
        raise ValueError(f"unsupported metric={cfg.metric!r}; expected 'cosine'.")

    if cfg.n_neighbors <= 1:
        raise ValueError(f"n_neighbors must be > 1, got {cfg.n_neighbors}.")
    if cfg.n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors must be < n_samples={n_samples}, got {cfg.n_neighbors}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values.")

    if not (0.0 <= float(cfg.min_dist) <= 1.0):
        raise ValueError(f"min_dist must be in [0, 1], got {cfg.min_dist}.")

    anomalies: list[str] = []

    norms = np.linalg.norm(embeddings.astype(np.float64, copy=False), axis=1)
    min_norm = float(np.min(norms)) if norms.size > 0 else 0.0
    max_norm = float(np.max(norms)) if norms.size > 0 else 0.0
    if min_norm <= 0.0:
        raise ValueError("embeddings contain a zero-norm vector; cosine is ill-posed.")
    if max_norm / min_norm > 10.0:
        anomalies.append("large embedding norm variation; cosine neighborhoods may be unstable")

    X = np.asarray(embeddings, dtype=np.float32)
    return X, tuple(anomalies)


def _fit_umap_2d(embeddings: FloatArray, cfg: UMAPConfig) -> FloatArray:
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Missing optional dependency 'umap-learn'. Install with: pip install umap-learn"
        ) from e

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        metric=str(cfg.metric),
        random_state=None if cfg.seed is None else int(cfg.seed),
    )

    coords = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
    if coords.ndim != 2 or int(coords.shape[1]) != 2:
        raise ValueError(f"UMAP returned unexpected shape {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError("UMAP produced non-finite coordinates.")

    return coords


def _compute_layout_diagnostics(
    embeddings: FloatArray,
    coords: FloatArray,
    cfg: UMAPConfig,
) -> UMAPDiagnostics:
    n_samples = int(embeddings.shape[0])
    n_dims = int(embeddings.shape[1])

    x = coords[:, 0].astype(np.float64, copy=False)
    y = coords[:, 1].astype(np.float64, copy=False)
    norms = np.linalg.norm(coords.astype(np.float64, copy=False), axis=1)

    return UMAPDiagnostics(
        n_samples=n_samples,
        n_dims=n_dims,
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        metric=str(cfg.metric),
        coord_min_x=float(np.min(x)),
        coord_max_x=float(np.max(x)),
        coord_min_y=float(np.min(y)),
        coord_max_y=float(np.max(y)),
        coord_mean_norm=float(np.mean(norms)),
        coord_std_norm=float(np.std(norms)),
    )


def _compute_quality_metrics(
    embeddings: FloatArray,
    coords: FloatArray,
    cfg: UMAPConfig,
) -> tuple[Trustworthiness, Shepard, tuple[str, ...]]:
    anomalies: list[str] = []

    n = int(embeddings.shape[0])
    k_default = min(int(cfg.n_neighbors), 15)
    k = int(cfg.quality_k) if cfg.quality_k is not None else k_default

    if k <= 0:
        raise ValueError(f"quality_k must be positive, got {k}.")
    if k >= (n / 2.0):
        raise ValueError(f"quality_k must be < n/2 for trustworthiness, got k={k}, n={n}.")

    trust = score_trustworthiness(
        embeddings,
        coords,
        k=k,
        metric_high="cosine",
        metric_low="euclidean",
        reference=bool(cfg.quality_reference),
        subsample_n=cfg.quality_subsample_n,
        subsample_seed=int(cfg.quality_seed),
    )

    shep = score_shepard(
        embeddings,
        coords,
        n_pairs=int(cfg.shepard_n_pairs),
        seed=int(cfg.quality_seed),
    )

    anomalies.extend(trust.anomalies)
    anomalies.extend(shep.anomalies)
    return trust, shep, tuple(anomalies)
