from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

SymmetrizeMode: TypeAlias = Literal["max", "mean", "min"]
NNAlgorithm: TypeAlias = Literal["auto", "ball_tree", "kd_tree", "brute"]


@dataclass(frozen=True)
class KNNConfig:
    """Configure kNN affinity graph construction.

    Attributes
    ----------
    k : int
        Number of neighbors per node in the directed kNN cache.
    symmetrize : bool
        If True, convert the directed kNN cache into an undirected adjacency.
    symmetrize_mode : SymmetrizeMode
        Method for symmetrization when symmetrize is True.
    clip_affinity : bool
        If True, clip affinity weights to [0.0, 1.0] both pre- and post-CSR construction.
    drop_diagonal : bool
        If True, enforce a zero diagonal after sparse construction and symmetrization.
    cache_directed : bool
        If True, store knn_indices/knn_distances and directed_adjacency for diagnostics.
    eps_diagonal : float
        Tolerance used by diagnostics to treat near-zero diagonal as zero.
    nn_algorithm : NNAlgorithm
        NearestNeighbors algorithm parameter.
    n_jobs : int | None
        NearestNeighbors n_jobs parameter.
    dtype : np.dtype
        Floating dtype for sparse weights.
    """

    k: int = 15
    symmetrize: bool = True
    symmetrize_mode: SymmetrizeMode = "max"
    clip_affinity: bool = True
    drop_diagonal: bool = True
    cache_directed: bool = True
    eps_diagonal: float = 1e-12
    nn_algorithm: NNAlgorithm = "auto"
    n_jobs: int | None = None
    dtype: np.dtype[Any] = np.dtype(np.float32)


@dataclass
class KNN:
    """Hold a kNN graph as an affinity adjacency plus optional directed caches.

    Attributes
    ----------
    adjacency : sp.csr_matrix
        Sparse affinity adjacency used as the core graph contract.
    config : KNNConfig
        Configuration used to build the graph.
    directed_adjacency : sp.csr_matrix | None
        Directed kNN adjacency. Present only if config.cache_directed is True.
    knn_indices : np.ndarray | None
        Directed neighbor indices with shape (n_nodes, k). Present only if cached.
    knn_distances : np.ndarray | None
        Directed cosine distances with shape (n_nodes, k). Present only if cached.
    """

    adjacency: sp.csr_matrix
    config: KNNConfig
    directed_adjacency: sp.csr_matrix | None = None
    knn_indices: np.ndarray | None = None
    knn_distances: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        shape = self.adjacency.shape
        if shape is None:
            raise ValueError("adjacency.shape is None.")
        return int(shape[0])

    def reciprocity(self) -> float:
        """Compute the fraction of directed edges that are reciprocated.

        Returns
        -------
        float
            mutual.nnz / directed.nnz, or 0.0 when the directed cache is empty.

        Raises
        ------
        ValueError
            If directed_adjacency is not available.
        """
        directed = self.directed_adjacency
        if directed is None:
            raise ValueError("directed_adjacency is not available (cache_directed=False).")

        if directed.nnz == 0:
            return 0.0

        mutual = directed.multiply(directed.T)
        return float(mutual.nnz) / float(directed.nnz)

    def diagonal_abs_sum(self) -> float:
        """Return the sum of absolute diagonal values for adjacency diagnostics."""
        diag = self.adjacency.diagonal()
        return float(np.sum(np.abs(diag)))


def build_knn(embeddings: np.ndarray, config: KNNConfig | None = None) -> KNN:
    """Build a kNN affinity graph from an embedding matrix.

    Args
    ----
    embeddings (np.ndarray)      : Embedding matrix of shape (n_samples, n_dims).
    config (KNNConfig | None)    : Construction configuration. Default is None.

    Returns
    -------
    KNN
        KNN graph with an affinity adjacency and optional directed caches.

    Raises
    ------
    ValueError
        If embeddings is not 2D, contains non-finite values, or k is out of range.
    """
    cfg = config or KNNConfig()
    _validate_embeddings(embeddings=embeddings, k=cfg.k)

    knn_indices, knn_distances = _compute_knn_cosine(
        embeddings=embeddings,
        k=cfg.k,
        algorithm=cfg.nn_algorithm,
        n_jobs=cfg.n_jobs,
    )

    shape = embeddings.shape
    n_nodes = int(shape[0])

    directed = _build_directed_affinity(
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        n_nodes=n_nodes,
        clip_affinity=cfg.clip_affinity,
        drop_diagonal=cfg.drop_diagonal,
        dtype=cfg.dtype,
    )

    adjacency = (
        _symmetrize_adjacency(
            directed=directed,
            mode=cfg.symmetrize_mode,
            drop_diagonal=cfg.drop_diagonal,
            clip_affinity=cfg.clip_affinity,
        )
        if cfg.symmetrize
        else directed
    )

    if cfg.cache_directed:
        return KNN(
            adjacency=adjacency,
            config=cfg,
            directed_adjacency=directed,
            knn_indices=knn_indices,
            knn_distances=knn_distances,
        )

    return KNN(adjacency=adjacency, config=cfg)


def _validate_embeddings(embeddings: np.ndarray, k: int) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}.")

    if embeddings.shape[0] < 2:
        raise ValueError("embeddings must contain at least 2 rows.")

    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")

    if k >= int(embeddings.shape[0]):
        raise ValueError(f"k must be < n_samples={embeddings.shape[0]}, got {k}.")

    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values.")


def _compute_knn_cosine(
    embeddings: np.ndarray,
    k: int,
    algorithm: NNAlgorithm,
    n_jobs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm=algorithm,
        n_jobs=n_jobs,
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)
    return indices[:, 1:], distances[:, 1:]


def _build_directed_affinity(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    n_nodes: int,
    clip_affinity: bool,
    drop_diagonal: bool,
    dtype: np.dtype,
) -> sp.csr_matrix:
    k = int(knn_indices.shape[1])
    rows = np.repeat(np.arange(n_nodes, dtype=np.int32), k)
    cols = knn_indices.reshape(-1).astype(np.int32, copy=False)

    affinity = (1.0 - knn_distances).reshape(-1).astype(dtype, copy=False)
    if clip_affinity:
        affinity = np.clip(affinity, 0.0, 1.0, out=affinity)

    directed = sp.csr_matrix((affinity, (rows, cols)), shape=(n_nodes, n_nodes), dtype=dtype)

    directed.sum_duplicates()
    if clip_affinity:
        directed.data = np.clip(directed.data, 0.0, 1.0)

    if drop_diagonal:
        _zero_diagonal_inplace(directed)

    return directed


def _symmetrize_adjacency(
    directed: sp.csr_matrix,
    mode: SymmetrizeMode,
    drop_diagonal: bool,
    clip_affinity: bool,
) -> sp.csr_matrix:
    if mode == "max":
        undirected = directed.maximum(directed.T)
    elif mode == "min":
        undirected = directed.minimum(directed.T)
    elif mode == "mean":
        undirected = (directed + directed.T) * 0.5
    else:
        raise ValueError(f"unsupported symmetrize_mode={mode!r}.")

    undirected = undirected.tocsr()
    undirected.sum_duplicates()

    if clip_affinity:
        undirected.data = np.clip(undirected.data, 0.0, 1.0)

    if drop_diagonal:
        _zero_diagonal_inplace(undirected)

    return undirected


def _zero_diagonal_inplace(matrix: sp.csr_matrix) -> None:
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
