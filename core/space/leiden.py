from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

IgraphGraph: TypeAlias = Any


class AffinityAdjacency(Protocol):
    """Expose an affinity adjacency matrix."""

    adjacency: sp.csr_matrix


class LeidenPartition(Protocol):
    @property
    def membership(self) -> Sequence[int]: ...

    @property
    def modularity(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class LeidenConfig:
    """Configure Leiden community detection.

    Attributes
    ----------
    resolution : float
        Leiden resolution parameter (RBConfigurationVertexPartition).
    n_iterations : int
        Iteration limit. Use -1 for convergence.
    seed : int | None
        Random seed for reproducibility.
    use_weights : bool
        If True, use affinity weights in the objective.
    enforce_undirected : bool
        If True, require adjacency symmetry before converting to igraph.
    eps_symmetric : float
        Tolerance for treating adjacency as symmetric.
    eps_diagonal : float
        Tolerance for treating diagonal as zero.
    """

    resolution: float = 1.0
    n_iterations: int = -1
    seed: int | None = None
    use_weights: bool = True
    enforce_undirected: bool = True
    eps_symmetric: float = 1e-12
    eps_diagonal: float = 1e-12


@dataclass(frozen=True, slots=True)
class LeidenDiagnostics:
    """Diagnostics computed from the affinity adjacency."""

    n_nodes: int
    nnz: int
    min_weight: float
    max_weight: float
    diagonal_abs_max: float
    max_abs_asymmetry: float


@dataclass(frozen=True, slots=True)
class LeidenResult:
    """Hold Leiden outputs for downstream analysis."""

    membership: NDArray[np.int64]
    modularity: float
    n_communities: int
    diagnostics: LeidenDiagnostics
    anomalies: tuple[str, ...]


def leiden_partition(
    graph: AffinityAdjacency, config: LeidenConfig | None = None
) -> LeidenResult:
    """Run Leiden community detection on an affinity adjacency.

    Args
    ----
    graph (AffinityAdjacency)    : Domain object exposing `.adjacency` as CSR affinity weights.
    config (LeidenConfig | None) : Leiden configuration. Default is None.

    Returns
    -------
    LeidenResult
        Membership array, modularity, and validation diagnostics/anomalies.
    """
    cfg = config or LeidenConfig()
    adjacency = graph.adjacency

    diagnostics, anomalies = _validate_affinity_adjacency(adjacency=adjacency, cfg=cfg)
    ig_graph = _to_igraph_undirected(adjacency=adjacency, use_weights=cfg.use_weights)

    partition = _run_leiden(
        graph=ig_graph,
        resolution=cfg.resolution,
        n_iterations=cfg.n_iterations,
        seed=cfg.seed,
        use_weights=cfg.use_weights,
    )

    membership = np.asarray(partition.membership, dtype=np.int64)
    n_communities = int(np.unique(membership).size)
    modularity_raw = partition.modularity
    if modularity_raw is None:
        raise ValueError("leidenalg returned modularity=None (unexpected).")
    modularity = float(modularity_raw)

    return LeidenResult(
        membership=membership,
        modularity=modularity,
        n_communities=n_communities,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def _validate_affinity_adjacency(
    adjacency: sp.csr_matrix,
    cfg: LeidenConfig,
) -> tuple[LeidenDiagnostics, tuple[str, ...]]:
    shape = adjacency.shape
    if shape is None:
        raise ValueError("adjacency.shape is None.")

    n_nodes = int(shape[0])
    if n_nodes != int(shape[1]):
        raise ValueError(f"adjacency must be square, got shape {shape}.")

    if adjacency.nnz == 0:
        raise ValueError("adjacency must have at least one non-zero entry.")

    anomalies: list[str] = []

    data = adjacency.data
    if data.size == 0:
        raise ValueError(
            "adjacency has nnz > 0 but empty .data; sparse structure is invalid."
        )

    if not np.isfinite(data).all():
        raise ValueError("adjacency contains non-finite edge weights.")

    min_weight = float(np.min(data))
    max_weight = float(np.max(data))
    if min_weight < 0.0:
        raise ValueError(
            f"adjacency must have non-negative affinity weights, min={min_weight}."
        )

    diag = adjacency.diagonal()
    diagonal_abs_max = float(np.max(np.abs(diag))) if diag.size > 0 else 0.0
    if diagonal_abs_max > cfg.eps_diagonal:
        raise ValueError(
            f"adjacency diagonal must be ~0, max(|diag|)={diagonal_abs_max} (eps={cfg.eps_diagonal})."
        )

    max_abs_asymmetry = 0.0
    if cfg.enforce_undirected:
        asym = (adjacency - adjacency.T).tocsr()
        if asym.nnz > 0:
            max_abs_asymmetry = float(np.max(np.abs(asym.data)))
        if max_abs_asymmetry > cfg.eps_symmetric:
            raise ValueError(
                "adjacency must be symmetric for undirected Leiden "
                f"(max(|A-A^T|)={max_abs_asymmetry}, eps={cfg.eps_symmetric})."
            )

    if min_weight == 0.0:
        anomalies.append("Some edges have zero affinity (min_weight=0.0).")

    diagnostics = LeidenDiagnostics(
        n_nodes=n_nodes,
        nnz=int(adjacency.nnz),
        min_weight=min_weight,
        max_weight=max_weight,
        diagonal_abs_max=diagonal_abs_max,
        max_abs_asymmetry=max_abs_asymmetry,
    )
    return diagnostics, tuple(anomalies)


def _to_igraph_undirected(adjacency: sp.csr_matrix, use_weights: bool) -> IgraphGraph:
    try:
        import igraph as ig
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Missing optional dependency 'igraph'. Install with: pip install python-igraph"
        ) from e

    shape = adjacency.shape
    if shape is None:
        raise ValueError("adjacency.shape is None.")

    upper = sp.triu(adjacency, k=1).tocoo()
    edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    graph = ig.Graph(n=int(shape[0]), edges=edges, directed=False)

    if use_weights:
        graph.es["weight"] = upper.data.tolist()

    return graph


def _run_leiden(
    graph: IgraphGraph,
    resolution: float,
    n_iterations: int,
    seed: int | None,
    use_weights: bool,
) -> LeidenPartition:
    try:
        import leidenalg
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Missing optional dependency 'leidenalg'. Install with: pip install leidenalg"
        ) from e

    weights_attr = "weight" if use_weights else None

    if seed is None:
        return leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights_attr,
            resolution_parameter=float(resolution),
            n_iterations=int(n_iterations),
        )

    return leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights_attr,
        resolution_parameter=float(resolution),
        n_iterations=int(n_iterations),
        seed=int(seed),
    )
