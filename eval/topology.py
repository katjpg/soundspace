from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix, diags, spmatrix
from scipy.stats import skew


Adjacency: TypeAlias = spmatrix | np.ndarray
CSRAdjacency: TypeAlias = csr_matrix
Diagnostics: TypeAlias = dict[str, float]

IntArray: TypeAlias = NDArray[np.integer]
FloatArray: TypeAlias = NDArray[np.floating]

SymmetrizeMethod: TypeAlias = Literal["max", "mean"]

CommunityScalarMap: TypeAlias = dict[int, float]
CommunitySizes: TypeAlias = IntArray
CommunityPairMetrics: TypeAlias = tuple[
    CommunityScalarMap,
    CommunityScalarMap,
    CommunityScalarMap,
    CommunityScalarMap,
    CommunitySizes,
]
PrepareResult: TypeAlias = tuple[CSRAdjacency, int, bool, Diagnostics, list[str]]

WEIGHT_ATTR = "weight"
WEIGHT_RANGE_EPS = 1e-9
SYMMETRIZE_METHODS: tuple[str, ...] = ("max", "mean")


@dataclass(frozen=True, slots=True)
class GraphStructure:
    """Graph connectivity and degree distribution metrics."""

    n_nodes: int
    n_edges: int
    n_components: int
    largest_component_size: int
    fragmentation: float

    mean_degree: float
    std_degree: float
    degree_skewness: float

    mean_strength: float
    std_strength: float
    strength_skewness: float

    mean_clustering: float
    mean_clustering_weighted: float

    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Modularity:
    """Community structure quality metrics."""

    score: float

    internal_density: dict[int, float]
    external_density: dict[int, float]

    internal_affinity: dict[int, float]
    external_affinity: dict[int, float]

    community_sizes: CommunitySizes

    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_graph_structure(
    adjacency: Adjacency,
    *,
    symmetrize: bool = True,
    symmetrize_method: SymmetrizeMethod = "max",
    allow_self_loops: bool = False,
    require_nonnegative: bool = True,
) -> GraphStructure:
    """
    Quantify graph connectivity and degree distribution properties.

    Args:
        adjacency           (Adjacency)        : adjacency matrix (n, n).
        symmetrize          (bool)             : if True, symmetrize adjacency before graph construction.
                                                 (Default is True).
        symmetrize_method   (SymmetrizeMethod) : symmetrization method: "max" (union) or "mean" (average).
                                                 (Default is "max").
        allow_self_loops    (bool)             : if False, drop diagonal values before any symmetrization.
                                                 (Default is False).
        require_nonnegative (bool)             : if True, raise if any weight is negative.
                                                 (Default is True).

    Returns:
        (GraphStructure) : connectivity and distribution summaries with diagnostics and anomalies.
    """
    adj, n, has_weight_attribute, prep_diagnostics, prep_anomalies = _prepare_adjacency(
        adjacency,
        symmetrize=symmetrize,
        symmetrize_method=symmetrize_method,
        allow_self_loops=allow_self_loops,
        require_nonnegative=require_nonnegative,
    )

    anomalies: list[str] = list(prep_anomalies)

    if n < 2:
        anomalies.append("insufficient nodes (n < 2)")
        return GraphStructure(
            n_nodes=n,
            n_edges=0,
            n_components=0,
            largest_component_size=0,
            fragmentation=1.0,
            mean_degree=0.0,
            std_degree=0.0,
            degree_skewness=0.0,
            mean_strength=0.0,
            std_strength=0.0,
            strength_skewness=0.0,
            mean_clustering=0.0,
            mean_clustering_weighted=0.0,
            diagnostics={**prep_diagnostics, "n_nodes": float(n), "n_edges": 0.0},
            anomalies=anomalies,
        )

    G = nx.from_scipy_sparse_array(
        adj, create_using=nx.Graph, edge_attribute=WEIGHT_ATTR
    )

    n_edges = int(G.number_of_edges())
    n_components, largest_nodes = _component_summary(G)

    largest_size = int(len(largest_nodes))
    fragmentation = float(1.0 - (largest_size / float(n)))

    degrees = np.fromiter((d for _, d in G.degree()), dtype=np.int64, count=n).astype(
        np.float64, copy=False
    )
    strengths = np.fromiter(
        (s for _, s in G.degree(weight=WEIGHT_ATTR)),
        dtype=np.float64,
        count=n,
    ).astype(np.float64, copy=False)

    mean_degree = float(np.mean(degrees)) if degrees.size > 0 else 0.0
    std_degree = float(np.std(degrees)) if degrees.size > 0 else 0.0
    degree_skew = float(skew(degrees)) if degrees.size > 0 else 0.0
    if np.isnan(degree_skew):
        degree_skew = 0.0
        anomalies.append("degree skewness is NaN; treating as 0.0")

    mean_strength = float(np.mean(strengths)) if strengths.size > 0 else 0.0
    std_strength = float(np.std(strengths)) if strengths.size > 0 else 0.0
    strength_skew = float(skew(strengths)) if strengths.size > 0 else 0.0
    if np.isnan(strength_skew):
        strength_skew = 0.0
        anomalies.append("strength skewness is NaN; treating as 0.0")

    mean_clustering = _mean_clustering_on_nodes(G, largest_nodes, weight=None)

    # compute weighted clustering whenever a weight attribute exists
    mean_clustering_weighted = (
        _mean_clustering_on_nodes(G, largest_nodes, weight=WEIGHT_ATTR)
        if has_weight_attribute
        else 0.0
    )

    if n_components > 1:
        anomalies.append(f"fragmented graph: {n_components} components")
    if fragmentation > 0.10:
        anomalies.append(f"high fragmentation ({fragmentation:.1%})")
    if degree_skew > 2.0:
        anomalies.append(f"heavy-tailed degree distribution (skew={degree_skew:.2f})")

    diagnostics: Diagnostics = {
        **prep_diagnostics,
        "n_nodes": float(n),
        "n_edges": float(n_edges),
        "density_undirected": float((2.0 * float(n_edges)) / (float(n) * float(n - 1)))
        if n > 1
        else 0.0,
        "min_degree": float(np.min(degrees)) if degrees.size > 0 else 0.0,
        "max_degree": float(np.max(degrees)) if degrees.size > 0 else 0.0,
        "median_degree": float(np.median(degrees)) if degrees.size > 0 else 0.0,
        "min_strength": float(np.min(strengths)) if strengths.size > 0 else 0.0,
        "max_strength": float(np.max(strengths)) if strengths.size > 0 else 0.0,
        "median_strength": float(np.median(strengths)) if strengths.size > 0 else 0.0,
        "weighted_clustering_enabled": float(has_weight_attribute),
        "weighted_clustering_informative": float(
            _diag_flag(prep_diagnostics, "has_weight_variation")
        ),
    }

    return GraphStructure(
        n_nodes=n,
        n_edges=n_edges,
        n_components=n_components,
        largest_component_size=largest_size,
        fragmentation=fragmentation,
        mean_degree=mean_degree,
        std_degree=std_degree,
        degree_skewness=degree_skew,
        mean_strength=mean_strength,
        std_strength=std_strength,
        strength_skewness=strength_skew,
        mean_clustering=mean_clustering,
        mean_clustering_weighted=mean_clustering_weighted,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def score_modularity(
    adjacency: Adjacency,
    membership: IntArray,
    *,
    symmetrize: bool = True,
    symmetrize_method: SymmetrizeMethod = "max",
    allow_self_loops: bool = False,
    require_nonnegative: bool = True,
    resolution: float = 1.0,
) -> Modularity:
    """
    Quantify community structure quality via modularity and per-community pair metrics.

    Args:
        adjacency           (Adjacency)        : adjacency matrix (n, n).
        membership          (IntArray)         : community labels (n,).
        symmetrize          (bool)             : if True, symmetrize adjacency before graph construction.
                                                 (Default is True).
        symmetrize_method   (SymmetrizeMethod) : symmetrization method: "max" (union) or "mean" (average).
                                                 (Default is "max").
        allow_self_loops    (bool)             : if False, drop diagonal values before any symmetrization.
                                                 (Default is False).
        require_nonnegative (bool)             : if True, raise if any weight is negative.
                                                 (Default is True).
        resolution          (float)            : modularity resolution parameter.
                                                 (Default is 1.0).

    Returns:
        (Modularity) : modularity score with per-community diagnostics.

    Notes:
        This function treats adjacency values as affinity weights, not distances.
        If allow_self_loops=True, modularity will include self-loops; pair metrics drop them.
    """
    if membership.ndim != 1:
        raise ValueError(f"membership must be 1D, got shape {membership.shape}")
    if not np.issubdtype(membership.dtype, np.integer):
        raise ValueError(
            f"membership must be integer labels, got dtype {membership.dtype}"
        )

    adj, n, has_weight_attribute, prep_diagnostics, prep_anomalies = _prepare_adjacency(
        adjacency,
        symmetrize=symmetrize,
        symmetrize_method=symmetrize_method,
        allow_self_loops=allow_self_loops,
        require_nonnegative=require_nonnegative,
    )

    if int(membership.shape[0]) != n:
        raise ValueError(
            f"membership length ({membership.shape[0]}) must match n ({n})"
        )

    anomalies: list[str] = list(prep_anomalies)

    if n < 2:
        anomalies.append("insufficient nodes (n < 2)")
        return Modularity(
            score=0.0,
            internal_density={},
            external_density={},
            internal_affinity={},
            external_affinity={},
            community_sizes=np.array([], dtype=np.int64),
            diagnostics={**prep_diagnostics, "n_nodes": float(n), "n_communities": 0.0},
            anomalies=anomalies,
        )

    if allow_self_loops and _diag_flag(prep_diagnostics, "has_self_loops"):
        anomalies.append(
            "self-loops retained; modularity includes them; pair metrics drop them"
        )

    G = nx.from_scipy_sparse_array(
        adj, create_using=nx.Graph, edge_attribute=WEIGHT_ATTR
    )

    labels = np.unique(membership)
    communities = [set(np.flatnonzero(membership == lab).tolist()) for lab in labels]

    modularity_score = float(
        nx.algorithms.community.modularity(
            G,
            communities,
            weight=WEIGHT_ATTR if has_weight_attribute else None,
            resolution=float(resolution),
        )
    )

    assume_symmetric_zero_diag = _diag_flag(prep_diagnostics, "is_symmetric") and (
        not _diag_flag(prep_diagnostics, "has_self_loops")
    )

    (
        internal_density,
        external_density,
        internal_affinity,
        external_affinity,
        community_sizes,
    ) = _community_pair_metrics(
        adj,
        membership,
        labels.astype(np.int64, copy=False),
        assume_symmetric_zero_diag=assume_symmetric_zero_diag,
    )

    n_communities = int(labels.size)

    if modularity_score < 0.3:
        anomalies.append(f"low modularity ({modularity_score:.3f})")
    if n_communities > (n // 2):
        anomalies.append(f"over-fragmented: {n_communities} communities for {n} nodes")

    size_cv = (
        float(np.std(community_sizes) / np.mean(community_sizes))
        if community_sizes.size > 0
        else 0.0
    )
    if size_cv > 2.0:
        anomalies.append(f"extreme size imbalance (CV={size_cv:.2f})")

    diagnostics: Diagnostics = {
        **prep_diagnostics,
        "n_nodes": float(n),
        "n_communities": float(n_communities),
        "mean_community_size": float(np.mean(community_sizes))
        if community_sizes.size > 0
        else 0.0,
        "min_community_size": float(np.min(community_sizes))
        if community_sizes.size > 0
        else 0.0,
        "max_community_size": float(np.max(community_sizes))
        if community_sizes.size > 0
        else 0.0,
        "size_coefficient_variation": float(size_cv),
        "resolution": float(resolution),
        "weighted_modularity": float(has_weight_attribute),
        "assume_symmetric_zero_diag": float(assume_symmetric_zero_diag),
        "modularity_includes_self_loops": float(
            allow_self_loops and _diag_flag(prep_diagnostics, "has_self_loops")
        ),
    }

    return Modularity(
        score=modularity_score,
        internal_density=internal_density,
        external_density=external_density,
        internal_affinity=internal_affinity,
        external_affinity=external_affinity,
        community_sizes=community_sizes,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def diagnose_graph(
    adjacency: Adjacency,
    membership: IntArray | None = None,
    *,
    symmetrize: bool = True,
    symmetrize_method: SymmetrizeMethod = "max",
    allow_self_loops: bool = False,
    require_nonnegative: bool = True,
    resolution: float = 1.0,
) -> dict[str, GraphStructure | Modularity]:
    """
    Compute graph structure diagnostics and optional modularity diagnostics.

    Args:
        adjacency           (Adjacency)        : adjacency matrix (n, n).
        membership (IntArray | None)           : community labels (n,).
        symmetrize          (bool)             : if True, symmetrize adjacency before graph construction.
                                                 (Default is True).
        symmetrize_method   (SymmetrizeMethod) : symmetrization method: "max" (union) or "mean" (average).
                                                 (Default is "max").
        allow_self_loops    (bool)             : if False, drop diagonal values before any symmetrization.
                                                 (Default is False).
        require_nonnegative (bool)             : if True, raise if any weight is negative.
                                                 (Default is True).
        resolution          (float)            : modularity resolution parameter.
                                                 (Default is 1.0).

    Returns:
        (dict[str, GraphStructure | Modularity]) : keys "structure" and optionally "modularity".
    """
    structure = score_graph_structure(
        adjacency,
        symmetrize=symmetrize,
        symmetrize_method=symmetrize_method,
        allow_self_loops=allow_self_loops,
        require_nonnegative=require_nonnegative,
    )
    report: dict[str, GraphStructure | Modularity] = {"structure": structure}

    if membership is None:
        return report

    report["modularity"] = score_modularity(
        adjacency,
        membership,
        symmetrize=symmetrize,
        symmetrize_method=symmetrize_method,
        allow_self_loops=allow_self_loops,
        require_nonnegative=require_nonnegative,
        resolution=resolution,
    )
    return report


def _diag_flag(diagnostics: Diagnostics, key: str) -> bool:
    """Interpret a float-valued diagnostics flag as a boolean."""
    return bool(float(diagnostics.get(key, 0.0)) > 0.5)


def _prepare_adjacency(
    adjacency: Adjacency,
    *,
    symmetrize: bool,
    symmetrize_method: SymmetrizeMethod,
    allow_self_loops: bool,
    require_nonnegative: bool,
) -> PrepareResult:
    adj, n = _as_square_csr(adjacency)

    anomalies: list[str] = []
    diagnostics: Diagnostics = {
        "n_nodes": float(n),
        "affinity_contract": 1.0,
    }

    if adj.nnz == 0:
        diagnostics["min_weight"] = 0.0
        diagnostics["max_weight"] = 0.0
        diagnostics["weight_range"] = 0.0
        diagnostics["has_weight_attribute"] = 0.0
        diagnostics["has_weight_variation"] = 0.0
        diagnostics["is_symmetric"] = 1.0
        diagnostics["has_self_loops"] = 0.0
        diagnostics["symmetrize_enabled"] = float(symmetrize)
        diagnostics["symmetrized_applied"] = 0.0
        diagnostics["allow_self_loops"] = float(allow_self_loops)
        diagnostics["asym_pre_nnz"] = 0.0
        diagnostics["asym_post_nnz"] = 0.0
        return adj, n, False, diagnostics, anomalies

    if not np.all(np.isfinite(adj.data)):
        raise ValueError("adjacency contains non-finite values")

    min_weight_pre = float(np.min(adj.data)) if adj.data.size > 0 else 0.0
    if require_nonnegative and min_weight_pre < 0.0:
        raise ValueError(
            f"adjacency contains negative weights (min={min_weight_pre:.3g})"
        )

    diag = adj.diagonal()
    diag_abs_sum = float(np.sum(np.abs(diag))) if diag.size > 0 else 0.0
    diagnostics["diag_abs_sum"] = float(diag_abs_sum)

    has_self_loops = bool(diag_abs_sum > 0.0)
    if not allow_self_loops and has_self_loops:
        anomalies.append("nonzero diagonal detected; dropping self-loops")
        adj = cast(CSRAdjacency, (adj - diags(diag, offsets=0, format="csr")).tocsr())
        adj.eliminate_zeros()
        has_self_loops = False

    if symmetrize_method not in SYMMETRIZE_METHODS:
        raise ValueError(
            f"unknown symmetrize_method '{symmetrize_method}', expected one of {SYMMETRIZE_METHODS}"
        )

    asym_pre = cast(CSRAdjacency, (adj - adj.T).tocsr())
    asym_pre.eliminate_zeros()
    asym_pre_nnz = int(asym_pre.nnz)
    diagnostics["asym_pre_nnz"] = float(asym_pre_nnz)

    symmetrized_applied = 0.0
    if asym_pre_nnz > 0:
        max_abs_asym = (
            float(np.max(np.abs(asym_pre.data))) if asym_pre.data.size > 0 else 0.0
        )
        diagnostics["max_abs_asym"] = float(max_abs_asym)

        if symmetrize:
            anomalies.append(
                f"asymmetric adjacency detected; symmetrizing via '{symmetrize_method}'"
            )
            if symmetrize_method == "max":
                adj = cast(CSRAdjacency, adj.maximum(adj.T).tocsr())
            else:
                adj = cast(CSRAdjacency, ((adj + adj.T) * 0.5).tocsr())
            adj.eliminate_zeros()
            symmetrized_applied = 1.0
        else:
            anomalies.append(
                "asymmetric adjacency detected but symmetrize=False; interpreting as undirected may be misleading"
            )

    asym_post = cast(CSRAdjacency, (adj - adj.T).tocsr())
    asym_post.eliminate_zeros()
    asym_post_nnz = int(asym_post.nnz)
    diagnostics["asym_post_nnz"] = float(asym_post_nnz)

    weight_min = float(np.min(adj.data)) if adj.data.size > 0 else 0.0
    weight_max = float(np.max(adj.data)) if adj.data.size > 0 else 0.0
    weight_range = float(weight_max - weight_min)

    has_weight_attribute = bool(adj.data.size > 0)
    has_weight_variation = bool(
        has_weight_attribute and weight_range > WEIGHT_RANGE_EPS
    )

    diagnostics["min_weight"] = float(weight_min)
    diagnostics["max_weight"] = float(weight_max)
    diagnostics["weight_range"] = float(weight_range)
    diagnostics["has_weight_attribute"] = float(has_weight_attribute)
    diagnostics["has_weight_variation"] = float(has_weight_variation)

    is_symmetric = bool(asym_post_nnz == 0)
    diagnostics["is_symmetric"] = float(is_symmetric)

    diagnostics["symmetrize_enabled"] = float(symmetrize)
    diagnostics["symmetrized_applied"] = float(symmetrized_applied)
    diagnostics["allow_self_loops"] = float(allow_self_loops)
    diagnostics["has_self_loops"] = float(has_self_loops)

    return adj, n, has_weight_attribute, diagnostics, anomalies


def _as_square_csr(adjacency: Adjacency) -> tuple[CSRAdjacency, int]:
    adj = cast(CSRAdjacency, sparse.csr_matrix(adjacency))

    shape = adj.shape
    if shape is None:
        raise ValueError("adjacency has no shape (unexpected scipy sparse state)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"adjacency must be square, got shape {shape}")

    return adj, n_rows


def _component_summary(G: nx.Graph) -> tuple[int, set[int]]:
    components = list(nx.connected_components(G))
    if not components:
        return 0, set()

    largest = max(components, key=len)
    return int(len(components)), set(largest)


def _mean_clustering_on_nodes(
    G: nx.Graph, nodes: set[int], *, weight: str | None
) -> float:
    if len(nodes) < 2:
        return 0.0

    # set count_zeros explicitly to stabilize interpretation across NetworkX versions
    return float(
        nx.average_clustering(G.subgraph(nodes), weight=weight, count_zeros=True)
    )


def _slice_csr(adj: CSRAdjacency, row_idx: IntArray, col_idx: IntArray) -> CSRAdjacency:
    """
    Slice a CSR adjacency using tuple indexing.

    This avoids chained indexing (A[rows][:, cols]) which can confuse type checkers.
    """
    rows = np.asarray(row_idx, dtype=np.int64)
    cols = np.asarray(col_idx, dtype=np.int64)
    return cast(CSRAdjacency, sparse.csr_matrix(adj[rows[:, None], cols]))


def _community_pair_metrics(
    adj: CSRAdjacency,
    membership: IntArray,
    labels: IntArray,
    *,
    assume_symmetric_zero_diag: bool,
) -> CommunityPairMetrics:
    """
    Compute per-community internal/external densities and mean affinities.

    Args:
        adj                        (CSRAdjacency) : adjacency matrix (n, n).
        membership                    (IntArray)  : integer community labels (n,).
        labels                        (IntArray)  : unique labels present in membership.
        assume_symmetric_zero_diag       (bool)   : if True, treat adj as symmetric with zero diagonal.

    Returns:
        (CommunityPairMetrics) : (internal_density, external_density, internal_affinity, external_affinity, sizes).

    Notes:
        When assume_symmetric_zero_diag is False, this function constructs an undirected view via max(A, A.T)
        and drops diagonal contributions to keep pair denominators interpretable.
    """
    n = int(membership.shape[0])

    internal_density: dict[int, float] = {}
    external_density: dict[int, float] = {}
    internal_affinity: dict[int, float] = {}
    external_affinity: dict[int, float] = {}

    community_sizes = np.zeros(int(labels.size), dtype=np.int64)

    adj_u: CSRAdjacency | None = None
    if not assume_symmetric_zero_diag:
        adj_u = cast(CSRAdjacency, adj.maximum(adj.T).tocsr())
        adj_u.eliminate_zeros()

        diag_u = adj_u.diagonal()
        diag_u_abs_sum = float(np.sum(np.abs(diag_u))) if diag_u.size > 0 else 0.0
        if diag_u_abs_sum > 0.0:
            adj_u = cast(
                CSRAdjacency, (adj_u - diags(diag_u, offsets=0, format="csr")).tocsr()
            )
            adj_u.eliminate_zeros()

    for idx, lab in enumerate(labels):
        label_int = int(lab)
        nodes = np.flatnonzero(membership == lab).astype(np.int64, copy=False)
        size = int(nodes.size)
        community_sizes[idx] = size

        if size < 2:
            internal_density[label_int] = 0.0
            external_density[label_int] = 0.0
            internal_affinity[label_int] = 0.0
            external_affinity[label_int] = 0.0
            continue

        complement_mask = np.ones(n, dtype=np.bool_)
        complement_mask[nodes] = False
        complement = np.flatnonzero(complement_mask).astype(np.int64, copy=False)

        A = adj if assume_symmetric_zero_diag else cast(CSRAdjacency, adj_u)

        sub = _slice_csr(A, nodes, nodes)
        cut = _slice_csr(A, nodes, complement)

        if assume_symmetric_zero_diag:
            sub.eliminate_zeros()
            internal_edges = int(sub.nnz // 2)
            internal_weight = float(sub.sum() * 0.5)
        else:
            sub_triu = sparse.triu(sub, k=1, format="csr")
            internal_edges = int(sub_triu.nnz)
            internal_weight = float(sub_triu.sum())

        external_edges = int(cut.nnz)
        external_weight = float(cut.sum())

        internal_possible = float(size * (size - 1) / 2.0)
        external_possible = float(size * (n - size))

        internal_density[label_int] = (
            float(internal_edges / internal_possible) if internal_possible > 0 else 0.0
        )
        external_density[label_int] = (
            float(external_edges / external_possible) if external_possible > 0 else 0.0
        )

        internal_affinity[label_int] = (
            float(internal_weight / internal_possible) if internal_possible > 0 else 0.0
        )
        external_affinity[label_int] = (
            float(external_weight / external_possible) if external_possible > 0 else 0.0
        )

    return (
        internal_density,
        external_density,
        internal_affinity,
        external_affinity,
        community_sizes,
    )
