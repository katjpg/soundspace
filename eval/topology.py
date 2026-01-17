from dataclasses import dataclass, field
from typing import TypeAlias

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.stats import skew

Adjacency: TypeAlias = sparse.spmatrix | np.ndarray


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
    mean_clustering: float
    diagnostics: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Modularity:
    """Community structure quality metrics."""

    score: float
    internal_density: dict[int, float]
    external_density: dict[int, float]
    community_sizes: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_graph_structure(adjacency: Adjacency) -> GraphStructure:
    """
    Quantify graph connectivity and degree distribution properties.

    Args:
        adjacency (Adjacency) : adjacency matrix (n, n).

    Returns:
        (GraphStructure) : connectivity metrics with diagnostics and anomalies.

    Notes:
        - fragmentation = 1 - (largest_component_size / n)
        - mean_clustering computed on largest component for efficiency
    """
    adj, n = _as_square_csr(adjacency)

    anomalies: list[str] = []
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
            mean_clustering=0.0,
            diagnostics={"n_nodes": n, "n_edges": 0},
            anomalies=anomalies,
        )

    G = nx.from_scipy_sparse_array(adj)

    n_edges = int(G.number_of_edges())
    n_components, largest_nodes = _component_summary(G)
    largest_size = int(len(largest_nodes))
    fragmentation = float(1.0 - (largest_size / n))

    degrees = np.fromiter((d for _, d in G.degree()), dtype=np.int64, count=n)
    mean_degree = float(np.mean(degrees))
    std_degree = float(np.std(degrees))
    degree_skewness = float(skew(degrees)) if degrees.size > 0 else 0.0

    mean_clustering = _mean_clustering_on_nodes(G, largest_nodes)

    if n_components > 1:
        anomalies.append(f"fragmented graph: {n_components} components")
    if fragmentation > 0.10:
        anomalies.append(f"high fragmentation ({fragmentation:.1%})")
    if degree_skewness > 2.0:
        anomalies.append(f"heavy-tailed degree distribution (skew={degree_skewness:.2f})")

    diagnostics = {
        "n_nodes": n,
        "n_edges": n_edges,
        "density_undirected": float((2 * n_edges) / (n * (n - 1))) if n > 1 else 0.0,
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "median_degree": float(np.median(degrees)),
    }

    return GraphStructure(
        n_nodes=n,
        n_edges=n_edges,
        n_components=n_components,
        largest_component_size=largest_size,
        fragmentation=fragmentation,
        mean_degree=mean_degree,
        std_degree=std_degree,
        degree_skewness=degree_skewness,
        mean_clustering=mean_clustering,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def score_modularity(adjacency: Adjacency, membership: np.ndarray) -> Modularity:
    """
    Quantify community structure quality via modularity and density metrics.

    Args:
        adjacency  (Adjacency) : adjacency matrix (n, n).
        membership (np.ndarray) : community labels (n,).

    Returns:
        (Modularity) : modularity score with per-community diagnostics.
    """
    adj, n = _as_square_csr(adjacency)

    if membership.ndim != 1:
        raise ValueError(f"membership must be 1D, got shape {membership.shape}")
    if int(membership.shape[0]) != n:
        raise ValueError(f"membership length ({membership.shape[0]}) must match n ({n})")

    anomalies: list[str] = []
    if n < 2:
        anomalies.append("insufficient nodes (n < 2)")
        return Modularity(
            score=0.0,
            internal_density={},
            external_density={},
            community_sizes=np.array([], dtype=np.int64),
            diagnostics={"n_nodes": n, "n_communities": 0},
            anomalies=anomalies,
        )

    G = nx.from_scipy_sparse_array(adj)

    labels = np.unique(membership)
    communities = [set(np.flatnonzero(membership == lab).tolist()) for lab in labels]
    modularity_score = float(nx.algorithms.community.modularity(G, communities))

    internal_density, external_density, community_sizes = _community_densities(G, membership, labels)

    n_communities = int(labels.size)
    if modularity_score < 0.3:
        anomalies.append(f"low modularity ({modularity_score:.3f})")
    if n_communities > (n // 2):
        anomalies.append(f"over-fragmented: {n_communities} communities for {n} nodes")

    size_cv = float(np.std(community_sizes) / np.mean(community_sizes)) if community_sizes.size > 0 else 0.0
    if size_cv > 2.0:
        anomalies.append(f"extreme size imbalance (CV={size_cv:.2f})")

    diagnostics = {
        "n_nodes": n,
        "n_communities": n_communities,
        "mean_community_size": float(np.mean(community_sizes)) if community_sizes.size > 0 else 0.0,
        "min_community_size": int(np.min(community_sizes)) if community_sizes.size > 0 else 0,
        "max_community_size": int(np.max(community_sizes)) if community_sizes.size > 0 else 0,
        "size_coefficient_variation": size_cv,
    }

    return Modularity(
        score=modularity_score,
        internal_density=internal_density,
        external_density=external_density,
        community_sizes=community_sizes,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def diagnose_graph(adjacency: Adjacency, membership: np.ndarray | None = None) -> dict[str, GraphStructure | Modularity]:
    """
    Compute graph structure diagnostics and optional modularity diagnostics.

    Args:
        adjacency  (Adjacency) : adjacency matrix (n, n).
        membership (np.ndarray | None) : community labels (n,).

    Returns:
        (dict[str, GraphStructure | Modularity]) : keys "structure" and optionally "modularity".
    """
    structure = score_graph_structure(adjacency)
    report: dict[str, GraphStructure | Modularity] = {"structure": structure}

    if membership is None:
        return report

    report["modularity"] = score_modularity(adjacency, membership)
    return report


def _as_square_csr(adjacency: Adjacency) -> tuple[sparse.csr_matrix, int]:
    adj = sparse.csr_matrix(adjacency)

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


def _mean_clustering_on_nodes(G: nx.Graph, nodes: set[int]) -> float:
    if len(nodes) < 2:
        return 0.0
    return float(nx.average_clustering(G.subgraph(nodes)))


def _community_densities(
    G: nx.Graph,
    membership: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[int, float], dict[int, float], np.ndarray]:
    n = int(membership.shape[0])

    internal_density: dict[int, float] = {}
    external_density: dict[int, float] = {}
    community_sizes = np.zeros(int(labels.size), dtype=np.int64)

    for community_id, lab in enumerate(labels):
        nodes = set(np.flatnonzero(membership == lab).tolist())
        size = int(len(nodes))
        community_sizes[community_id] = size

        if size < 2:
            internal_density[community_id] = 0.0
            external_density[community_id] = 0.0
            continue

        internal_edges, external_edges = _internal_external_edge_counts(G, nodes)

        internal_possible = float(size * (size - 1))
        external_possible = float(size * (n - size))

        internal_density[community_id] = float(internal_edges / internal_possible) if internal_possible > 0 else 0.0
        external_density[community_id] = float(external_edges / external_possible) if external_possible > 0 else 0.0

    return internal_density, external_density, community_sizes


def _internal_external_edge_counts(G: nx.Graph, nodes: set[int]) -> tuple[int, int]:
    internal_edges = 0
    external_edges = 0

    for node in nodes:
        neighbors = set(G.neighbors(node))
        internal_edges += len(neighbors & nodes)
        external_edges += len(neighbors - nodes)

    return internal_edges, external_edges
