from dataclasses import dataclass
from typing import cast

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags

from dtypes import Adjacency, FloatArray, IntArray, SymmetrizeMode

WEIGHT_ATTR = "weight"


@dataclass(frozen=True, slots=True)
class GraphQuality:
    """
    kNN graph and community structure quality metrics.

    Args
    ----
        n_nodes          (int) : number of nodes in the graph.
        n_edges          (int) : number of edges in the undirected graph.
        n_components     (int) : number of connected components.
        modularity     (float) : Newman-Girvan modularity score.
        n_communities    (int) : number of unique community labels.
        community_sizes (IntArray) : size of each community, sorted by label.
        mean_degree    (float) : average node degree.
    """

    n_nodes: int
    n_edges: int
    n_components: int
    modularity: float
    n_communities: int
    community_sizes: IntArray
    mean_degree: float


def score_graph_quality(
    adjacency: Adjacency,
    membership: IntArray,
    *,
    resolution: float = 1.0,
    symmetrize: bool = True,
    symmetrize_mode: SymmetrizeMode = "max",
) -> GraphQuality:
    """
    Compute graph and community structure quality metrics.

    Args
    ----
        adjacency              (Adjacency) : adjacency matrix of shape (n, n).
        membership              (IntArray) : community labels of shape (n,).
        resolution               (float)   : modularity resolution parameter.
                                             (Default is 1.0).
        symmetrize                (bool)   : if True, symmetrize adjacency.
                                             (Default is True).
        symmetrize_mode     (SymmetrizeMode) : "max" (union) or "mean" (average).
                                               (Default is "max").

    Returns
    -------
        (GraphQuality) : graph connectivity and modularity metrics.
    """
    if membership.ndim != 1:
        raise ValueError(f"membership must be 1D, got shape {membership.shape}")

    # convert to CSR and validate
    adj = _to_csr(adjacency)
    n = int(adj.shape[0])  # type: ignore[index]

    if len(membership) != n:
        raise ValueError(
            f"membership length ({len(membership)}) must match adjacency size ({n})"
        )

    if n < 2:
        return GraphQuality(
            n_nodes=n,
            n_edges=0,
            n_components=0,
            modularity=0.0,
            n_communities=0,
            community_sizes=np.array([], dtype=np.int64),
            mean_degree=0.0,
        )

    # prepare adjacency: remove self-loops, symmetrize
    adj = _prepare_adjacency(adj, symmetrize=symmetrize, mode=symmetrize_mode)

    # build NetworkX graph
    G = nx.from_scipy_sparse_array(
        adj, create_using=nx.Graph, edge_attribute=WEIGHT_ATTR
    )

    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)

    # compute degree statistics
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    mean_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    # compute community sizes
    unique_labels = np.unique(membership)
    n_communities = len(unique_labels)
    community_sizes = np.array(
        [int(np.sum(membership == lab)) for lab in unique_labels], dtype=np.int64
    )

    # compute modularity
    if n_communities < 2:
        modularity = 0.0
    else:
        communities = [
            set(np.flatnonzero(membership == lab).tolist()) for lab in unique_labels
        ]
        has_weights = adj.nnz > 0 and np.any(adj.data != 1.0)
        modularity = float(
            nx.algorithms.community.modularity(
                G,
                communities,
                weight=WEIGHT_ATTR if has_weights else None,
                resolution=resolution,
            )
        )

    return GraphQuality(
        n_nodes=n,
        n_edges=n_edges,
        n_components=n_components,
        modularity=modularity,
        n_communities=n_communities,
        community_sizes=community_sizes,
        mean_degree=mean_degree,
    )


def _to_csr(adjacency: Adjacency) -> csr_matrix:
    """Convert adjacency to CSR format and validate."""
    adj = sparse.csr_matrix(adjacency)
    shape = adj.shape
    if shape is None or len(shape) != 2:
        raise ValueError("adjacency must be a 2D matrix")
    if shape[0] != shape[1]:
        raise ValueError(f"adjacency must be square, got shape {shape}")
    return adj


def _prepare_adjacency(
    adj: csr_matrix,
    *,
    symmetrize: bool,
    mode: SymmetrizeMode,
) -> csr_matrix:
    """Remove self-loops and optionally symmetrize."""
    # remove self-loops
    diag = adj.diagonal()
    if np.any(diag != 0):
        adj = cast(csr_matrix, (adj - diags(diag, offsets=0, format="csr")).tocsr())
        adj.eliminate_zeros()

    # symmetrize if asymmetric
    if symmetrize:
        diff = adj - adj.T
        diff.eliminate_zeros()
        if diff.nnz > 0:
            if mode == "max":
                adj = cast(csr_matrix, adj.maximum(adj.T).tocsr())
            elif mode == "min":
                adj = cast(csr_matrix, adj.minimum(adj.T).tocsr())
            else:
                adj = cast(csr_matrix, ((adj + adj.T) * 0.5).tocsr())
            adj.eliminate_zeros()

    return adj
