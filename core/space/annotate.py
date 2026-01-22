from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import math

import pandas as pd

from dtypes import EdgeWeights, IgraphGraph


@dataclass(frozen=True, slots=True)
class ClusterLabels:
    """Result of tag clustering via Leiden.

    Attributes
    ----------
    clusters : dict[str, tuple[str, ...]]
        Mapping from cluster ID (C1, C2, ...) to member tags.
    tag_to_cluster : dict[str, str]
        Mapping from tag to its cluster ID.
    n_tags : int
        Total number of tags clustered.
    n_clusters : int
        Number of clusters found.
    modularity : float
        Leiden modularity score.
    """

    clusters: dict[str, tuple[str, ...]]
    tag_to_cluster: dict[str, str]
    n_tags: int
    n_clusters: int
    modularity: float


def extract_top_k_tags(tags: str | None, weights: str | None) -> tuple[str, ...]:
    """Extract tags with maximum weight from comma-separated strings."""
    if tags is None or (isinstance(tags, float) and math.isnan(tags)):
        return ()

    tag_list = [t.strip().lower() for t in str(tags).split(",") if t.strip()]
    if not tag_list:
        return ()

    if weights is None or (isinstance(weights, float) and math.isnan(weights)):
        return _dedupe(tag_list)

    weight_strs = str(weights).split(",")
    weight_list = []
    for w in weight_strs:
        w = w.strip()
        if w:
            try:
                weight_list.append(int(w))
            except ValueError:
                weight_list.append(0)

    if len(weight_list) != len(tag_list):
        return _dedupe(tag_list)

    max_weight = max(weight_list)
    top_tags = [tag_list[i] for i, w in enumerate(weight_list) if w == max_weight]
    return _dedupe(top_tags)


def build_top_k_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add top_k_* columns for genre, theme, mood, and style."""
    result = df.copy()

    mappings = [
        ("genre", "genre_weights", "top_k_genre"),
        ("theme", "theme_weights", "top_k_theme"),
        ("mood_all", "mood_all_weights", "top_k_mood"),
        ("style", "style_weights", "top_k_style"),
    ]

    for tag_col, weight_col, out_col in mappings:
        if tag_col not in result.columns:
            result[out_col] = [() for _ in range(len(result))]
            continue

        weight_series = result.get(weight_col)
        result[out_col] = [
            extract_top_k_tags(
                row[tag_col],
                row.get(weight_col) if weight_series is not None else None,
            )
            for _, row in result.iterrows()
        ]

    return result


def compute_npmi(n_i: int, n_j: int, n_ij: int, n_total: int) -> float:
    """Compute normalized pointwise mutual information."""
    if n_total == 0 or n_i == 0 or n_j == 0 or n_ij == 0:
        return 0.0

    p_i = n_i / n_total
    p_j = n_j / n_total
    p_ij = n_ij / n_total

    pmi = math.log(p_ij / (p_i * p_j))
    npmi = pmi / -math.log(p_ij)
    return npmi


def build_tag_graph(
    df: pd.DataFrame,
    tag_col: str,
    min_cooccurrence: int = 2,
    min_tag_count: int = 3,
) -> IgraphGraph:
    """Build an NPMI-weighted tag co-occurrence graph."""
    try:
        import igraph as ig
    except ImportError as e:
        raise ImportError(
            "Missing optional dependency 'igraph'. Install with: pip install python-igraph"
        ) from e

    if tag_col not in df.columns:
        raise ValueError(f"dataframe must have '{tag_col}' column")

    n_total = len(df)

    tag_counts: Counter[str] = Counter()
    for tags in df[tag_col]:
        for t in tags:
            tag_counts[t] += 1

    valid_tags = {t for t, c in tag_counts.items() if c >= min_tag_count}

    pair_counts: Counter[tuple[str, str]] = Counter()
    for tags in df[tag_col]:
        filtered = [t for t in tags if t in valid_tags]
        for i, j in combinations(sorted(set(filtered)), 2):
            pair_counts[(i, j)] += 1

    edges: list[tuple[str, str]] = []
    weights: list[float] = []

    for (t_i, t_j), n_ij in pair_counts.items():
        if n_ij < min_cooccurrence:
            continue

        npmi = compute_npmi(tag_counts[t_i], tag_counts[t_j], n_ij, n_total)
        if npmi > 0:
            edges.append((t_i, t_j))
            weights.append(npmi)

    tag_list = sorted(valid_tags)
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}

    g = ig.Graph(n=len(tag_list), directed=False)
    g.vs["name"] = tag_list

    if edges:
        edge_indices = [(tag_to_idx[a], tag_to_idx[b]) for a, b in edges]
        g.add_edges(edge_indices)
        g.es["weight"] = weights

    return g


def prune_graph_topk(
    graph: IgraphGraph,
    k: int = 10,
) -> IgraphGraph:
    """
    Sparsify graph via top-k per-node pruning to reduce hub effects.

    For each node, keeps only its k highest-NPMI edges. Symmetrizes by
    keeping an edge if either endpoint retained it. This bounds hub degree
    while preserving at least one edge for rare nodes (their best connection).
    """
    try:
        import igraph as ig
    except ImportError as e:
        raise ImportError(
            "Missing optional dependency 'igraph'. Install with: pip install python-igraph"
        ) from e

    if graph.ecount() == 0:
        return graph.copy()

    n_vertices = graph.vcount()
    tag_names = graph.vs["name"]

    keep_edges: set[int] = set()

    for v_idx in range(n_vertices):
        incident = graph.incident(v_idx, mode="all")
        if not incident:
            continue

        edge_weights = [(eid, graph.es[eid]["weight"]) for eid in incident]

        edge_weights.sort(key=lambda x: x[1], reverse=True)
        for eid, _ in edge_weights[:k]:
            keep_edges.add(eid)

    new_graph = ig.Graph(n=n_vertices, directed=False)
    new_graph.vs["name"] = tag_names

    if keep_edges:
        surviving_edges = []
        surviving_weights = []
        for eid in keep_edges:
            edge = graph.es[eid]
            surviving_edges.append((edge.source, edge.target))
            surviving_weights.append(edge["weight"])

        new_graph.add_edges(surviving_edges)
        new_graph.es["weight"] = surviving_weights

    return new_graph


def cluster_tags(
    graph: IgraphGraph,
    resolution: float = 1.0,
    seed: int | None = None,
    prefix: str = "C",
) -> ClusterLabels:
    """Cluster tags using Leiden community detection."""
    try:
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "Missing optional dependency 'leidenalg'. Install with: pip install leidenalg"
        ) from e

    n_tags = graph.vcount()
    if n_tags == 0:
        return ClusterLabels(
            clusters={},
            tag_to_cluster={},
            n_tags=0,
            n_clusters=0,
            modularity=0.0,
        )

    weights = graph.es["weight"] if graph.ecount() > 0 else None

    if seed is not None:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=seed,
        )
    else:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
        )

    membership = partition.membership
    modularity = partition.modularity if partition.modularity is not None else 0.0

    cluster_to_tags: dict[int, list[str]] = {}
    tag_names = graph.vs["name"]

    for idx, cluster_id in enumerate(membership):
        if cluster_id not in cluster_to_tags:
            cluster_to_tags[cluster_id] = []
        cluster_to_tags[cluster_id].append(tag_names[idx])

    sorted_clusters = sorted(cluster_to_tags.items(), key=lambda x: (-len(x[1]), x[0]))

    clusters: dict[str, tuple[str, ...]] = {}
    tag_to_cluster: dict[str, str] = {}

    for i, (_, tags) in enumerate(sorted_clusters, start=1):
        cluster_id = f"{prefix}{i}"
        sorted_tags = tuple(sorted(tags))
        clusters[cluster_id] = sorted_tags
        for t in sorted_tags:
            tag_to_cluster[t] = cluster_id

    return ClusterLabels(
        clusters=clusters,
        tag_to_cluster=tag_to_cluster,
        n_tags=n_tags,
        n_clusters=len(clusters),
        modularity=float(modularity),
    )


def merge_clusters(
    graph: IgraphGraph,
    labels: ClusterLabels,
    target_k: int = 7,
    min_similarity: float = 0.0,
    prefix: str = "T",
) -> ClusterLabels:
    """
    Merge clusters via average-link agglomeration until target_k remain.

    Uses NPMI edge weights from the graph to compute average inter-cluster
    similarity. At each step, merges the pair with highest average NPMI.
    Stops early if no pair exceeds min_similarity.
    """
    if labels.n_clusters <= target_k:
        return labels

    tag_names = graph.vs["name"]

    edge_weights: EdgeWeights = {}
    for edge in graph.es:
        src, tgt = edge.source, edge.target
        tag_a, tag_b = tag_names[src], tag_names[tgt]
        weight = edge["weight"]
        edge_weights[(tag_a, tag_b)] = weight
        edge_weights[(tag_b, tag_a)] = weight

    cluster_sets: dict[int, set[str]] = {}
    for i, (_, tags) in enumerate(labels.clusters.items()):
        cluster_sets[i] = set(tags)

    def compute_similarity(c_a: int, c_b: int) -> float:
        """Compute average NPMI between two clusters."""
        tags_a = cluster_sets[c_a]
        tags_b = cluster_sets[c_b]

        total_npmi = 0.0
        edge_count = 0

        for tag_a in tags_a:
            for tag_b in tags_b:
                if (tag_a, tag_b) in edge_weights:
                    total_npmi += edge_weights[(tag_a, tag_b)]
                    edge_count += 1

        if edge_count == 0:
            return float("-inf")
        return total_npmi / edge_count

    cluster_ids = list(cluster_sets.keys())
    similarities: dict[tuple[int, int], float] = {}

    for i, c_a in enumerate(cluster_ids):
        for c_b in cluster_ids[i + 1 :]:
            sim = compute_similarity(c_a, c_b)
            similarities[(c_a, c_b)] = sim

    while len(cluster_sets) > target_k:
        best_pair = None
        best_sim = float("-inf")

        for (c_a, c_b), sim in similarities.items():
            if c_a not in cluster_sets or c_b not in cluster_sets:
                continue
            if sim > best_sim:
                best_sim = sim
                best_pair = (c_a, c_b)

        if best_pair is None or best_sim < min_similarity:
            break

        c_a, c_b = best_pair

        cluster_sets[c_a] = cluster_sets[c_a] | cluster_sets[c_b]
        del cluster_sets[c_b]

        remaining = [c for c in cluster_sets.keys() if c != c_a]
        for c_other in remaining:
            sim = compute_similarity(c_a, c_other)
            key = (min(c_a, c_other), max(c_a, c_other))
            similarities[key] = sim

    if len(cluster_sets) > target_k:
        orphan_ids = [c for c in cluster_sets if len(cluster_sets[c]) == 1]
        non_orphan_ids = [c for c in cluster_sets if len(cluster_sets[c]) > 1]

        for orphan_id in orphan_ids:
            if len(cluster_sets) <= target_k:
                break
            if orphan_id not in cluster_sets:
                continue

            orphan_tag = next(iter(cluster_sets[orphan_id]))

            best_cluster = None
            best_score = float("-inf")

            for c_id in non_orphan_ids:
                if c_id not in cluster_sets:
                    continue

                scores = [
                    edge_weights.get((orphan_tag, t), 0.0) for t in cluster_sets[c_id]
                ]
                top_scores = sorted(scores, reverse=True)[:3]
                avg = sum(top_scores) / len(top_scores) if top_scores else float("-inf")

                if avg > best_score:
                    best_score = avg
                    best_cluster = c_id

            if best_cluster is not None:
                cluster_sets[best_cluster] = (
                    cluster_sets[best_cluster] | cluster_sets[orphan_id]
                )
                del cluster_sets[orphan_id]

    sorted_clusters = sorted(cluster_sets.items(), key=lambda x: (-len(x[1]), x[0]))

    clusters: dict[str, tuple[str, ...]] = {}
    tag_to_cluster: dict[str, str] = {}

    for i, (_, tags) in enumerate(sorted_clusters, start=1):
        cluster_id = f"{prefix}{i}"
        sorted_tags = tuple(sorted(tags))
        clusters[cluster_id] = sorted_tags
        for t in sorted_tags:
            tag_to_cluster[t] = cluster_id

    return ClusterLabels(
        clusters=clusters,
        tag_to_cluster=tag_to_cluster,
        n_tags=labels.n_tags,
        n_clusters=len(clusters),
        modularity=labels.modularity,
    )


def write_cluster_labels(result: ClusterLabels, output_path: Path) -> None:
    """Write cluster labels to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {k: list(v) for k, v in result.clusters.items()}

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
