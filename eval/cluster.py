from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score

from dtypes import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class ClusterQuality:
    """
    Geometric cluster quality metrics.

    Note: these metrics assume geometric clustering (k-means, HDBSCAN) but
    communities typically come from graph partitioning (Leiden). High modularity
    does not imply high silhouette. This is secondary to semantic coherence.

    Args
    ----
        n_clusters        (int) : number of unique cluster labels.
        cluster_sizes (IntArray) : size of each cluster, sorted by label.
        silhouette_mean (float) : mean silhouette coefficient across samples.
    """

    n_clusters: int
    cluster_sizes: IntArray
    silhouette_mean: float

    @property
    def size_balance(self) -> float:
        """
        Coefficient of variation of cluster sizes.

        Lower values indicate more balanced cluster sizes. Returns NaN if
        no clusters exist or mean size is zero.
        """
        if len(self.cluster_sizes) == 0:
            return float("nan")
        mean_size = float(np.mean(self.cluster_sizes))
        if mean_size == 0:
            return float("nan")
        return float(np.std(self.cluster_sizes) / mean_size)


def score_cluster_quality(
    embeddings: FloatArray,
    membership: IntArray,
    metric: str = "cosine",
) -> ClusterQuality:
    """
    Compute geometric cluster quality metrics.

    Args
    ----
        embeddings (FloatArray) : embedding matrix of shape (n_samples, n_dims).
        membership    (IntArray) : cluster labels of shape (n_samples,).
        metric           (str)  : distance metric for silhouette.
                                  (Default is "cosine").

    Returns
    -------
        (ClusterQuality) : cluster count, sizes, and silhouette score.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if membership.ndim != 1:
        raise ValueError(f"membership must be 1D, got shape {membership.shape}")
    if len(embeddings) != len(membership):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) must match membership length ({len(membership)})"
        )

    n_samples = len(embeddings)
    if n_samples == 0:
        return ClusterQuality(
            n_clusters=0,
            cluster_sizes=np.array([], dtype=np.int64),
            silhouette_mean=float("nan"),
        )

    unique_labels = np.unique(membership)
    n_clusters = len(unique_labels)

    # compute cluster sizes sorted by label
    cluster_sizes = np.array(
        [int(np.sum(membership == lab)) for lab in unique_labels], dtype=np.int64
    )

    # silhouette requires at least 2 clusters and more samples than clusters
    if n_clusters < 2 or n_clusters >= n_samples:
        return ClusterQuality(
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            silhouette_mean=float("nan"),
        )

    try:
        sil = float(silhouette_score(embeddings, membership, metric=metric))
    except Exception:
        sil = float("nan")

    return ClusterQuality(
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        silhouette_mean=sil,
    )
