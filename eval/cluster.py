from dataclasses import dataclass, field
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)

FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]
MeanByLabel: TypeAlias = dict[int, float]
Diagnostics: TypeAlias = dict[str, float]


@dataclass(frozen=True, slots=True)
class ClusterQuality:
    """Internal cluster validity evidence for a fixed labeling."""

    silhouette_mean: float
    silhouette_std: float
    silhouette_by_label: MeanByLabel

    davies_bouldin: float
    calinski_harabasz: float
    dbcv: float

    n_samples: int
    n_scored: int
    n_clusters: int
    n_noise: int
    noise_fraction: float
    cluster_sizes: IntArray

    diagnostics: Diagnostics = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def score_cluster_quality(
    X: FloatArray,
    labels: IntArray,
    metric: str = "euclidean",
    noise_label: int = -1,
) -> ClusterQuality:
    """
    Compute internal validation metrics for a fixed clustering assignment.

    Args:
        X (FloatArray)     : data matrix of shape (n_samples, n_dims).
        labels (IntArray)  : integer labels of shape (n_samples,).
        metric (str)       : distance metric for silhouette ("euclidean", "cosine", ...).
        noise_label (int)  : label used for noise points. Default is -1.

    Returns:
        ClusterQuality : metrics, size diagnostics, and anomaly flags.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    n_samples = int(X.shape[0])
    if int(labels.shape[0]) != n_samples:
        raise ValueError(
            f"shape mismatch: X has {n_samples} rows, labels has {labels.shape[0]} entries"
        )

    anomalies: list[str] = []
    empty_sizes: IntArray = np.array([], dtype=np.int64)

    if n_samples < 2:
        anomalies.append("insufficient samples (n < 2)")
        diagnostics = _size_diagnostics(empty_sizes)
        diagnostics.update(
            {
                "n_samples": float(n_samples),
                "n_scored": 0.0,
                "n_clusters": 0.0,
                "n_noise": 0.0,
                "noise_fraction": 0.0,
            }
        )
        return ClusterQuality(
            silhouette_mean=0.0,
            silhouette_std=0.0,
            silhouette_by_label={},
            davies_bouldin=0.0,
            calinski_harabasz=0.0,
            dbcv=0.0,
            n_samples=n_samples,
            n_scored=0,
            n_clusters=0,
            n_noise=0,
            noise_fraction=0.0,
            cluster_sizes=empty_sizes,
            diagnostics=diagnostics,
            anomalies=anomalies,
        )

    non_noise = labels != noise_label
    n_noise = int(np.sum(~non_noise))
    noise_fraction = float(n_noise / n_samples)

    n_scored_raw = int(np.sum(non_noise))
    if n_scored_raw < 2:
        anomalies.append("no non-noise points available for scoring")
        diagnostics = _size_diagnostics(empty_sizes)
        diagnostics.update(
            {
                "n_samples": float(n_samples),
                "n_scored": float(n_scored_raw),
                "n_clusters": 0.0,
                "n_noise": float(n_noise),
                "noise_fraction": float(noise_fraction),
            }
        )
        return ClusterQuality(
            silhouette_mean=0.0,
            silhouette_std=0.0,
            silhouette_by_label={},
            davies_bouldin=0.0,
            calinski_harabasz=0.0,
            dbcv=0.0,
            n_samples=n_samples,
            n_scored=n_scored_raw,
            n_clusters=0,
            n_noise=n_noise,
            noise_fraction=noise_fraction,
            cluster_sizes=empty_sizes,
            diagnostics=diagnostics,
            anomalies=anomalies,
        )

    X_scored = cast(FloatArray, X[non_noise])
    labels_scored = cast(IntArray, labels[non_noise].astype(np.int64, copy=False))

    unique_labels, cluster_sizes = _cluster_sizes(labels_scored)
    n_scored = int(X_scored.shape[0])
    n_clusters = int(unique_labels.size)

    if n_clusters < 2:
        anomalies.append(
            "fewer than 2 clusters after excluding noise; metrics undefined"
        )
        diagnostics = _size_diagnostics(cluster_sizes)
        diagnostics.update(
            {
                "n_samples": float(n_samples),
                "n_scored": float(n_scored),
                "n_clusters": float(n_clusters),
                "n_noise": float(n_noise),
                "noise_fraction": float(noise_fraction),
            }
        )
        return ClusterQuality(
            silhouette_mean=0.0,
            silhouette_std=0.0,
            silhouette_by_label={},
            davies_bouldin=0.0,
            calinski_harabasz=0.0,
            dbcv=0.0,
            n_samples=n_samples,
            n_scored=n_scored,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_fraction=noise_fraction,
            cluster_sizes=cluster_sizes,
            diagnostics=diagnostics,
            anomalies=anomalies,
        )

    if n_scored <= n_clusters:
        anomalies.append(
            "n_scored must exceed n_clusters for silhouette/CH; metrics undefined"
        )
        diagnostics = _size_diagnostics(cluster_sizes)
        diagnostics.update(
            {
                "n_samples": float(n_samples),
                "n_scored": float(n_scored),
                "n_clusters": float(n_clusters),
                "n_noise": float(n_noise),
                "noise_fraction": float(noise_fraction),
            }
        )
        return ClusterQuality(
            silhouette_mean=0.0,
            silhouette_std=0.0,
            silhouette_by_label={},
            davies_bouldin=0.0,
            calinski_harabasz=0.0,
            dbcv=0.0,
            n_samples=n_samples,
            n_scored=n_scored,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_fraction=noise_fraction,
            cluster_sizes=cluster_sizes,
            diagnostics=diagnostics,
            anomalies=anomalies,
        )

    if int(np.min(cluster_sizes)) < 2:
        anomalies.append(
            "at least one cluster has size < 2; silhouette may be unstable"
        )

    silhouette_mean, silhouette_std, silhouette_by_label = _silhouette_metrics(
        X_scored,
        labels_scored,
        metric=metric,
        anomalies=anomalies,
    )
    davies_bouldin, calinski_harabasz = _dbi_ch_metrics(
        X_scored, labels_scored, anomalies=anomalies
    )
    dbcv = _dbcv_metric(X_scored, labels_scored, anomalies=anomalies)

    diagnostics = _size_diagnostics(cluster_sizes)
    diagnostics.update(
        {
            "n_samples": float(n_samples),
            "n_scored": float(n_scored),
            "n_clusters": float(n_clusters),
            "n_noise": float(n_noise),
            "noise_fraction": float(noise_fraction),
        }
    )

    return ClusterQuality(
        silhouette_mean=float(silhouette_mean),
        silhouette_std=float(silhouette_std),
        silhouette_by_label=silhouette_by_label,
        davies_bouldin=float(davies_bouldin),
        calinski_harabasz=float(calinski_harabasz),
        dbcv=float(dbcv),
        n_samples=n_samples,
        n_scored=n_scored,
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_fraction=noise_fraction,
        cluster_sizes=cluster_sizes,
        diagnostics=diagnostics,
        anomalies=anomalies,
    )


def _cluster_sizes(labels_scored: IntArray) -> tuple[IntArray, IntArray]:
    unique_labels, counts = np.unique(labels_scored, return_counts=True)
    order = np.argsort(unique_labels, kind="mergesort")
    return (
        cast(IntArray, unique_labels[order].astype(np.int64, copy=False)),
        cast(IntArray, counts[order].astype(np.int64, copy=False)),
    )


def _silhouette_metrics(
    X: FloatArray,
    labels: IntArray,
    metric: str,
    anomalies: list[str],
) -> tuple[float, float, MeanByLabel]:
    try:
        raw = silhouette_samples(X, labels, metric=metric)
        per_point = np.asarray(raw, dtype=np.float64)
    except Exception as e:
        anomalies.append(f"silhouette computation failed ({type(e).__name__})")
        return 0.0, 0.0, {}

    mean_val = float(np.mean(per_point))
    std_val = float(np.std(per_point))

    by_label: MeanByLabel = {}
    for lab in np.unique(labels):
        mask = labels == lab
        by_label[int(lab)] = (
            float(np.mean(per_point[mask])) if bool(np.any(mask)) else 0.0
        )

    return mean_val, std_val, by_label


def _dbi_ch_metrics(
    X: FloatArray, labels: IntArray, anomalies: list[str]
) -> tuple[float, float]:
    dbi = 0.0
    chi = 0.0

    try:
        dbi = float(davies_bouldin_score(X, labels))
    except Exception as e:
        anomalies.append(f"davies-bouldin computation failed ({type(e).__name__})")

    try:
        chi = float(calinski_harabasz_score(X, labels))
    except Exception as e:
        anomalies.append(f"calinski-harabasz computation failed ({type(e).__name__})")

    return dbi, chi


def _coerce_float(value: object, anomalies: list[str], name: str) -> float:
    if isinstance(value, Real):
        return float(value)

    if isinstance(value, np.generic):
        return float(value.item())

    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])

    anomalies.append(f"{name} returned non-scalar value")
    return 0.0


def _dbcv_metric(X: FloatArray, labels: IntArray, anomalies: list[str]) -> float:
    try:
        from hdbscan.validity import validity_index
    except Exception:
        anomalies.append(
            "dbcv unavailable (failed to import hdbscan.validity.validity_index)"
        )
        return 0.0

    try:
        out: object = validity_index(X, labels)
        if isinstance(out, tuple):
            return _coerce_float(out[0], anomalies, "dbcv")
        return _coerce_float(out, anomalies, "dbcv")
    except TypeError:
        try:
            out2: object = validity_index(X, labels, metric="euclidean")
            if isinstance(out2, tuple):
                return _coerce_float(out2[0], anomalies, "dbcv")
            return _coerce_float(out2, anomalies, "dbcv")
        except Exception as e:
            anomalies.append(f"dbcv computation failed ({type(e).__name__})")
            return 0.0
    except Exception as e:
        anomalies.append(f"dbcv computation failed ({type(e).__name__})")
        return 0.0


def _size_diagnostics(cluster_sizes: IntArray) -> Diagnostics:
    if cluster_sizes.size == 0:
        return {
            "min_cluster_size": 0.0,
            "max_cluster_size": 0.0,
            "mean_cluster_size": 0.0,
            "median_cluster_size": 0.0,
            "std_cluster_size": 0.0,
            "cluster_size_cv": 0.0,
        }

    sizes = np.asarray(cluster_sizes, dtype=np.float64)
    mean_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))
    cv = float(std_size / mean_size) if mean_size > 0.0 else 0.0

    return {
        "min_cluster_size": float(np.min(sizes)),
        "max_cluster_size": float(np.max(sizes)),
        "mean_cluster_size": mean_size,
        "median_cluster_size": float(np.median(sizes)),
        "std_cluster_size": std_size,
        "cluster_size_cv": cv,
    }
