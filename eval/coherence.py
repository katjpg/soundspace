from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dataset.filter import Track
from dtypes import IntArray


@dataclass(frozen=True, slots=True)
class LabelCoherence:
    """Semantic coherence metrics for a single community/cluster."""

    label: int
    size: int
    style_entropy: float
    theme_entropy: float
    mood_entropy: float
    va_spread: float
    quadrant_coverage: float


@dataclass(frozen=True, slots=True)
class SemanticQuality:
    """
    Aggregate semantic quality across all communities.

    Args
    ----
        n_labels              (int) : number of unique community labels.
        n_samples             (int) : total number of tracks.
        mean_style_entropy  (float) : mean normalized entropy of style tags (lower = coherent).
        mean_theme_entropy  (float) : mean normalized entropy of theme tags (lower = coherent).
        mean_mood_entropy   (float) : mean normalized entropy of mood tags (higher = diverse).
        mean_va_spread      (float) : mean std of V-A distances from centroid (higher = diverse).
        mean_quadrant_coverage (float) : mean fraction of V-A quadrants covered (higher = diverse).
        labels (list[LabelCoherence]) : per-label breakdown.
    """

    n_labels: int
    n_samples: int
    mean_style_entropy: float
    mean_theme_entropy: float
    mean_mood_entropy: float
    mean_va_spread: float
    mean_quadrant_coverage: float
    labels: list[LabelCoherence]


def score_semantic_quality(
    tracks: Sequence[Track],
    membership: IntArray,
) -> SemanticQuality:
    """Compute semantic coherence metrics for community assignments."""
    if len(tracks) != len(membership):
        raise ValueError(
            f"tracks length ({len(tracks)}) must match membership length ({len(membership)})"
        )

    n_samples = len(tracks)
    if n_samples == 0:
        return SemanticQuality(
            n_labels=0,
            n_samples=0,
            mean_style_entropy=0.0,
            mean_theme_entropy=0.0,
            mean_mood_entropy=0.0,
            mean_va_spread=0.0,
            mean_quadrant_coverage=0.0,
            labels=[],
        )

    unique_labels = np.unique(membership)
    label_results: list[LabelCoherence] = []

    for lab in unique_labels:
        mask = membership == lab
        community_tracks = [t for t, m in zip(tracks, mask) if m]

        if len(community_tracks) == 0:
            continue

        style_tags = [t.style_tags for t in community_tracks]
        theme_tags = [t.theme_tags for t in community_tracks]
        mood_tags = [t.mood_tags for t in community_tracks]
        arousals = [t.arousal for t in community_tracks]
        valences = [t.valence for t in community_tracks]

        label_results.append(
            LabelCoherence(
                label=int(lab),
                size=len(community_tracks),
                style_entropy=_tag_entropy(style_tags),
                theme_entropy=_tag_entropy(theme_tags),
                mood_entropy=_tag_entropy(mood_tags),
                va_spread=_va_spread(arousals, valences),
                quadrant_coverage=_quadrant_coverage(arousals, valences),
            )
        )

    n_labels = len(label_results)
    if n_labels == 0:
        return SemanticQuality(
            n_labels=0,
            n_samples=n_samples,
            mean_style_entropy=0.0,
            mean_theme_entropy=0.0,
            mean_mood_entropy=0.0,
            mean_va_spread=0.0,
            mean_quadrant_coverage=0.0,
            labels=[],
        )

    return SemanticQuality(
        n_labels=n_labels,
        n_samples=n_samples,
        mean_style_entropy=float(np.mean([lc.style_entropy for lc in label_results])),
        mean_theme_entropy=float(np.mean([lc.theme_entropy for lc in label_results])),
        mean_mood_entropy=float(np.mean([lc.mood_entropy for lc in label_results])),
        mean_va_spread=float(np.mean([lc.va_spread for lc in label_results])),
        mean_quadrant_coverage=float(
            np.mean([lc.quadrant_coverage for lc in label_results])
        ),
        labels=label_results,
    )


def _tag_entropy(tag_lists: Sequence[tuple[str, ...]]) -> float:
    """
    Compute normalized Shannon entropy over tag distribution.

    Each track contributes all its tags to the distribution. Normalization
    by log(n_unique) bounds entropy in [0, 1] where 0 means all tracks share
    the same tag and 1 means uniform distribution.
    """
    counts = _tag_distribution(tag_lists)
    if len(counts) == 0:
        return 0.0
    return _normalized_entropy(list(counts.values()))


def _tag_distribution(tag_lists: Sequence[tuple[str, ...]]) -> dict[str, int]:
    """Count occurrences of each tag across all tracks."""
    counter: Counter[str] = Counter()
    for tags in tag_lists:
        counter.update(tags)
    return dict(counter)


def _normalized_entropy(counts: Sequence[int]) -> float:
    """
    Compute Shannon entropy normalized by log(n_unique).

    Returns 0 if there's only one unique value or all counts are zero.
    """
    if len(counts) == 0:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    n_unique = len([c for c in counts if c > 0])
    if n_unique <= 1:
        return 0.0

    probs = np.array([c / total for c in counts if c > 0], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(n_unique))

    return entropy / max_entropy if max_entropy > 0 else 0.0


def _va_spread(arousals: Sequence[float], valences: Sequence[float]) -> float:
    """
    Compute standard deviation of Euclidean distances from V-A centroid.

    Measures how spread out tracks are in the valence-arousal space within
    a community. Higher values indicate more emotional diversity.
    """
    if len(arousals) < 2:
        return 0.0

    a = np.array(arousals, dtype=np.float64)
    v = np.array(valences, dtype=np.float64)

    centroid_a = float(np.mean(a))
    centroid_v = float(np.mean(v))

    distances = np.sqrt((a - centroid_a) ** 2 + (v - centroid_v) ** 2)
    return float(np.std(distances))


def _quadrant_coverage(arousals: Sequence[float], valences: Sequence[float]) -> float:
    """
    Compute fraction of V-A quadrants covered by tracks.

    Quadrants are defined by the origin (0, 0):
    - Q1: high arousal, high valence (excited, happy)
    - Q2: high arousal, low valence (tense, angry)
    - Q3: low arousal, low valence (sad, depressed)
    - Q4: low arousal, high valence (calm, relaxed)

    Returns fraction in [0, 1] where 1 means all 4 quadrants are represented.
    """
    if len(arousals) == 0:
        return 0.0

    quadrants_present: set[int] = set()

    for a, v in zip(arousals, valences):
        if a >= 0 and v >= 0:
            quadrants_present.add(1)
        elif a >= 0 and v < 0:
            quadrants_present.add(2)
        elif a < 0 and v < 0:
            quadrants_present.add(3)
        else:
            quadrants_present.add(4)

    return len(quadrants_present) / 4.0
