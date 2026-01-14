from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
import math
from typing import Literal, TypeAlias

from .filter import TagGroup, Track

TagCounts: TypeAlias = Counter[str]
PairCounts: TypeAlias = Counter[tuple[str, str]]
PairSort: TypeAlias = Literal["count", "npmi"]


@dataclass(frozen=True, slots=True)
class TagMetric:
    tag: str
    presence_count: int
    p_track: float


@dataclass(frozen=True, slots=True)
class PairMetric:
    tag_a: str
    tag_b: str
    cooccur_count: int
    p_xy: float
    npmi: float


@dataclass(frozen=True, slots=True)
class GroupMetrics:
    group: TagGroup
    n_tracks: int
    n_unique_tags: int
    gini: float
    entropy: float
    entropy_norm: float
    imbalance_ratio: float
    tags: list[TagMetric]
    cooccurring_pairs: list[PairMetric]
    npmi_pairs: list[PairMetric]


def compute_metrics(
    tracks: Sequence[Track],
    *,
    groups: Sequence[TagGroup] = ("mood", "genre", "instrument"),
    top_k_pairs: int = 200,
    min_pair_count: int = 2,
) -> dict[TagGroup, GroupMetrics]:
    """Compute tag distribution and co-occurrence metrics for each group."""
    out: dict[TagGroup, GroupMetrics] = {}
    for group in groups:
        out[group] = compute_group_metrics(
            tracks,
            group=group,
            top_k_pairs=top_k_pairs,
            min_pair_count=min_pair_count,
        )
    return out


def compute_group_metrics(
    tracks: Sequence[Track],
    *,
    group: TagGroup,
    top_k_pairs: int = 200,
    min_pair_count: int = 2,
) -> GroupMetrics:
    """Compute metrics for single tag group."""
    n_tracks = len(tracks)
    
    tag_counts = count_tag_presence(tracks, group=group)
    tag_list = tag_distribution(tag_counts=tag_counts, n_tracks=n_tracks)
    values = [m.presence_count for m in tag_list]
    
    gini = gini_coefficient(values)
    entropy = shannon_entropy(values)
    entropy_norm = normalized_entropy(values)
    imbalance_ratio = imbalance_max_min(values)
    
    pair_counts = count_pair_cooccurrence(tracks, group=group)
    
    pairs_by_count = cooccurring_pairs(
        tag_counts=tag_counts,
        pair_counts=pair_counts,
        n_tracks=n_tracks,
        top_k=top_k_pairs,
        min_pair_count=min_pair_count,
        sort_by="count",
    )
    
    pairs_by_npmi = cooccurring_pairs(
        tag_counts=tag_counts,
        pair_counts=pair_counts,
        n_tracks=n_tracks,
        top_k=top_k_pairs,
        min_pair_count=min_pair_count,
        sort_by="npmi",
    )
    
    return GroupMetrics(
        group=group,
        n_tracks=n_tracks,
        n_unique_tags=len(tag_counts),
        gini=gini,
        entropy=entropy,
        entropy_norm=entropy_norm,
        imbalance_ratio=imbalance_ratio,
        tags=tag_list,
        cooccurring_pairs=pairs_by_count,
        npmi_pairs=pairs_by_npmi,
    )


def count_tag_presence(
    tracks: Sequence[Track],
    *,
    group: TagGroup,
) -> TagCounts:
    """Count unique tag presence per track (not total occurrences)."""
    counts: TagCounts = Counter()
    for track in tracks:
        counts.update(set(track.tags(group)))
    return counts


def tag_distribution(
    *,
    tag_counts: Mapping[str, int],
    n_tracks: int,
) -> list[TagMetric]:
    """Convert tag counts to metrics, sorted desc by count then asc by name."""
    out: list[TagMetric] = []
    for tag, c in tag_counts.items():
        p_track = float(c) / float(n_tracks) if n_tracks > 0 else 0.0
        out.append(TagMetric(tag=tag, presence_count=c, p_track=p_track))
    
    out.sort(key=lambda x: (-x.presence_count, x.tag))
    return out


def count_pair_cooccurrence(
    tracks: Sequence[Track],
    *,
    group: TagGroup,
) -> PairCounts:
    """Count co-occurrence of tag pairs within same track."""
    counts: PairCounts = Counter()
    for track in tracks:
        tags = sorted(set(track.tags(group)))
        if len(tags) < 2:
            continue
        for a, b in itertools.combinations(tags, 2):
            counts[(a, b)] += 1
    return counts


def cooccurring_pairs(
    *,
    tag_counts: Mapping[str, int],
    pair_counts: Mapping[tuple[str, str], int],
    n_tracks: int,
    top_k: int,
    min_pair_count: int,
    sort_by: PairSort,
) -> list[PairMetric]:
    """Compute pair metrics with NPMI scores."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if min_pair_count <= 0:
        raise ValueError("min_pair_count must be > 0")
    
    if n_tracks == 0:
        return []
    
    out: list[PairMetric] = []
    
    for (a, b), c_xy in pair_counts.items():
        if c_xy < min_pair_count:
            continue
        
        c_a = tag_counts.get(a, 0)
        c_b = tag_counts.get(b, 0)
        
        if c_a <= 0 or c_b <= 0:
            continue
        
        p_x = float(c_a) / float(n_tracks)
        p_y = float(c_b) / float(n_tracks)
        p_xy = float(c_xy) / float(n_tracks)
        
        npmi = normalized_pmi(p_x=p_x, p_y=p_y, p_xy=p_xy, eps=1e-12)
        
        if math.isnan(npmi):
            continue
        
        out.append(
            PairMetric(
                tag_a=a,
                tag_b=b,
                cooccur_count=c_xy,
                p_xy=p_xy,
                npmi=npmi,
            )
        )
    
    if sort_by == "count":
        out.sort(key=lambda x: (-x.cooccur_count, x.tag_a, x.tag_b))
    else:
        out.sort(key=lambda x: (-x.npmi, -x.cooccur_count, x.tag_a, x.tag_b))
    
    return out[:top_k]


def normalized_pmi(
    *,
    p_x: float,
    p_y: float,
    p_xy: float,
    eps: float,
) -> float:
    """
    Normalized pointwise mutual information (NPMI).
    
    Measures association strength between two tags, normalized to [-1, 1].
    Returns NaN for invalid probability values.
    """
    if eps < 0.0:
        raise ValueError("eps must be >= 0")
    
    if p_x <= 0.0 or p_y <= 0.0 or p_xy <= 0.0:
        return math.nan
    if p_xy - p_x > eps or p_xy - p_y > eps:
        return math.nan
    if p_x - 1.0 > eps or p_y - 1.0 > eps or p_xy - 1.0 > eps:
        return math.nan
    
    denom = -math.log(p_xy)
    if denom <= 0.0:
        return math.nan
    
    pmi = math.log(p_xy / (p_x * p_y))
    return pmi / denom


def shannon_entropy(values: Sequence[int]) -> float:
    """H(X) = -sum(p_i * log(p_i))."""
    total = sum(values)
    if total <= 0:
        return 0.0
    
    h = 0.0
    for v in values:
        if v <= 0:
            continue
        p = float(v) / float(total)
        h -= p * math.log(p)
    
    return h


def normalized_entropy(values: Sequence[int]) -> float:
    """Normalize entropy by max possible entropy (log k)."""
    k = len([v for v in values if v > 0])
    if k <= 1:
        return 0.0
    
    h = shannon_entropy(values)
    return h / math.log(float(k))


def gini_coefficient(values: Sequence[int]) -> float:
    """Gini coefficient measures inequality in distribution."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    if n == 0:
        return 0.0
    
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    
    cumsum = 0.0
    gini_sum = 0.0
    
    for i, val in enumerate(sorted_vals):
        cumsum += val
        gini_sum += (2 * (i + 1) - n - 1) * val
    
    return gini_sum / (n * total)


def imbalance_max_min(values: Sequence[int]) -> float:
    """Ratio of max to min tag count."""
    positive_vals = [v for v in values if v > 0]
    if len(positive_vals) == 0:
        return 0.0
    
    max_val = max(positive_vals)
    min_val = min(positive_vals)
    
    if min_val == 0:
        return float("inf")
    
    return float(max_val) / float(min_val)
