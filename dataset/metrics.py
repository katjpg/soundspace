from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
import math
from typing import Literal, TypeAlias

from .filter import ALL_GROUPS, TagGroup, TrackRow


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
    rows: Sequence[TrackRow],
    *,
    groups: Sequence[TagGroup] = ALL_GROUPS,
    top_k_pairs: int = 200,
    min_pair_count: int = 2,
) -> dict[TagGroup, GroupMetrics]:
    out: dict[TagGroup, GroupMetrics] = {}
    for group in groups:
        out[group] = compute_group_metrics(
            rows,
            group=group,
            top_k_pairs=top_k_pairs,
            min_pair_count=min_pair_count,
        )
    return out


def compute_group_metrics(
    rows: Sequence[TrackRow],
    *,
    group: TagGroup,
    top_k_pairs: int = 200,
    min_pair_count: int = 2,
) -> GroupMetrics:
    n_tracks: int = int(len(rows))

    tag_counts: TagCounts = count_tag_presence(rows, group=group)
    tag_list: list[TagMetric] = tag_distribution(tag_counts=tag_counts, n_tracks=n_tracks)

    values: list[int] = [m.presence_count for m in tag_list]
    gini: float = gini_coefficient(values)
    entropy: float = shannon_entropy(values)
    entropy_norm: float = normalized_entropy(values)
    imbalance_ratio: float = imbalance_max_min(values)

    pair_counts: PairCounts = count_pair_cooccurrence(rows, group=group)

    pairs_by_count: list[PairMetric] = cooccurring_pairs(
        tag_counts=tag_counts,
        pair_counts=pair_counts,
        n_tracks=n_tracks,
        top_k=top_k_pairs,
        min_pair_count=min_pair_count,
        sort_by="count",
    )
    pairs_by_npmi: list[PairMetric] = cooccurring_pairs(
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
        n_unique_tags=int(len(tag_counts)),
        gini=gini,
        entropy=entropy,
        entropy_norm=entropy_norm,
        imbalance_ratio=imbalance_ratio,
        tags=tag_list,
        cooccurring_pairs=pairs_by_count,
        npmi_pairs=pairs_by_npmi,
    )


def _raw_tags(row: TrackRow, group: TagGroup) -> Sequence[str]:
    if group == "mood/theme":
        return row["mood_tags"]
    if group == "genre":
        return row["genre_tags"]
    return row["instrument_tags"]


def _tag_tokens(row: TrackRow, group: TagGroup) -> list[str]:
    out: list[str] = []
    for t in _raw_tags(row, group):
        if not isinstance(t, str):
            continue
        x: str = t.strip().lower()
        if not x:
            continue
        out.append(x)
    return out


def count_tag_presence(rows: Sequence[TrackRow], *, group: TagGroup) -> TagCounts:
    counts: TagCounts = Counter()
    for r in rows:
        counts.update(set(_tag_tokens(r, group)))
    return counts


def tag_distribution(*, tag_counts: Mapping[str, int], n_tracks: int) -> list[TagMetric]:
    out: list[TagMetric] = []
    for tag, c in tag_counts.items():
        p_track: float = (float(c) / float(n_tracks)) if n_tracks > 0 else 0.0
        out.append(TagMetric(tag=str(tag), presence_count=int(c), p_track=p_track))
    out.sort(key=lambda x: (-x.presence_count, x.tag))
    return out


def count_pair_cooccurrence(rows: Sequence[TrackRow], *, group: TagGroup) -> PairCounts:
    counts: PairCounts = Counter()
    for r in rows:
        tags: list[str] = sorted(set(_tag_tokens(r, group)))
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
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if min_pair_count <= 0:
        raise ValueError("min_pair_count must be > 0")
    if n_tracks < 0:
        raise ValueError("n_tracks must be >= 0")

    out: list[PairMetric] = []
    for (a, b), c_xy in pair_counts.items():
        if int(c_xy) < int(min_pair_count):
            continue
        if n_tracks == 0:
            continue

        c_a: int = int(tag_counts.get(a, 0))
        c_b: int = int(tag_counts.get(b, 0))
        if c_a <= 0 or c_b <= 0:
            continue

        p_x: float = float(c_a) / float(n_tracks)
        p_y: float = float(c_b) / float(n_tracks)
        p_xy: float = float(c_xy) / float(n_tracks)

        npmi: float = normalized_pmi(p_x=p_x, p_y=p_y, p_xy=p_xy, eps=1e-12)
        if math.isnan(npmi):
            continue

        out.append(
            PairMetric(
                tag_a=str(a),
                tag_b=str(b),
                cooccur_count=int(c_xy),
                p_xy=p_xy,
                npmi=npmi,
            )
        )

    if sort_by == "count":
        out.sort(key=lambda x: (-x.cooccur_count, x.tag_a, x.tag_b))
    else:
        out.sort(key=lambda x: (-x.npmi, -x.cooccur_count, x.tag_a, x.tag_b))

    return out[:top_k]


def normalized_pmi(*, p_x: float, p_y: float, p_xy: float, eps: float) -> float:
    if eps < 0.0:
        raise ValueError("eps must be >= 0")

    if p_x <= 0.0 or p_y <= 0.0 or p_xy <= 0.0:
        return math.nan

    if p_xy - p_x > eps or p_xy - p_y > eps:
        return math.nan

    if p_x - 1.0 > eps or p_y - 1.0 > eps or p_xy - 1.0 > eps:
        return math.nan

    denom: float = -math.log(p_xy)
    if denom <= 0.0:
        return math.nan

    pmi: float = math.log(p_xy / (p_x * p_y))
    return pmi / denom


def shannon_entropy(values: Sequence[int]) -> float:
    total: int = int(sum(int(v) for v in values))
    if total <= 0:
        return 0.0

    h: float = 0.0
    for v in values:
        c: int = int(v)
        if c <= 0:
            continue
        p: float = float(c) / float(total)
        h -= p * math.log(p)
    return h


def normalized_entropy(values: Sequence[int]) -> float:
    k: int = int(len([v for v in values if int(v) > 0]))
    if k <= 1:
        return 0.0
    h: float = shannon_entropy(values)
    return h / math.log(float(k))


def imbalance_max_min(values: Sequence[int]) -> float:
    pos: list[int] = [int(v) for v in values if int(v) > 0]
    if len(pos) == 0:
        return 0.0
    v_min: int = min(pos)
    v_max: int = max(pos)
    if v_min <= 0:
        return 0.0
    return float(v_max) / float(v_min)


def gini_coefficient(values: Sequence[int]) -> float:
    x: list[int] = [int(v) for v in values]
    if any(v < 0 for v in x):
        raise ValueError("gini_coefficient values must be >= 0")

    n: int = int(len(x))
    if n == 0:
        return 0.0

    total: int = int(sum(x))
    if total == 0:
        return 0.0

    x_sorted: list[int] = sorted(x)
    cum: int = 0
    for i, v in enumerate(x_sorted, start=1):
        cum += i * int(v)

    return (2.0 * float(cum)) / (float(n) * float(total)) - (float(n) + 1.0) / float(n)
