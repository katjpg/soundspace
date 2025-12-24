from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import random
from typing import Any, Literal, TypedDict


TagGroup = Literal["mood/theme", "genre", "instrument"]
SamplingStrategy = Literal["stratified", "uniform", "none"]

ALL_GROUPS: tuple[TagGroup, ...] = ("mood/theme", "genre", "instrument")


class TrackRow(TypedDict):
    track_id: str
    audio_path: str
    duration_s: float
    mood_tags: list[str]
    genre_tags: list[str]
    instrument_tags: list[str]


@dataclass(frozen=True, slots=True)
class TagConfig:
    require: tuple[TagGroup, ...] = ALL_GROUPS
    
    stratify_by: TagGroup = "mood/theme"
    min_samples_per_tag: int = 30
    
    quota_tag_count: int | None = None
    
    normalize: Mapping[TagGroup, Mapping[str, str]] | None = None
    
    def validate(self) -> None:
        if len(self.require) == 0:
            raise ValueError("tags.require must not be empty")
        if self.quota_tag_count is not None and int(self.quota_tag_count) <= 0:
            raise ValueError("tags.quota_tag_count must be > 0 when set")
        if int(self.min_samples_per_tag) < 0:
            raise ValueError("tags.min_samples_per_tag must be >= 0")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    strategy: SamplingStrategy = "none"
    sample_size: int | None = None
    seed: int = 7
    
    def validate(self) -> None:
        if self.strategy == "none":
            return
        if self.sample_size is None:
            raise ValueError(f"sampling.sample_size is required for strategy={self.strategy!r}")
        if int(self.sample_size) <= 0:
            raise ValueError("sampling.sample_size must be > 0")


@dataclass(frozen=True, slots=True)
class FilterConfig:
    tags: TagConfig = TagConfig()
    sampling: SamplingConfig = SamplingConfig()
    
    def validate(self) -> None:
        self.tags.validate()
        self.sampling.validate()


def config_to_dict(config: FilterConfig) -> dict[str, Any]:
    return asdict(config)


def filter_rows(rows: Sequence[TrackRow], config: FilterConfig) -> tuple[list[TrackRow], dict[str, int]]:
    """Filter and sample track rows.
    
    Args:
        rows: Input track rows.
        config: Filter configuration.
    
    Returns:
        Filtered rows and statistics dict.
    """
    config.validate()
    
    norm: Mapping[TagGroup, Mapping[str, str]] = config.tags.normalize or {}
    
    # clean and normalize
    cleaned: list[TrackRow] = []
    for r in rows:
        cleaned.append(_clean_row(r, norm))
    
    # filter by required groups
    kept: list[TrackRow] = []
    for r in cleaned:
        if _has_required_groups(r, config.tags.require):
            kept.append(r)
    
    # sample
    out: list[TrackRow] = _sample_rows(kept, config)
    
    stats: dict[str, int] = {
        "tracks_in": int(len(rows)),
        "tracks_after_clean": int(len(cleaned)),
        "tracks_after_require": int(len(kept)),
        "tracks_out": int(len(out)),
    }
    return out, stats


def _tags(row: TrackRow, group: TagGroup) -> list[str]:
    """Get tags for a group from a row."""
    if group == "mood/theme":
        return row["mood_tags"]
    if group == "genre":
        return row["genre_tags"]
    return row["instrument_tags"]


def _set_tags(row: TrackRow, group: TagGroup, tags: list[str]) -> None:
    """Set tags for a group in a row."""
    if group == "mood/theme":
        row["mood_tags"] = tags
    elif group == "genre":
        row["genre_tags"] = tags
    else:
        row["instrument_tags"] = tags


def _clean_row(row: TrackRow, norm: Mapping[TagGroup, Mapping[str, str]]) -> TrackRow:
    """Clean and normalize a row."""
    return {
        "track_id": str(row["track_id"]),
        "audio_path": str(row["audio_path"]),
        "duration_s": float(row["duration_s"]),
        "mood_tags": _clean_tags(row["mood_tags"], norm.get("mood/theme")),
        "genre_tags": _clean_tags(row["genre_tags"], norm.get("genre")),
        "instrument_tags": _clean_tags(row["instrument_tags"], norm.get("instrument")),
    }


def _clean_tags(tags: Sequence[str], mapping: Mapping[str, str] | None) -> list[str]:
    """Clean tags: strip, lowercase, deduplicate, normalize."""
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        x: str = str(t).strip().lower()
        if not x:
            continue
        if mapping is not None:
            x = mapping.get(x, x)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _has_required_groups(row: TrackRow, required: Sequence[TagGroup]) -> bool:
    """Check if row has non-empty tags for all required groups."""
    for group in required:
        if len(_tags(row, group)) == 0:
            return False
    return True


def _select_quota_tags(
    rows: Sequence[TrackRow],
    *,
    group: TagGroup,
    tag_count: int | None,
) -> list[str]:
    """Select tags for quota-based stratified sampling.
    
    Args:
        rows: Track rows.
        group: Tag group to select from.
        tag_count: Max tags to select (None = all tags).
    
    Returns:
        Tags sorted by count desc, then name.
    """
    counts: Counter[str] = Counter()
    for r in rows:
        counts.update(_tags(r, group))
    
    # sort by count desc, then name
    items: list[tuple[str, int]] = [(tag, int(n)) for tag, n in counts.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    
    # apply tag_count limit
    if tag_count is None:
        return [tag for tag, _ in items]
    return [tag for tag, _ in items[: int(tag_count)]]


def _sample_rows(rows: Sequence[TrackRow], config: FilterConfig) -> list[TrackRow]:
    """Sample rows according to config strategy."""
    sampling: SamplingConfig = config.sampling
    tags: TagConfig = config.tags
    
    if sampling.strategy == "none":
        return list(rows)
    
    if len(rows) == 0:
        return []
    
    sample_size: int | None = sampling.sample_size
    if sample_size is None:
        raise ValueError(f"sampling.sample_size is required for strategy={sampling.strategy!r}")
    n: int = int(sample_size)
    
    if sampling.strategy == "uniform":
        rng: random.Random = random.Random(int(sampling.seed))
        k: int = min(n, len(rows))
        idx: list[int] = sorted(rng.sample(range(len(rows)), k=k))
        return [rows[i] for i in idx]
    
    if sampling.strategy == "stratified":
        min_per_tag: int = int(tags.min_samples_per_tag)
        if min_per_tag <= 0:
            raise ValueError("tags.min_samples_per_tag must be > 0 for stratified sampling")
        
        quota_tags: list[str] = _select_quota_tags(
            rows,
            group=tags.stratify_by,
            tag_count=tags.quota_tag_count,
        )
        if len(quota_tags) == 0:
            raise ValueError(f"no quota tags available for group={tags.stratify_by!r}")
        
        return _sample_quota_multilabel(
            rows=rows,
            group=tags.stratify_by,
            quota_tags=quota_tags,
            sample_size=n,
            min_per_tag=min_per_tag,
            seed=int(sampling.seed),
        )
    
    raise ValueError(f"unknown sampling strategy: {sampling.strategy!r}")


def _quota_gaps(
    quota_tags: Sequence[str],
    hit_by_tag: Mapping[str, int],
    min_per_tag: int,
) -> list[tuple[int, str]]:
    """Compute quota gaps: tags below min_per_tag and their deficit."""
    gaps: list[tuple[int, str]] = []
    for tag in quota_tags:
        hit: int = int(hit_by_tag[tag])
        if hit < min_per_tag:
            gaps.append((min_per_tag - hit, tag))
    return gaps


def _sample_quota_multilabel(
    *,
    rows: Sequence[TrackRow],
    group: TagGroup,
    quota_tags: Sequence[str],
    sample_size: int,
    min_per_tag: int,
    seed: int,
) -> list[TrackRow]:
    """Stratified sampling w quota guarantees for multi-label tags.
    
    Two-phase process:
    1. Build quota core: ensure each quota tag has min_per_tag tracks.
    2. Fill remainder: uniform random from remaining tracks.
    
    Args:
        rows: Track rows.
        group: Tag group for stratification.
        quota_tags: Tags requiring quota guarantees.
        sample_size: Target sample size.
        min_per_tag: Minimum tracks per quota tag.
        seed: Random seed.
    
    Returns:
        Sampled rows.
    """
    if len(quota_tags) == 0:
        raise ValueError("quota_tags must not be empty")
    if min_per_tag <= 0:
        raise ValueError(f"min_per_tag must be > 0, got {min_per_tag}")
    if sample_size <= 0:
        raise ValueError(f"sample_size must be > 0, got {sample_size}")
    if len(rows) == 0:
        return []
    
    quota_list: list[str] = list(quota_tags)
    quota_set: set[str] = set(quota_list)
    
    # build index: tag -> row indices
    row_idx_by_tag: dict[str, list[int]] = {tag: [] for tag in quota_list}
    quota_tags_by_row: list[set[str]] = []
    
    for i, r in enumerate(rows):
        row_tags: set[str] = set(_tags(r, group))
        row_quota_tags: set[str] = set()
        for tag in row_tags:
            if tag in quota_set:
                row_idx_by_tag[tag].append(i)
                row_quota_tags.add(tag)
        quota_tags_by_row.append(row_quota_tags)
    
    # verify each quota tag has enough rows
    for tag in quota_list:
        have_n: int = len(row_idx_by_tag[tag])
        if have_n < min_per_tag:
            raise ValueError(f"insufficient rows for tag={tag!r}: have {have_n}, need {min_per_tag}")
    
    rng: random.Random = random.Random(seed)
    picked_idx: set[int] = set()
    hit_by_tag: dict[str, int] = {tag: 0 for tag in quota_list}
    
    # phase 1: fill quota core
    while True:
        gaps: list[tuple[int, str]] = _quota_gaps(quota_list, hit_by_tag, min_per_tag)
        if len(gaps) == 0:
            break
        
        # pick tag w largest gap
        gaps.sort(key=lambda x: (-x[0], x[1]))
        target_tag: str = gaps[0][1]
        
        # pick unpicked row w target tag
        candidates: list[int] = [i for i in row_idx_by_tag[target_tag] if i not in picked_idx]
        if len(candidates) == 0:
            raise ValueError(f"cannot satisfy quota for tag={target_tag!r} with unique tracks")
        
        chosen: int = rng.choice(candidates)
        picked_idx.add(chosen)
        
        # increment hit counts for all quota tags in this row
        for tag in quota_tags_by_row[chosen]:
            if hit_by_tag[tag] < min_per_tag:
                hit_by_tag[tag] += 1
        
        if len(picked_idx) > sample_size:
            raise ValueError(f"quota core too large: {len(picked_idx)} > sample_size={sample_size}")
    
    # verify quota satisfied
    _check_quota(
        rows=rows,
        picked=picked_idx,
        group=group,
        quota_tags=quota_list,
        min_per_tag=min_per_tag,
    )
    
    # phase 2: fill remainder w uniform random
    if len(picked_idx) < sample_size:
        remaining: list[int] = [i for i in range(len(rows)) if i not in picked_idx]
        need: int = min(sample_size - len(picked_idx), len(remaining))
        picked_idx.update(rng.sample(remaining, k=need))
    
    out_idx: list[int] = sorted(picked_idx)
    return [rows[i] for i in out_idx]


def _check_quota(
    *,
    rows: Sequence[TrackRow],
    picked: set[int],
    group: TagGroup,
    quota_tags: Sequence[str],
    min_per_tag: int,
) -> None:
    """Verify quota constraints are satisfied."""
    counts: Counter[str] = Counter()
    for i in picked:
        counts.update(_tags(rows[i], group))
    
    missing: list[str] = []
    for tag in quota_tags:
        if int(counts.get(tag, 0)) < int(min_per_tag):
            missing.append(tag)
    
    if len(missing) != 0:
        raise ValueError(f"quota check failed: tags below min_per_tag={min_per_tag}: {sorted(missing)}")
