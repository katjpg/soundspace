from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal, TypeAlias

TagGroup: TypeAlias = Literal["mood", "genre", "instrument"]
SamplingStrategy: TypeAlias = Literal["stratified", "uniform", "none"]
TagMapping: TypeAlias = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    audio_path: str
    duration_s: float
    mood_tags: tuple[str, ...]
    genre_tags: tuple[str, ...]
    instrument_tags: tuple[str, ...]

    def tags(self, group: TagGroup) -> tuple[str, ...]:
        """Get tags for specified group."""
        if group == "mood":
            return self.mood_tags
        if group == "genre":
            return self.genre_tags
        return self.instrument_tags

    def has_tags_in(self, groups: Sequence[TagGroup]) -> bool:
        """Check if track has at least one tag in all specified groups."""
        for group in groups:
            if len(self.tags(group)) == 0:
                return False
        return True


@dataclass(frozen=True, slots=True)
class FilterConfig:
    require_groups: tuple[TagGroup, ...] = ("mood", "genre", "instrument")
    stratify_by: TagGroup = "mood"
    min_per_tag: int = 30
    top_k_tags: int | None = None
    tag_mapping: Mapping[TagGroup, TagMapping] | None = None
    sampling: SamplingStrategy = "none"
    sample_size: int | None = None
    seed: int = 7


def filter_tracks(
    tracks: Sequence[Track],
    config: FilterConfig,
) -> list[Track]:
    """
    Filter tracks by required groups and apply sampling strategy.
    
    Pipeline: require groups -> sample
    """
    if len(config.require_groups) == 0:
        raise ValueError("require_groups cannot be empty")
    
    filtered = [t for t in tracks if t.has_tags_in(config.require_groups)]
    
    if config.sampling == "none":
        return filtered
    
    if config.sample_size is None or config.sample_size <= 0:
        raise ValueError("sample_size must be > 0 for sampling")
    
    if config.sampling == "uniform":
        return _sample_uniform(filtered, config.sample_size, config.seed)
    
    if config.sampling == "stratified":
        if config.top_k_tags is not None and config.top_k_tags <= 0:
            raise ValueError("top_k_tags must be > 0")
        
        return _sample_stratified(
            filtered,
            group=config.stratify_by,
            sample_size=config.sample_size,
            min_per_tag=config.min_per_tag,
            top_k=config.top_k_tags,
            seed=config.seed,
        )
    
    raise ValueError(f"unknown sampling strategy: {config.sampling}")


def clean_tags(
    raw_tags: Sequence[str],
    mapping: TagMapping | None = None,
) -> tuple[str, ...]:
    """Strip, lowercase, deduplicate, normalize tags."""
    seen: set[str] = set()
    out: list[str] = []
    
    for tag in raw_tags:
        cleaned = tag.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        
        if mapping is not None:
            cleaned = mapping.get(cleaned, cleaned)
        
        if cleaned in seen:
            continue
        
        seen.add(cleaned)
        out.append(cleaned)
    
    return tuple(out)


def _sample_uniform(
    tracks: Sequence[Track],
    sample_size: int,
    seed: int,
) -> list[Track]:
    """Uniform random sampling without replacement."""
    if len(tracks) == 0:
        return []
    
    rng = Random(seed)
    k = min(sample_size, len(tracks))
    indices = sorted(rng.sample(range(len(tracks)), k=k))
    return [tracks[i] for i in indices]


def _sample_stratified(
    tracks: Sequence[Track],
    *,
    group: TagGroup,
    sample_size: int,
    min_per_tag: int,
    top_k: int | None,
    seed: int,
) -> list[Track]:
    """
    Stratified sampling ensuring min coverage per tag.
    
    Two-phase: fill min_per_tag quotas greedily, then uniform fill.
    """
    if len(tracks) == 0:
        return []
    
    if min_per_tag <= 0:
        raise ValueError(f"min_per_tag must be > 0, got {min_per_tag}")
    
    strat_tags = _select_top_tags(tracks, group, top_k)
    if len(strat_tags) == 0:
        raise ValueError(f"no tags found for stratification in group={group}")
    
    tag_set = set(strat_tags)
    track_tags = [set(t.tags(group)) & tag_set for t in tracks]
    
    tag_to_indices: dict[str, list[int]] = {tag: [] for tag in strat_tags}
    for i, tags in enumerate(track_tags):
        for tag in tags:
            tag_to_indices[tag].append(i)
    
    for tag in strat_tags:
        if len(tag_to_indices[tag]) < min_per_tag:
            raise ValueError(
                f"insufficient tracks for tag={tag}: "
                f"have {len(tag_to_indices[tag])}, need {min_per_tag}"
            )
    
    rng = Random(seed)
    picked: set[int] = set()
    tag_counts: dict[str, int] = {tag: 0 for tag in strat_tags}
    
    while any(count < min_per_tag for count in tag_counts.values()):
        deficits = [
            (min_per_tag - count, tag)
            for tag, count in tag_counts.items()
            if count < min_per_tag
        ]
        deficits.sort(key=lambda x: (-x[0], x[1]))
        target_tag = deficits[0][1]
        
        candidates = [i for i in tag_to_indices[target_tag] if i not in picked]
        if len(candidates) == 0:
            raise ValueError(
                f"cannot satisfy min_per_tag={min_per_tag} for tag={target_tag}"
            )
        
        chosen = rng.choice(candidates)
        picked.add(chosen)
        
        for tag in track_tags[chosen]:
            if tag_counts[tag] < min_per_tag:
                tag_counts[tag] += 1
        
        if len(picked) >= sample_size:
            break
    
    if len(picked) < sample_size:
        remaining = [i for i in range(len(tracks)) if i not in picked]
        need = min(sample_size - len(picked), len(remaining))
        picked.update(rng.sample(remaining, k=need))
    
    return [tracks[i] for i in sorted(picked)]


def _select_top_tags(
    tracks: Sequence[Track],
    group: TagGroup,
    top_k: int | None,
) -> list[str]:
    """Select top tags by frequency, sorted desc by count then asc by name."""
    counts: Counter[str] = Counter()
    for t in tracks:
        counts.update(t.tags(group))
    
    items = [(tag, count) for tag, count in counts.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    
    if top_k is None:
        return [tag for tag, _ in items]
    return [tag for tag, _ in items[:top_k]]
