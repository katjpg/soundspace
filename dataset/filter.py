from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal, TypeAlias

from pathlib import Path


TagGroup: TypeAlias = Literal["mood", "genre", "theme", "style"]
SamplingStrategy: TypeAlias = Literal["none", "uniform"]


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    audio_path: Path
    duration_s: float
    quadrant: str | None
    arousal: float | None
    valence: float | None
    mood_tags: tuple[str, ...]
    genre_tags: tuple[str, ...]
    theme_tags: tuple[str, ...]
    style_tags: tuple[str, ...]


    def tags(self, group: TagGroup) -> tuple[str, ...]:
        if group == "mood":
            return self.mood_tags
        if group == "genre":
            return self.genre_tags
        if group == "theme":
            return self.theme_tags
        return self.style_tags


    def has_tags_in(self, groups: Sequence[TagGroup]) -> bool:
        for group in groups:
            if len(self.tags(group)) == 0:
                return False
        return True


@dataclass(frozen=True, slots=True)
class FilterConfig:
    require_groups: tuple[TagGroup, ...] = ()
    sampling: SamplingStrategy = "none"
    sample_size: int | None = None
    seed: int = 7


def filter_tracks(tracks: Sequence[Track], config: FilterConfig) -> list[Track]:
    filtered = (
        list(tracks)
        if len(config.require_groups) == 0
        else [t for t in tracks if t.has_tags_in(config.require_groups)]
    )

    if config.sampling == "none":
        return filtered

    if config.sample_size is None or config.sample_size <= 0:
        raise ValueError("sample_size must be > 0 for sampling")

    return _sample_uniform(filtered, config.sample_size, config.seed)


def clean_tags(raw_tags: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in raw_tags:
        cleaned = tag.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return tuple(out)


def _sample_uniform(tracks: Sequence[Track], sample_size: int, seed: int) -> list[Track]:
    if len(tracks) == 0:
        return []

    rng = Random(seed)
    k = min(sample_size, len(tracks))
    indices = sorted(rng.sample(range(len(tracks)), k=k))
    return [tracks[i] for i in indices]
