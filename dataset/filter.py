from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal, TypeAlias

import pandas as pd


TagGroup: TypeAlias = Literal["mood", "genre", "theme", "style"]
SamplingStrategy: TypeAlias = Literal["none", "uniform"]


# column mappings: raw CSV headers -> standardized names
METADATA_COLUMNS: dict[str, str] = {
    "Song": "song_id",
    "Quadrant": "quadrant",
    "Artist": "artist",
    "Title": "title",
    "Duration": "duration",
    "Moods": "mood",
    "MoodsAll": "mood_all",
    "MoodsAllWeights": "mood_all_weights",
    "Genres": "genre",
    "GenreWeights": "genre_weights",
    "Themes": "theme",
    "ThemeWeights": "theme_weights",
    "Styles": "style",
    "StylesWeights": "style_weights",
}

AV_COLUMNS: dict[str, str] = {
    "Song": "song_id",
    "Arousal": "arousal",
    "Valence": "valence",
}

# allmusic 21 categories -> 10 consolidated categories
GENRE_TAXONOMY: dict[str, list[str]] = {
    "Rock/Pop": ["Pop/Rock"],
    "Electronic": ["Electronic", "Easy Listening"],
    "Hip-Hop/Rap": ["Rap"],
    "R&B/Soul": ["R&B"],
    "Jazz": ["Jazz"],
    "Classical": ["Classical"],
    "Folk/Country": ["Folk", "Country"],
    "Blues": ["Blues"],
    "World": ["International", "Latin", "Reggae"],
    "Experimental": ["Avant-Garde", "New Age"],
}

# categories describing audience/purpose rather than sonic characteristics
EXCLUDED_GENRES: tuple[str, ...] = (
    "Children's",
    "Holiday",
    "Religious",
    "Comedy/Spoken",
    "Stage & Screen",
    "Vocal",
)


@dataclass(frozen=True, slots=True)
class Track:
    """Single audio track with metadata and annotations."""

    track_id: str
    audio_path: Path
    quadrant: str
    duration_s: float
    artist: str
    title: str
    arousal: float
    valence: float
    mood_tags: tuple[str, ...]
    genre_tags: tuple[str, ...]
    genre_tags_consolidated: tuple[str, ...]
    theme_tags: tuple[str, ...]
    style_tags: tuple[str, ...]

    def tags(self, group: TagGroup) -> tuple[str, ...]:
        """Return tags for specified group."""
        if group == "mood":
            return self.mood_tags
        if group == "genre":
            return self.genre_tags_consolidated
        if group == "theme":
            return self.theme_tags
        return self.style_tags

    def has_tags_in(self, groups: Sequence[TagGroup]) -> bool:
        """Check if track has at least one tag in each specified group."""
        return all(len(self.tags(g)) > 0 for g in groups)


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Configuration for track filtering and sampling."""

    require_groups: tuple[TagGroup, ...] = ()
    exclude_invalid_genres: bool = True
    sampling: SamplingStrategy = "none"
    sample_size: int | None = None
    seed: int = 7


def load_merge_dataset(
    metadata_path: str | Path,
    av_path: str | Path,
    audio_dir: str | Path,
    config: FilterConfig | None = None,
) -> list[Track]:
    """
    Load MERGE balanced dataset and return filtered Track objects.

    Args
    ----
        metadata_path (str | Path) : path to merge_audio_balanced_metadata.csv.
        av_path       (str | Path) : path to merge_audio_balanced_av_values.csv.
        audio_dir     (str | Path) : root directory containing audio/{quadrant}/*.mp3.
        config     (FilterConfig)  : filtering configuration.
                                     (Default is FilterConfig()).

    Returns
    -------
        (list[Track]) : filtered tracks with complete annotations.
    """
    cfg = config or FilterConfig()

    meta_df = _read_metadata(Path(metadata_path))
    av_df = _read_av_values(Path(av_path))
    merged = _merge_dataframes(meta_df, av_df)

    audio_root = Path(audio_dir)
    tracks: list[Track] = []

    for _, row in merged.iterrows():
        track = _build_track(row, audio_root)
        if track is None:
            continue

        # skip tracks with only excluded genres
        if cfg.exclude_invalid_genres and not _has_valid_genres(track.genre_tags):
            continue

        tracks.append(track)

    return filter_tracks(tracks, cfg)


def filter_tracks(tracks: Sequence[Track], config: FilterConfig) -> list[Track]:
    """Apply tag requirements and sampling to track list."""
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


def consolidate_genres(raw_genres: Sequence[str]) -> tuple[str, ...]:
    """Map raw AllMusic genres to 10-category taxonomy."""
    consolidated: list[str] = []
    seen: set[str] = set()

    for genre in raw_genres:
        mapped = _map_genre(genre.strip())
        if mapped and mapped not in seen:
            seen.add(mapped)
            consolidated.append(mapped)

    return tuple(consolidated)


def clean_tags(raw_tags: Sequence[str]) -> tuple[str, ...]:
    """Normalize tags: strip, lowercase, deduplicate."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in raw_tags:
        cleaned = tag.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return tuple(out)

def _read_metadata(path: Path) -> pd.DataFrame:
    """Load and standardize metadata CSV."""
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df, set(METADATA_COLUMNS.keys()), path)

    df = df.rename(columns=METADATA_COLUMNS)
    return df[list(METADATA_COLUMNS.values())]


def _read_av_values(path: Path) -> pd.DataFrame:
    """Load and standardize AV values CSV."""
    if not path.exists():
        raise FileNotFoundError(f"AV values file not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df, set(AV_COLUMNS.keys()), path)

    df = df.rename(columns=AV_COLUMNS)
    return df[list(AV_COLUMNS.values())]


def _validate_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    """Raise ValueError if required columns missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def _merge_dataframes(meta_df: pd.DataFrame, av_df: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata with AV values, filter rows with all fields populated."""
    merged = meta_df.merge(av_df, on="song_id", how="inner")

    # require all tag columns populated
    tag_cols = ["mood_all", "genre", "theme", "style"]
    for col in tag_cols:
        merged = merged[merged[col].notna() & (merged[col].str.strip() != "")]

    # require numeric fields
    merged = merged[merged["arousal"].notna() & merged["valence"].notna()]

    return merged


def _build_track(row: pd.Series, audio_root: Path) -> Track | None:
    """Construct Track from DataFrame row, return None if audio missing."""
    song_id = str(row["song_id"])
    quadrant = str(row["quadrant"])
    audio_path = audio_root / quadrant / f"{song_id}.mp3"

    if not audio_path.exists():
        return None

    raw_genres = _parse_tags(row.get("genre", ""))
    raw_moods = _parse_tags(row.get("mood_all", ""))
    raw_themes = _parse_tags(row.get("theme", ""))
    raw_styles = _parse_tags(row.get("style", ""))

    return Track(
        track_id=song_id,
        audio_path=audio_path,
        quadrant=quadrant,
        duration_s=_parse_float(row.get("duration"), 0.0),
        artist=str(row.get("artist", "")),
        title=str(row.get("title", "")),
        arousal=float(row["arousal"]),
        valence=float(row["valence"]),
        mood_tags=clean_tags(raw_moods),
        genre_tags=tuple(raw_genres),
        genre_tags_consolidated=consolidate_genres(raw_genres),
        theme_tags=clean_tags(raw_themes),
        style_tags=clean_tags(raw_styles),
    )


def _parse_tags(value: object) -> list[str]:
    """Split comma-separated tag string into list."""
    if value is None:
        return []
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return []
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _parse_float(value: object, default: float) -> float:
    """Convert pandas cell value to float, return default if invalid or NaN."""
    if value is None:
        return default
    # handle pandas NA/NaN
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return default
    except (TypeError, ValueError):
        pass
    # handle string values
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def _map_genre(genre: str) -> str | None:
    """Map single genre to consolidated category, None if excluded."""
    if genre in EXCLUDED_GENRES:
        return None

    for consolidated, originals in GENRE_TAXONOMY.items():
        if genre in originals:
            return consolidated

    return None


def _has_valid_genres(genres: Sequence[str]) -> bool:
    """Check if track has at least one genre not in excluded list."""
    return any(g not in EXCLUDED_GENRES for g in genres)


def _sample_uniform(
    tracks: Sequence[Track], sample_size: int, seed: int
) -> list[Track]:
    """Random sample without replacement."""
    if len(tracks) == 0:
        return []

    rng = Random(seed)
    k = min(sample_size, len(tracks))
    indices = sorted(rng.sample(range(len(tracks)), k=k))
    return [tracks[i] for i in indices]