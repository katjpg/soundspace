import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from configs.dataset import DatasetConfig
from dataset.filter import Track, clean_tags


SplitName: TypeAlias = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class MergeMeta:
    duration_s: float | None
    mood_tags: tuple[str, ...]
    genre_tags: tuple[str, ...]
    theme_tags: tuple[str, ...]
    style_tags: tuple[str, ...]
    quadrant: str | None


@dataclass(frozen=True, slots=True)
class AV:
    arousal: float
    valence: float


def load_merge_tracks(config: DatasetConfig, *, split: SplitName = "train") -> list[Track]:
    ds = config.datasets["merge"]
    metadata_file = _require_path(ds.metadata, "metadata_file")
    av_values_file = _require_path(ds.metadata, "av_values_file")
    split_file = _merge_split_file(ds.metadata, split)

    meta_map = _read_merge_metadata(metadata_file)
    av_map = _read_merge_av_values(av_values_file)
    split_rows = _read_split_rows(split_file)

    duration_cache: dict[Path, float] = {}
    tracks: list[Track] = []

    for song_id, quadrant in split_rows:
        meta = meta_map.get(song_id)
        if meta is None:
            raise ValueError(f"MERGE metadata missing Song={song_id}")

        av = av_map.get(song_id)
        arousal = av.arousal if av is not None else None
        valence = av.valence if av is not None else None

        audio_path = ds.audio.dir / quadrant / f"{song_id}.mp3"

        duration_s = meta.duration_s
        if duration_s is None or duration_s <= 0.0:
            cached = duration_cache.get(audio_path)
            if cached is None:
                computed = _duration_from_audio(audio_path)
                if computed is not None and computed > 0.0:
                    duration_cache[audio_path] = computed
                    cached = computed
            duration_s = cached

        if duration_s is None:
            duration_s = 0.0

        tracks.append(
            Track(
                track_id=song_id,
                audio_path=audio_path,
                duration_s=duration_s,
                quadrant=quadrant,
                arousal=arousal,
                valence=valence,
                mood_tags=meta.mood_tags,
                genre_tags=meta.genre_tags,
                theme_tags=meta.theme_tags,
                style_tags=meta.style_tags,
            )
        )

    return tracks


def _merge_split_file(metadata: dict[str, object], split: SplitName) -> Path:
    split_spec = metadata.get("split")
    if not isinstance(split_spec, dict):
        raise ValueError("merge.metadata.split must be a dict")

    key = f"{split}_file"
    val = split_spec.get(key)
    if not isinstance(val, Path):
        raise ValueError(f"merge.metadata.split.{key} must be a resolved Path")

    return val


def _require_path(metadata: dict[str, object], key: str) -> Path:
    val = metadata.get(key)
    if not isinstance(val, Path):
        raise ValueError(f"merge.metadata.{key} must be a resolved Path")
    return val


def _read_merge_metadata(path: Path) -> dict[str, MergeMeta]:
    out: dict[str, MergeMeta] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = _get(row, "Song")
            quadrant = _get_optional(row, "Quadrant")
            duration_s = _to_optional_float(_get_optional(row, "Duration"))

            moods_all = _split_csv_list(_get_optional(row, "MoodsAll"))
            genres = _split_csv_list(_get_optional(row, "Genres"))
            themes = _split_csv_list(_get_optional(row, "Themes"))
            styles = _split_csv_list(_get_optional(row, "Styles"))

            out[song_id] = MergeMeta(
                duration_s=duration_s,
                mood_tags=clean_tags(moods_all),
                genre_tags=clean_tags(genres),
                theme_tags=clean_tags(themes),
                style_tags=clean_tags(styles),
                quadrant=quadrant,
            )

    return out


def _read_merge_av_values(path: Path) -> dict[str, AV]:
    out: dict[str, AV] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = _get(row, "Song")
            arousal = float(_get(row, "Arousal"))
            valence = float(_get(row, "Valence"))
            out[song_id] = AV(arousal=arousal, valence=valence)
    return out


def _read_split_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = _get(row, "Song")
            quadrant = _get(row, "Quadrant")
            rows.append((song_id, quadrant))
    return rows


def _duration_from_audio(path: Path) -> float | None:
    if not path.exists():
        return None

    duration = _duration_from_librosa(path)
    if duration is not None and duration > 0.0:
        return duration

    duration = _duration_from_essentia_metadata(path)
    if duration is not None and duration > 0.0:
        return duration

    duration = _duration_from_essentia_decode(path)
    if duration is not None and duration > 0.0:
        return duration

    return None


def _duration_from_librosa(path: Path) -> float | None:
    try:
        import librosa
    except Exception:
        return None

    try:
        val = float(librosa.get_duration(path=str(path)))
        return val if val > 0.0 else None
    except Exception:
        return None


def _essentia_standard() -> Any | None:
    try:
        return importlib.import_module("essentia.standard")
    except Exception:
        return None


def _duration_from_essentia_metadata(path: Path) -> float | None:
    es = _essentia_standard()
    if es is None:
        return None

    reader_ctor = getattr(es, "MetadataReader", None)
    if reader_ctor is None:
        return None

    try:
        meta = reader_ctor(filename=str(path))()
        if not isinstance(meta, (list, tuple)) or len(meta) < 9:
            return None
        val = meta[8]
        if val is None:
            return None
        out = float(val)
        return out if out > 0.0 else None
    except Exception:
        return None


def _duration_from_essentia_decode(path: Path) -> float | None:
    es = _essentia_standard()
    if es is None:
        return None

    loader_ctor = getattr(es, "MonoLoader", None)
    if loader_ctor is None:
        return None

    try:
        sample_rate = 44100
        audio = loader_ctor(filename=str(path), sampleRate=sample_rate)()
        n = int(getattr(audio, "shape", (0,))[0])
        if n <= 0:
            return None
        return float(n) / float(sample_rate)
    except Exception:
        return None


def _split_csv_list(blob: str | None) -> list[str]:
    if blob is None:
        return []
    s = blob.strip()
    if not s:
        return []
    return [part.strip() for part in s.split(",") if part.strip()]


def _to_optional_float(val: str | None) -> float | None:
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _get(row: dict[str, str | None], key: str) -> str:
    val = row.get(key)
    if val is None:
        raise ValueError(f"missing column '{key}'")
    s = val.strip()
    if not s:
        raise ValueError(f"empty value for column '{key}'")
    return s


def _get_optional(row: dict[str, str | None], key: str) -> str | None:
    val = row.get(key)
    if val is None:
        return None
    s = val.strip()
    return s if s else None
