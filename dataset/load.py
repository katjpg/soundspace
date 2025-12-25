from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Literal, TypedDict

from configs.dataset import AppConfig

DatasetName = Literal["mtg_jamendo", "deam"]
TagGroup = Literal["mood/theme", "genre", "instrument"]


class TrackRow(TypedDict):
    track_id: str
    audio_path: str
    duration_s: float
    mood_tags: list[str]
    genre_tags: list[str]
    instrument_tags: list[str]


@dataclass(frozen=True, slots=True)
class Dataset:
    """Store a dataset name and its root folder."""
    name: str
    root: Path


def load_dataset(config: AppConfig, project_root: Path, name: str) -> Dataset:
    """Load a dataset by name."""
    try:
        loader = _LOADERS[name]
    except KeyError as e:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_LOADERS)}") from e
    return loader(config, project_root)


def load_mtg_rows(config: AppConfig) -> list[TrackRow]:
    """Load MTG-Jamendo metadata TSV into TrackRow records.

    This expects TAG tokens formatted like:
    - mood/theme---happy
    - genre---rock
    - instrument---guitar

    It also rewrites PATH values to match low-quality downloads:
    - 13/95713.mp3 -> 13/95713.low.mp3
    """
    mtg = config.datasets.mtg_jamendo

    out: list[TrackRow] = []
    with mtg.metadata_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        _header = next(reader, None)

        for cols in reader:
            if len(cols) < 6:
                continue

            track_id: str = cols[0].strip()
            rel_path: str = cols[3].strip()
            duration_s: float = float(cols[4])

            low_rel_path: str = _to_low_mp3_path(rel_path)
            audio_path: Path = (mtg.audio_root / low_rel_path).resolve(strict=False)

            tags_blob: str = " ".join(cols[5:])
            tokens: list[str] = [t for t in tags_blob.split() if t]

            mood_tags: list[str] = []
            genre_tags: list[str] = []
            instrument_tags: list[str] = []

            for tok in tokens:
                group, tag = _split_tag_token(tok)
                if group == "mood/theme":
                    mood_tags.append(tag)
                elif group == "genre":
                    genre_tags.append(tag)
                elif group == "instrument":
                    instrument_tags.append(tag)

            out.append(
                {
                    "track_id": track_id,
                    "audio_path": str(audio_path),
                    "duration_s": duration_s,
                    "mood_tags": mood_tags,
                    "genre_tags": genre_tags,
                    "instrument_tags": instrument_tags,
                }
            )

    return out


def _to_low_mp3_path(rel_path: str) -> str:
    """Rewrite MTG path to low-quality filename."""
    p: Path = Path(rel_path)
    if p.suffix.lower() != ".mp3":
        return rel_path
    return str(p.with_name(f"{p.stem}.low{p.suffix}"))


def _split_tag_token(token: str) -> tuple[str, str]:
    """Split 'group---tag' into ('group', 'tag'); return ('','') if invalid."""
    tok: str = token.strip()
    if "---" not in tok:
        return "", ""
    group, tag = tok.split("---", 1)
    return group.strip(), tag.strip()


def _load_mtg_jamendo(config: AppConfig, project_root: Path) -> Dataset:
    """Build paths for the MTG-Jamendo dataset."""
    root = (project_root / config.datasets.mtg_jamendo.root).resolve()
    return Dataset(name="mtg_jamendo", root=root)


def _load_deam(config: AppConfig, project_root: Path) -> Dataset:
    """Build paths for the DEAM dataset."""
    root = (project_root / config.datasets.deam.root).resolve()
    return Dataset(name="deam", root=root)


_LOADERS = {
    "mtg_jamendo": _load_mtg_jamendo,
    "deam": _load_deam,
}
