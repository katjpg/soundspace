from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Literal, TypedDict

from configs.dataset import DatasetConfig


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
    name: str
    root: Path


def load_dataset(config: DatasetConfig, name: str) -> Dataset:
    if name not in config.datasets:
        available: list[str] = sorted(config.datasets.keys())
        raise ValueError(f"unknown dataset {name!r}. available: {available}")
    
    dataset = config.datasets[name]
    return Dataset(name=name, root=dataset.root)


def load_mtg_rows(config: DatasetConfig) -> list[TrackRow]:
    """
    Load MTG-Jamendo metadata TSV into TrackRow records.
    
    Expects tag tokens formatted as 'group---tag' (i.e., 'mood/theme---happy').
    Rewrites paths to low-quality format (i.e., '13/95713.mp3' -> '13/95713.low.mp3').
    """
    mtg = config.datasets["mtg_jamendo"]
    metadata_file: Path = mtg.metadata["path"]
    audio_dir: Path = mtg.audio.dir
    
    out: list[TrackRow] = []
    
    with metadata_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        _header = next(reader, None)
        
        for cols in reader:
            if len(cols) < 6:
                continue
            
            track_id: str = cols[0].strip()
            rel_path: str = cols[3].strip()
            duration_s: float = float(cols[4])
            
            # rewrite to low-quality MP3 path
            low_rel_path: str = _to_low_mp3_path(rel_path)
            audio_path: Path = (audio_dir / low_rel_path).resolve(strict=False)
            
            # parse tags from remaining columns
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
            
            out.append({
                "track_id": track_id,
                "audio_path": str(audio_path),
                "duration_s": duration_s,
                "mood_tags": mood_tags,
                "genre_tags": genre_tags,
                "instrument_tags": instrument_tags,
            })
    
    return out


def _to_low_mp3_path(rel_path: str) -> str:
    # 13/95713.mp3 -> 13/95713.low.mp3
    p: Path = Path(rel_path)
    if p.suffix.lower() != ".mp3":
        return rel_path
    return str(p.with_name(f"{p.stem}.low{p.suffix}"))


def _split_tag_token(token: str) -> tuple[str, str]:
    # 'mood/theme---happy' -> ('mood/theme', 'happy')
    tok: str = token.strip()
    if "---" not in tok:
        return "", ""
    group, tag = tok.split("---", 1)
    return group.strip(), tag.strip()
