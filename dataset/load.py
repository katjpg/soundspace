from pathlib import Path
import csv

from configs.dataset import DatasetConfig
from dataset.filter import Track, TagGroup

TAG_SEPARATOR = "---"
AUDIO_SUFFIX = ".low"


def load_mtg_tracks(config: DatasetConfig) -> list[Track]:
    """Load MTG-Jamendo metadata TSV into Track records."""
    mtg = config.datasets["mtg_jamendo"]
    metadata_file = mtg.metadata["path"]
    audio_dir = mtg.audio.dir
    
    tracks: list[Track] = []
    
    with metadata_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        
        for cols in reader:
            if len(cols) < 6:
                continue
            
            track_id = cols[0].strip()
            rel_path = cols[3].strip()
            duration_s = float(cols[4])
            
            low_rel_path = _to_low_mp3_path(rel_path)
            audio_path = audio_dir / low_rel_path
            
            tags_blob = " ".join(cols[5:])
            tokens = [t for t in tags_blob.split() if t]
            
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
            
            tracks.append(
                Track(
                    track_id=track_id,
                    audio_path=str(audio_path),
                    duration_s=duration_s,
                    mood_tags=tuple(mood_tags),
                    genre_tags=tuple(genre_tags),
                    instrument_tags=tuple(instrument_tags),
                )
            )
    
    return tracks


def _to_low_mp3_path(rel_path: str) -> str:
    """Rewrite '13/95713.mp3' -> '13/95713.low.mp3'."""
    p = Path(rel_path)
    if p.suffix.lower() != ".mp3":
        return rel_path
    return str(p.with_name(f"{p.stem}{AUDIO_SUFFIX}{p.suffix}"))


def _split_tag_token(token: str) -> tuple[str, str]:
    """Parse 'mood/theme---happy' -> ('mood/theme', 'happy')."""
    tok = token.strip()
    if TAG_SEPARATOR not in tok:
        return "", ""
    
    group, tag = tok.split(TAG_SEPARATOR, 1)
    return group.strip(), tag.strip()
