from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class MTGJamendoConfig:
    root: Path
    metadata_file: Path
    splits_dir: Path
    audio_root: Path


@dataclass(frozen=True, slots=True)
class DEAMConfig:
    root: Path
    audio_dir: Path
    features_dir: Path
    annotations_averaged_dynamic_dir: Path
    annotations_averaged_song_level_dir: Path
    annotations_per_rater_dynamic_dir: Path
    annotations_per_rater_song_level_dir: Path


@dataclass(frozen=True, slots=True)
class DatasetsConfig:
    mtg_jamendo: MTGJamendoConfig
    deam: DEAMConfig


@dataclass(frozen=True, slots=True)
class AppConfig:
    data_root: Path
    datasets: DatasetsConfig


def load_config(config_path: Path) -> AppConfig:
    """Load config.yaml and build absolute paths."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {config_path}") from e

    raw = _mapping(raw, "config")
    base_dir = config_path.parent

    data_root = (base_dir / _str_field(raw, "data_root", "config")).resolve(strict=False)

    datasets_raw = _mapping(raw.get("datasets"), "config.datasets")

    # MTG Jamendo 
    mtg_raw = _mapping(datasets_raw.get("mtg_jamendo"), "config.datasets.mtg_jamendo")
    mtg_root = (data_root / _str_field(mtg_raw, "root", "config.datasets.mtg_jamendo")).resolve(strict=False)
    mtg = MTGJamendoConfig(
        root=mtg_root,
        metadata_file=(mtg_root / _str_field(mtg_raw, "metadata_file", "config.datasets.mtg_jamendo")).resolve(strict=False),
        splits_dir=(mtg_root / _str_field(mtg_raw, "splits_dir", "config.datasets.mtg_jamendo")).resolve(strict=False),
        audio_root=(mtg_root / _str_field(mtg_raw, "audio_root", "config.datasets.mtg_jamendo")).resolve(strict=False),
    )

    # DEAM
    deam_raw = _mapping(datasets_raw.get("deam"), "config.datasets.deam")
    ann_raw = _mapping(deam_raw.get("annotations"), "config.datasets.deam.annotations")
    deam_root = (data_root / _str_field(deam_raw, "root", "config.datasets.deam")).resolve(strict=False)
    deam = DEAMConfig(
        root=deam_root,
        audio_dir=(deam_root / _str_field(deam_raw, "audio_dir", "config.datasets.deam")).resolve(strict=False),
        features_dir=(deam_root / _str_field(deam_raw, "features_dir", "config.datasets.deam")).resolve(strict=False),
        annotations_averaged_dynamic_dir=(deam_root / _str_field(ann_raw, "averaged_dynamic_dir", "config.datasets.deam.annotations")).resolve(strict=False),
        annotations_averaged_song_level_dir=(deam_root / _str_field(ann_raw, "averaged_song_level_dir", "config.datasets.deam.annotations")).resolve(strict=False),
        annotations_per_rater_dynamic_dir=(deam_root / _str_field(ann_raw, "per_rater_dynamic_dir", "config.datasets.deam.annotations")).resolve(strict=False),
        annotations_per_rater_song_level_dir=(deam_root / _str_field(ann_raw, "per_rater_song_level_dir", "config.datasets.deam.annotations")).resolve(strict=False),
    )

    return AppConfig(
        data_root=data_root,
        datasets=DatasetsConfig(mtg_jamendo=mtg, deam=deam),
    )


def _mapping(val, where: str) -> dict:
    """Require a mapping."""
    if not isinstance(val, dict):
        raise ValueError(f"{where} must be a mapping")
    return val


def _str_field(d: dict, key: str, where: str) -> str:
    """Require a non-empty string field."""
    val = d.get(key)
    if not isinstance(val, str) or not val:
        raise ValueError(f"{where}.{key} must be a non-empty string")
    return val
