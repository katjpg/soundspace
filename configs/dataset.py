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
    data_root = _resolve_path(base_dir, raw, "data_root", "config")
    datasets_raw = _mapping(raw.get("datasets"), "config.datasets")

    mtg = _load_mtg_config(data_root, datasets_raw)
    deam = _load_deam_config(data_root, datasets_raw)

    return AppConfig(
        data_root=data_root,
        datasets=DatasetsConfig(mtg_jamendo=mtg, deam=deam),
    )


def _load_mtg_config(data_root: Path, datasets_raw: dict) -> MTGJamendoConfig:
    """Build MTG-Jamendo config from raw YAML."""
    ctx = "config.datasets.mtg_jamendo"
    mtg_raw = _mapping(datasets_raw.get("mtg_jamendo"), ctx)
    
    root = _resolve_path(data_root, mtg_raw, "root", ctx)
    metadata_file = _resolve_path(root, mtg_raw, "metadata_file", ctx)
    splits_dir = _resolve_path(root, mtg_raw, "splits_dir", ctx)
    audio_root = _resolve_path(root, mtg_raw, "audio_root", ctx)

    return MTGJamendoConfig(
        root=root,
        metadata_file=metadata_file,
        splits_dir=splits_dir,
        audio_root=audio_root,
    )


def _load_deam_config(data_root: Path, datasets_raw: dict) -> DEAMConfig:
    """Build DEAM config from raw YAML."""
    ctx = "config.datasets.deam"
    deam_raw = _mapping(datasets_raw.get("deam"), ctx)
    ann_raw = _mapping(deam_raw.get("annotations"), f"{ctx}.annotations")
    
    root = _resolve_path(data_root, deam_raw, "root", ctx)
    audio_dir = _resolve_path(root, deam_raw, "audio_dir", ctx)
    features_dir = _resolve_path(root, deam_raw, "features_dir", ctx)

    ann_ctx = f"{ctx}.annotations"
    avg_dyn = _resolve_path(root, ann_raw, "averaged_dynamic_dir", ann_ctx)
    avg_song = _resolve_path(root, ann_raw, "averaged_song_level_dir", ann_ctx)
    per_dyn = _resolve_path(root, ann_raw, "per_rater_dynamic_dir", ann_ctx)
    per_song = _resolve_path(root, ann_raw, "per_rater_song_level_dir", ann_ctx)

    return DEAMConfig(
        root=root,
        audio_dir=audio_dir,
        features_dir=features_dir,
        annotations_averaged_dynamic_dir=avg_dyn,
        annotations_averaged_song_level_dir=avg_song,
        annotations_per_rater_dynamic_dir=per_dyn,
        annotations_per_rater_song_level_dir=per_song,
    )


def _resolve_path(parent: Path, raw: dict, key: str, context: str) -> Path:
    """Extract string field from raw dict, join with parent, and resolve."""
    rel = _str_field(raw, key, context)
    return (parent / rel).resolve(strict=False)


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
