from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PathConfig:
    project_root: Path
    dataset_root: Path
    models_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class AudioConfig:
    dir: Path
    path_pattern: str


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    root: Path
    audio: AudioConfig
    metadata: dict[str, Any]
    task: dict[str, Any]
    cv: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    paths: PathConfig
    datasets: dict[str, Dataset]


def load_config(config_path: Path) -> DatasetConfig:
    """Load dataset config from YAML file."""
    raw = _load_yaml(config_path)
    paths = _load_paths(config_path, raw)
    datasets = _load_datasets(paths, raw)
    return DatasetConfig(paths=paths, datasets=datasets)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"failed to load {path}: {e}") from e
    
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML dict")
    
    if "paths" not in data:
        raise ValueError(f"{path} missing 'paths' section")
    if "datasets" not in data:
        raise ValueError(f"{path} missing 'datasets' section")
    
    return data


def _load_paths(config_path: Path, raw: dict[str, Any]) -> PathConfig:
    config_dir = config_path.resolve().parent
    paths_raw = raw["paths"]
    
    for key in ("project_root", "dataset_root", "models_root", "output_root"):
        if key not in paths_raw:
            raise ValueError(f"paths section missing '{key}'")
    
    project = (config_dir / str(paths_raw["project_root"])).resolve()
    
    return PathConfig(
        project_root=project,
        dataset_root=(project / str(paths_raw["dataset_root"])).resolve(),
        models_root=(project / str(paths_raw["models_root"])).resolve(),
        output_root=(project / str(paths_raw["output_root"])).resolve(),
    )


def _load_datasets(
    paths: PathConfig,
    raw: dict[str, Any],
) -> dict[str, Dataset]:
    datasets_raw = raw["datasets"]
    
    if not isinstance(datasets_raw, dict):
        raise ValueError("datasets section must be a dict")
    
    datasets: dict[str, Dataset] = {}
    
    for name, spec in datasets_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"dataset '{name}' spec must be a dict")
        
        if "root" not in spec:
            raise ValueError(f"dataset '{name}' missing 'root'")
        
        root = paths.dataset_root / str(spec["root"])
        audio_cfg = _load_audio(root, spec.get("audio", {}), name)
        metadata = _load_metadata(root, spec.get("metadata", {}))
        
        datasets[name] = Dataset(
            name=name,
            root=root,
            audio=audio_cfg,
            metadata=metadata,
            task=spec.get("task", {}),
            cv=spec.get("cv"),
        )
    
    return datasets


def _load_audio(
    root: Path,
    raw: dict[str, Any],
    dataset_name: str,
) -> AudioConfig:
    if not raw or not isinstance(raw, dict):
        raise ValueError(f"dataset '{dataset_name}' missing audio config")
    
    if "dir" not in raw:
        raise ValueError(f"dataset '{dataset_name}' audio missing 'dir'")
    if "path" not in raw:
        raise ValueError(f"dataset '{dataset_name}' audio missing 'path'")
    
    return AudioConfig(
        dir=root / str(raw["dir"]),
        path_pattern=str(raw["path"]),
    )


def _load_metadata(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve paths in nested metadata dicts/lists."""
    result: dict[str, Any] = {}
    
    for key, val in raw.items():
        if isinstance(val, dict):
            result[key] = _load_metadata(root, val)
        elif isinstance(val, list):
            result[key] = _resolve_path_list(root, val)
        elif _is_path_key(key):
            result[key] = root / str(val)
        else:
            result[key] = val
    
    return result


def _resolve_path_list(root: Path, items: list[Any]) -> list[Any]:
    return [root / str(item) if isinstance(item, str) else item for item in items]


def _is_path_key(key: str) -> bool:
    """Keys ending w/ _file, _dir, _path or named 'path'/'files' are treated as paths."""
    return key.endswith(("_file", "_dir", "_path")) or key in {"path", "files"}
