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
    raw: dict[str, Any] = _load_yaml(config_path)
    paths: PathConfig = _load_paths(config_path, raw)
    datasets: dict[str, Dataset] = _load_datasets(paths, raw)
    return DatasetConfig(paths=paths, datasets=datasets)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        content: str = path.read_text(encoding="utf-8")
        data: Any = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"failed to load {path}: {e}") from e
    
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML dict")
    
    _require_key(data, "paths", f"config file {path}")
    _require_key(data, "datasets", f"config file {path}")
    
    return data


def _load_paths(config_path: Path, raw: dict[str, Any]) -> PathConfig:
    config_dir: Path = config_path.resolve().parent
    paths_raw: dict[str, Any] = raw["paths"]
    
    _require_key(paths_raw, "project_root", "paths section")
    _require_key(paths_raw, "dataset_root", "paths section")
    _require_key(paths_raw, "models_root", "paths section")
    _require_key(paths_raw, "output_root", "paths section")
    
    project: Path = (config_dir / str(paths_raw["project_root"])).resolve()
    
    return PathConfig(
        project_root=project,
        dataset_root=(project / str(paths_raw["dataset_root"])).resolve(),
        models_root=(project / str(paths_raw["models_root"])).resolve(),
        output_root=(project / str(paths_raw["output_root"])).resolve(),
    )


def _load_datasets(paths: PathConfig, raw: dict[str, Any]) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {}
    datasets_raw: dict[str, Any] = raw["datasets"]
    
    if not isinstance(datasets_raw, dict):
        raise ValueError("datasets section must be a dict")
    
    for name, spec in datasets_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"dataset '{name}' spec must be a dict, got {type(spec)}")
        
        _require_key(spec, "root", f"dataset '{name}'")
        
        root: Path = paths.dataset_root / str(spec["root"])
        
        audio_cfg: AudioConfig = _load_audio(root, spec.get("audio", {}), name)
        metadata: dict[str, Any] = _load_metadata(root, spec.get("metadata", {}))
        task: dict[str, Any] = spec.get("task", {})
        cv: dict[str, Any] | None = spec.get("cv")
        
        datasets[name] = Dataset(
            name=name,
            root=root,
            audio=audio_cfg,
            metadata=metadata,
            task=task,
            cv=cv,
        )
    
    return datasets


def _load_audio(root: Path, raw: dict[str, Any], dataset_name: str) -> AudioConfig:
    if not raw:
        raise ValueError(f"dataset '{dataset_name}' missing audio config")
    
    if not isinstance(raw, dict):
        raise ValueError(f"dataset '{dataset_name}' audio config must be a dict")
    
    _require_key(raw, "dir", f"dataset '{dataset_name}' audio")
    _require_key(raw, "path", f"dataset '{dataset_name}' audio")
    
    return AudioConfig(
        dir=root / str(raw["dir"]),
        path_pattern=str(raw["path"]),
    )


def _load_metadata(root: Path, raw: dict[str, Any]) -> dict[str, Any]:
    # recursively resolve paths in nested metadata dicts/lists
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
    return [
        root / str(item) if isinstance(item, str) else item
        for item in items
    ]


def _is_path_key(key: str) -> bool:
    # keys ending w/ _file, _dir, _path or named 'path'/'files' are treated as paths
    path_suffixes: tuple[str, ...] = ("_file", "_dir", "_path")
    return key.endswith(path_suffixes) or key in {"path", "files"}


def _require_key(d: dict[str, Any], key: str, context: str) -> Any:
    if key not in d:
        raise ValueError(f"{context} missing required key '{key}'")
    return d[key]
