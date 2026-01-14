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
class ModelSpec:
    model_id: str
    output: str
    weights: Path | None = None
    metadata: Path | None = None


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    provider: str
    model_id: str | None = None
    device: str | None = None
    dtype: str = "float32"
    encoder: ModelSpec | None = None
    predictor: ModelSpec | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    paths: PathConfig
    models: dict[str, Model]


def load_config(config_path: Path) -> ModelConfig:
    raw: dict[str, Any] = _load_yaml(config_path)
    paths: PathConfig = _load_paths(config_path, raw)
    models: dict[str, Model] = _load_models(paths, raw)
    return ModelConfig(paths=paths, models=models)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        content: str = path.read_text(encoding="utf-8")
        data: Any = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"failed to load {path}: {e}") from e
    
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML dict")
    
    _require_key(data, "paths", f"config file {path}")
    _require_key(data, "models", f"config file {path}")
    
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


def _load_models(paths: PathConfig, raw: dict[str, Any]) -> dict[str, Model]:
    models: dict[str, Model] = {}
    models_raw: dict[str, Any] = raw["models"]
    
    if not isinstance(models_raw, dict):
        raise ValueError("models section must be a dict")
    
    for name, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"model '{name}' spec must be a dict, got {type(spec)}")
        
        _require_key(spec, "provider", f"model '{name}'")
        
        provider: str = str(spec["provider"])
        provider_dir: Path = paths.models_root / provider
        
        encoder: ModelSpec | None = _load_spec(
            provider_dir, spec.get("encoder"), name, "encoder"
        )
        
        predictor: ModelSpec | None = _load_spec(
            provider_dir, spec.get("predictor"), name, "predictor"
        )
        
        models[name] = Model(
            name=name,
            provider=provider,
            model_id=spec.get("model_id"),
            device=spec.get("device"),
            dtype=spec.get("dtype", "float32"),
            encoder=encoder,
            predictor=predictor,
        )
    
    return models


def _load_spec(
    base_dir: Path,
    raw: dict[str, Any] | None,
    model_name: str,
    spec_type: str,
) -> ModelSpec | None:
    if not raw:
        return None
    
    if not isinstance(raw, dict):
        raise ValueError(
            f"model '{model_name}' {spec_type} must be a dict, got {type(raw)}"
        )
    
    _require_key(raw, "model_id", f"model '{model_name}' {spec_type}")
    _require_key(raw, "output", f"model '{model_name}' {spec_type}")
    
    model_id: str = str(raw["model_id"])
    output: str = str(raw["output"])
    
    weights: Path | None = None
    if "weights" in raw:
        weights = base_dir / str(raw["weights"])
    
    metadata: Path | None = None
    if "metadata" in raw:
        metadata = base_dir / str(raw["metadata"])
    
    return ModelSpec(
        model_id=model_id,
        output=output,
        weights=weights,
        metadata=metadata,
    )


def _require_key(d: dict[str, Any], key: str, context: str) -> Any:
    if key not in d:
        raise ValueError(f"{context} missing required key '{key}'")
    return d[key]
