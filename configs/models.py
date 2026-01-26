from dataclasses import dataclass, field
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
class Model:
    name: str
    provider: str
    model_id: str | None = None
    device: str | None = None
    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class KNNParameters:
    """k-nearest neighbor graph construction parameters."""

    k: int = 10


@dataclass(frozen=True, slots=True)
class LeidenParameters:
    """Leiden community detection parameters."""

    resolution: float = 1.0
    seed: int | None = 42


@dataclass(frozen=True, slots=True)
class UMAPParameters:
    """UMAP dimensionality reduction parameters."""

    n_neighbors: int = 100
    min_dist: float = 0.10
    seed: int | None = 42
    metric: str = "cosine"


@dataclass(frozen=True, slots=True)
class PipelineParameters:
    """Tunable parameters for the embedding-to-visualization pipeline."""

    knn: KNNParameters = field(default_factory=KNNParameters)
    leiden: LeidenParameters = field(default_factory=LeidenParameters)
    umap: UMAPParameters = field(default_factory=UMAPParameters)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    paths: PathConfig
    models: dict[str, Model]
    parameters: PipelineParameters = field(default_factory=PipelineParameters)


def load_config(config_path: Path) -> ModelConfig:
    """Load model config from YAML file."""
    raw = _load_yaml(config_path)
    paths = _load_paths(config_path, raw)
    models = _load_models(raw)
    parameters = _load_parameters(raw)
    return ModelConfig(paths=paths, models=models, parameters=parameters)


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
    if "models" not in data:
        raise ValueError(f"{path} missing 'models' section")

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


def _load_models(raw: dict[str, Any]) -> dict[str, Model]:
    models_raw = raw["models"]

    if not isinstance(models_raw, dict):
        raise ValueError("models section must be a dict")

    models: dict[str, Model] = {}

    for name, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"model '{name}' spec must be a dict")

        if "provider" not in spec:
            raise ValueError(f"model '{name}' missing 'provider'")

        models[name] = Model(
            name=name,
            provider=str(spec["provider"]),
            model_id=spec.get("model_id"),
            device=spec.get("device"),
            dtype=spec.get("dtype", "float32"),
        )

    return models


def _load_parameters(raw: dict[str, Any]) -> PipelineParameters:
    """Load optional parameters section, (or alternatively use defaults)."""
    params_raw = raw.get("parameters")
    if params_raw is None:
        return PipelineParameters()

    if not isinstance(params_raw, dict):
        raise ValueError("parameters section must be a dict")

    knn_raw = params_raw.get("knn", {})
    leiden_raw = params_raw.get("leiden", {})
    umap_raw = params_raw.get("umap", {})

    return PipelineParameters(
        knn=KNNParameters(**knn_raw) if knn_raw else KNNParameters(),
        leiden=LeidenParameters(**leiden_raw) if leiden_raw else LeidenParameters(),
        umap=UMAPParameters(**umap_raw) if umap_raw else UMAPParameters(),
    )
