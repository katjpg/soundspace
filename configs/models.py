from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class EssentiaModelSpec:
    pb: str
    json: str
    url_pb: str
    url_json: str


@dataclass(frozen=True, slots=True)
class EssentiaInferenceConfig:
    sample_rate: int
    embedding_output: str
    valence_arousal_output: str


@dataclass(frozen=True, slots=True)
class EssentiaConfig:
    cache_dir: Path
    inference: EssentiaInferenceConfig
    models: dict[str, EssentiaModelSpec]


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    root: Path
    essentia: EssentiaConfig

def load_config(config_path: Path | None = None) -> ModelsConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    models_root = Path(raw["models"]["root"]).resolve()
    essentia_cache = models_root / raw["models"]["essentia"]["cache_dir"]
    
    inference_raw = raw["models"]["essentia"]["inference"]
    inference_config = EssentiaInferenceConfig(
        sample_rate=inference_raw["sample_rate"],
        embedding_output=inference_raw["embedding_output"],
        valence_arousal_output=inference_raw["valence_arousal_output"],
    )
    
    essentia_models_raw = raw["models"]["essentia"]["models"]
    essentia_models = {
        key: EssentiaModelSpec(**spec)
        for key, spec in essentia_models_raw.items()
    }
    
    essentia_config = EssentiaConfig(
        cache_dir=essentia_cache,
        inference=inference_config,
        models=essentia_models,
    )

    return ModelsConfig(
        root=models_root,
        essentia=essentia_config,
    )
