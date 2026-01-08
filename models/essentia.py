from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
import urllib.request

import numpy as np
import numpy.typing as npt

from configs.models import EssentiaConfig


FloatArray = npt.NDArray[np.float32]

MSD_MUSICNN_V1: Final[str] = "msd_musicnn_v1"
DEAM_V2: Final[str] = "deam_v2"


@dataclass(frozen=True, slots=True)
class ModelPaths:
    pb: Path
    json: Path


@dataclass(frozen=True, slots=True)
class AffectPredictor:
    emb_predictor: Any
    va_predictor: Any
    sr: int


def from_pretrained(config: EssentiaConfig) -> AffectPredictor:
    _check_essentia_available()

    model_paths = _get_model_paths(config)

    import essentia.standard as es  # type: ignore

    emb_pred = es.TensorflowPredictMusiCNN(  # type: ignore
        graphFilename=str(model_paths[MSD_MUSICNN_V1].pb),
        output=config.inference.embedding_output,
    )
    va_pred = es.TensorflowPredict2D(  # type: ignore
        graphFilename=str(model_paths[DEAM_V2].pb),
        output=config.inference.valence_arousal_output,
    )

    return AffectPredictor(
        emb_predictor=emb_pred,
        va_predictor=va_pred,
        sr=config.inference.sample_rate,
    )


def download_pretrained(config: EssentiaConfig, *, force: bool = False) -> dict[str, ModelPaths]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, ModelPaths] = {}

    for model_key, model_spec in config.models.items():
        pb_path = config.cache_dir / model_spec.pb
        json_path = config.cache_dir / model_spec.json

        if not pb_path.exists() or force:
            _download_file(model_spec.url_pb, pb_path)

        if not json_path.exists() or force:
            _download_file(model_spec.url_json, json_path)

        paths[model_key] = ModelPaths(pb=pb_path, json=json_path)

    return paths


def _get_model_paths(config: EssentiaConfig) -> dict[str, ModelPaths]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    missing: list[Path] = []
    paths: dict[str, ModelPaths] = {}

    for model_key, model_spec in config.models.items():
        pb_path = config.cache_dir / model_spec.pb
        json_path = config.cache_dir / model_spec.json

        paths[model_key] = ModelPaths(pb=pb_path, json=json_path)

        if not pb_path.exists():
            missing.append(pb_path)
        if not json_path.exists():
            missing.append(json_path)

    if missing:
        download_pretrained(config)

    for model_key, model_path in paths.items():
        if not model_path.pb.exists():
            raise FileNotFoundError(f"model file not found after download: {model_path.pb}")
        if not model_path.json.exists():
            raise FileNotFoundError(f"model file not found after download: {model_path.json}")

    return paths


def _check_essentia_available() -> None:
    try:
        import essentia.standard as es  # type: ignore
    except ImportError as e:
        raise ImportError(
            "essentia-tensorflow required for affect prediction. "
            "Install with: pip install essentia-tensorflow"
        ) from e

    required = ["TensorflowPredict2D", "TensorflowPredictMusiCNN"]
    missing = [name for name in required if not hasattr(es, name)]
    if missing:
        raise ImportError(f"essentia.standard missing required components: {missing}")


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
