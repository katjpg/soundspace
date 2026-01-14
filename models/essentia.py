from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from configs.models import ModelConfig


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AffectPredictor:
    emb_predictor: Any
    va_predictor: Any
    sample_rate: int


def from_pretrained(
    config: ModelConfig,
    model_name: str = "musicnn",
    sample_rate: int = 16000,
) -> AffectPredictor:
    """
    Load encoder and VA predictor from config.
    
    Args:
        config: Model configuration w/ paths to weights and metadata.
        model_name: Name of model in config (default: "musicnn").
        sample_rate: Audio sample rate for inference (default: 16000).
    """
    _check_essentia_available()
    
    if model_name not in config.models:
        available: list[str] = sorted(config.models.keys())
        raise ValueError(f"unknown model {model_name!r}. available: {available}")
    
    model = config.models[model_name]
    
    if model.encoder is None:
        raise ValueError(f"model {model_name!r} has no encoder spec")
    if model.predictor is None:
        raise ValueError(f"model {model_name!r} has no predictor spec")
    
    # verify model files exist
    if model.encoder.weights is None or not model.encoder.weights.exists():
        raise FileNotFoundError(f"encoder weights not found: {model.encoder.weights}")
    if model.predictor.weights is None or not model.predictor.weights.exists():
        raise FileNotFoundError(f"predictor weights not found: {model.predictor.weights}")
    
    import essentia.standard as es  # type: ignore
    
    emb_pred = es.TensorflowPredictMusiCNN(  # type: ignore
        graphFilename=str(model.encoder.weights),
        output=model.encoder.output,
    )
    
    va_pred = es.TensorflowPredict2D(  # type: ignore
        graphFilename=str(model.predictor.weights),
        output=model.predictor.output,
    )
    
    return AffectPredictor(
        emb_predictor=emb_pred,
        va_predictor=va_pred,
        sample_rate=sample_rate,
    )


def _check_essentia_available() -> None:
    try:
        import essentia.standard as es  # type: ignore
    except ImportError as e:
        raise ImportError(
            "essentia-tensorflow required for affect prediction. "
            "install: pip install essentia-tensorflow"
        ) from e
    
    required = ["TensorflowPredict2D", "TensorflowPredictMusiCNN"]
    missing = [name for name in required if not hasattr(es, name)]
    if missing:
        raise ImportError(f"essentia.standard missing required components: {missing}")
