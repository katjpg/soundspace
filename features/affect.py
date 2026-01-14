from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from models.essentia import AffectPredictor


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AffectFeatures:
    valence_1_9: float
    arousal_1_9: float
    valence_m1_1: float
    arousal_m1_1: float
    
    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def extract_affect(audio_path: Path, model: AffectPredictor) -> AffectFeatures:
    import librosa
    
    y, sr_loaded = librosa.load(audio_path, sr=model.sample_rate, mono=True)
    return compute_affect(y=y, sr=int(sr_loaded), model=model)


def compute_affect(*, y: FloatArray, sr: int, model: AffectPredictor) -> AffectFeatures:
    if sr != model.sample_rate:
        raise ValueError(f"sample rate mismatch: expected {model.sample_rate}, got {sr}")
    
    y_f32 = np.asarray(y, dtype=np.float32)
    
    # embedding extraction
    emb_raw = model.emb_predictor(y_f32)
    emb_pooled = _pool_embedding(emb_raw)
    
    # valence-arousal prediction (1-9 scale)
    va_19 = model.va_predictor(emb_pooled)
    va_19 = np.asarray(va_19, dtype=np.float32)
    
    if va_19.ndim != 2 or va_19.shape[1] < 2:
        raise ValueError(f"unexpected VA output shape: {va_19.shape}")
    
    valence_19 = float(va_19[0, 0])
    arousal_19 = float(va_19[0, 1])
    
    # normalize to [-1, 1]
    valence_m11 = _to_m1_1(valence_19)
    arousal_m11 = _to_m1_1(arousal_19)
    
    return AffectFeatures(
        valence_1_9=valence_19,
        arousal_1_9=arousal_19,
        valence_m1_1=valence_m11,
        arousal_m1_1=arousal_m11,
    )


def _pool_embedding(emb_raw: FloatArray) -> FloatArray:
    # handle both 1D and 2D embeddings (temporal dimension)
    emb = np.asarray(emb_raw, dtype=np.float32)
    
    if emb.ndim == 1:
        return emb.reshape(1, -1)
    if emb.ndim == 2:
        return emb.mean(axis=0, keepdims=True).astype(np.float32)
    
    raise ValueError(f"unexpected embedding shape: {emb.shape}")


def _to_m1_1(x_19: float) -> float:
    # convert 1-9 scale to [-1, 1]
    return (x_19 - 5.0) / 4.0
