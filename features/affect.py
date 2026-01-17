from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from models.essentia import AffectPredictor

VA_SCALE_MIN = 1.0
VA_SCALE_MAX = 9.0
VA_SCALE_CENTER = 5.0
VA_SCALE_RANGE = 4.0


@dataclass(frozen=True, slots=True)
class AffectFeatures:
    valence_1_9: float
    arousal_1_9: float
    valence_m1_1: float
    arousal_m1_1: float


def extract_affect(audio_path: Path, model: AffectPredictor) -> AffectFeatures:
    """Extract valence-arousal features from audio file."""
    y, sr = librosa.load(str(audio_path), sr=model.sample_rate, mono=True)
    return compute_affect(y, int(sr), model)


def compute_affect(y: np.ndarray, sr: int, model: AffectPredictor) -> AffectFeatures:
    """Compute valence-arousal from audio waveform using affect model."""
    if sr != model.sample_rate:
        raise ValueError(
            f"sample rate mismatch: expected {model.sample_rate}, got {sr}"
        )

    y = np.asarray(y, dtype=np.float32)

    emb_raw = model.emb_predictor(y)
    emb_pooled = _pool_embedding(emb_raw)

    va_19 = model.va_predictor(emb_pooled)
    va_19 = np.asarray(va_19, dtype=np.float32)

    if va_19.ndim != 2 or va_19.shape[1] < 2:
        raise ValueError(f"unexpected VA output shape: {va_19.shape}")

    if va_19.shape[0] > 1:
        va_19 = va_19.mean(axis=0, keepdims=True).astype(np.float32)

    valence_19 = float(va_19[0, 0])
    arousal_19 = float(va_19[0, 1])

    valence_m11 = _scale_to_m1_1(valence_19)
    arousal_m11 = _scale_to_m1_1(arousal_19)

    return AffectFeatures(
        valence_1_9=valence_19,
        arousal_1_9=arousal_19,
        valence_m1_1=valence_m11,
        arousal_m1_1=arousal_m11,
    )


def _pool_embedding(emb_raw: np.ndarray) -> np.ndarray:
    """Prepare embedding for VA predictor (no temporal pooling)."""
    emb = np.asarray(emb_raw, dtype=np.float32)

    if emb.ndim == 1:
        return emb.reshape(1, -1)

    if emb.ndim == 2:
        return emb.astype(np.float32)

    raise ValueError(f"unexpected embedding shape: {emb.shape}")


def _scale_to_m1_1(x_19: float) -> float:
    """Convert 1-9 scale to [-1, 1] scale."""
    return (x_19 - VA_SCALE_CENTER) / VA_SCALE_RANGE
