from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import load_audio


_MAJOR_DEGREES: tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)
_MINOR_DEGREES: tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)
_EPS: float = 1e-12


@dataclass(frozen=True, slots=True)
class TonalFeatures:
    chroma_entropy: float
    major_alignment: float
    minor_alignment: float
    
    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def extract_tonal(audio_path: Path, *, sr: int = 22050) -> TonalFeatures:
    y, sr_loaded = load_audio(audio_path, sr=sr)
    return compute_tonal(y=y, sr=sr_loaded)


def compute_tonal(*, y: np.ndarray, sr: int) -> TonalFeatures:
    chroma = _compute_chroma(y=y, sr=sr)
    chroma_mean = _mean_chroma(chroma)
    
    chroma_entropy = _entropy(chroma_mean)
    major_alignment = _scale_alignment(chroma_mean, degrees=_MAJOR_DEGREES)
    minor_alignment = _scale_alignment(chroma_mean, degrees=_MINOR_DEGREES)
    
    return TonalFeatures(
        chroma_entropy=chroma_entropy,
        major_alignment=major_alignment,
        minor_alignment=minor_alignment,
    )


def _compute_chroma(*, y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=y, sr=int(sr))
    return np.asarray(chroma, dtype=np.float32)


def _mean_chroma(chroma: np.ndarray) -> np.ndarray:
    if chroma.size == 0:
        return np.zeros((12,), dtype=np.float32)
    
    mean_vec = np.asarray(np.mean(chroma, axis=1), dtype=np.float32).reshape(-1)
    
    if mean_vec.size != 12:
        raise ValueError(f"unexpected chroma size: {mean_vec.size}")
    
    return mean_vec


def _entropy(p: np.ndarray) -> float:
    x = np.asarray(p, dtype=np.float64)
    total = float(np.sum(x))
    
    if total <= 0.0:
        return 0.0
    
    q = x / total
    q = np.clip(q, _EPS, 1.0)
    
    return float(-np.sum(q * np.log2(q)))


def _scale_alignment(chroma_mean: np.ndarray, *, degrees: tuple[int, ...]) -> float:
    x = np.asarray(chroma_mean, dtype=np.float64)
    total = float(np.sum(x))
    
    if total <= 0.0:
        return 0.0
    
    return float(np.sum(x[list(degrees)]) / total)
