from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import SAMPLE_RATE, load_audio

MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)
MINOR_DEGREES = (0, 2, 3, 5, 7, 8, 10)
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class TonalFeatures:
    chroma_entropy: float
    major_alignment: float
    minor_alignment: float


def extract_tonal(audio_path: Path, *, sr: int = SAMPLE_RATE) -> TonalFeatures:
    """Extract tonal features from audio file."""
    y, sr = load_audio(audio_path, sr=sr)
    return compute_tonal(y, sr)


def compute_tonal(y: np.ndarray, sr: int) -> TonalFeatures:
    """Compute tonal features from audio waveform."""
    chroma = _compute_chroma(y, sr)
    chroma_mean = _mean_chroma(chroma)
    chroma_entropy = _entropy(chroma_mean)
    major_alignment = _scale_alignment(chroma_mean, MAJOR_DEGREES)
    minor_alignment = _scale_alignment(chroma_mean, MINOR_DEGREES)

    return TonalFeatures(
        chroma_entropy=chroma_entropy,
        major_alignment=major_alignment,
        minor_alignment=minor_alignment,
    )


def _compute_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.asarray(chroma, dtype=np.float32)


def _mean_chroma(chroma: np.ndarray) -> np.ndarray:
    """Average chroma over time, returns 12-dim vector."""
    if chroma.size == 0:
        return np.zeros(12, dtype=np.float32)

    mean_vec = np.asarray(np.mean(chroma, axis=1), dtype=np.float32).reshape(-1)

    if mean_vec.size != 12:
        raise ValueError(f"unexpected chroma size: {mean_vec.size}")

    return mean_vec


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of distribution."""
    x = np.asarray(p, dtype=np.float64)
    total = float(np.sum(x))

    if total <= 0.0:
        return 0.0

    q = x / total
    q = np.clip(q, EPS, 1.0)

    return float(-np.sum(q * np.log2(q)))


def _scale_alignment(chroma_mean: np.ndarray, degrees: tuple[int, ...]) -> float:
    """Fraction of energy aligned with scale degrees."""
    x = np.asarray(chroma_mean, dtype=np.float64)
    total = float(np.sum(x))

    if total <= 0.0:
        return 0.0

    return float(np.sum(x[list(degrees)]) / total)
