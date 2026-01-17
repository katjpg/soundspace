from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import SAMPLE_RATE, load_audio

N_MFCC = 5


@dataclass(frozen=True, slots=True)
class SpectralFeatures:
    mfcc_1_mean: float
    mfcc_2_mean: float
    mfcc_3_mean: float
    mfcc_4_mean: float
    mfcc_5_mean: float
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_contrast_mean: float


def extract_spectral(audio_path: Path, *, sr: int = SAMPLE_RATE) -> SpectralFeatures:
    """Extract spectral features from audio file."""
    y, sr = load_audio(audio_path, sr=sr)
    return compute_spectral(y, sr)


def compute_spectral(y: np.ndarray, sr: int) -> SpectralFeatures:
    """Compute spectral features from audio waveform."""
    mfcc_means = _compute_mfcc_means(y, sr, N_MFCC)
    centroid_mean, centroid_std = _compute_centroid_stats(y, sr)
    contrast_mean = _compute_contrast_mean(y, sr)

    return SpectralFeatures(
        mfcc_1_mean=mfcc_means[0],
        mfcc_2_mean=mfcc_means[1],
        mfcc_3_mean=mfcc_means[2],
        mfcc_4_mean=mfcc_means[3],
        mfcc_5_mean=mfcc_means[4],
        spectral_centroid_mean=centroid_mean,
        spectral_centroid_std=centroid_std,
        spectral_contrast_mean=contrast_mean,
    )


def _compute_mfcc_means(y: np.ndarray, sr: int, n_mfcc: int) -> tuple[float, ...]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    means = np.asarray(np.mean(mfcc, axis=1), dtype=np.float32).reshape(-1)

    if means.size != n_mfcc:
        raise ValueError(f"unexpected MFCC shape: {mfcc.shape}")

    return tuple(float(v) for v in means)


def _compute_centroid_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Mean and std of spectral centroid."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    values = np.asarray(centroid, dtype=np.float32).reshape(-1)

    if values.size == 0:
        return 0.0, 0.0

    return float(np.mean(values)), float(np.std(values))


def _compute_contrast_mean(y: np.ndarray, sr: int) -> float:
    """Mean spectral contrast across frequency bands."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    values = np.asarray(contrast, dtype=np.float32).reshape(-1)

    if values.size == 0:
        return 0.0

    return float(np.mean(values))
