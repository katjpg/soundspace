from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import load_audio


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

    def as_dict(self) -> dict[str, float]:
        """Return flat dict for CSV export."""
        return {k: float(v) for k, v in asdict(self).items()}


def extract_spectral(audio_path: Path, *, sr: int = 22050) -> SpectralFeatures:
    """Extract spectral features from audio file."""
    y, sr_loaded = load_audio(audio_path, sr=sr)
    return compute_spectral(y=y, sr=sr_loaded)


def compute_spectral(*, y: np.ndarray, sr: int) -> SpectralFeatures:
    """Compute spectral features from waveform."""
    mfcc_means = _compute_mfcc_means(y=y, sr=sr, n_mfcc=5)
    centroid_mean, centroid_std = _compute_centroid_stats(y=y, sr=sr)
    contrast_mean = _compute_contrast_mean(y=y, sr=sr)

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


def _compute_mfcc_means(*, y: np.ndarray, sr: int, n_mfcc: int) -> tuple[float, ...]:
    mfcc = librosa.feature.mfcc(y=y, sr=int(sr), n_mfcc=int(n_mfcc))
    means = np.asarray(np.mean(mfcc, axis=1), dtype=np.float32).reshape(-1)

    if means.size != int(n_mfcc):
        raise ValueError(f"unexpected MFCC shape: {mfcc.shape}")

    return tuple(float(v) for v in means)


def _compute_centroid_stats(*, y: np.ndarray, sr: int) -> tuple[float, float]:
    centroid = librosa.feature.spectral_centroid(y=y, sr=int(sr))
    values = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _compute_contrast_mean(*, y: np.ndarray, sr: int) -> float:
    contrast = librosa.feature.spectral_contrast(y=y, sr=int(sr))
    values = np.asarray(contrast, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))
