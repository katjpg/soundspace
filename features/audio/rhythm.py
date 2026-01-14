from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import SAMPLE_RATE, AudioData, load_audio


@dataclass(frozen=True, slots=True)
class RhythmFeatures:
    tempo_bpm: float
    onset_strength_mean: float
    onset_strength_std: float


def extract_rhythm(audio_path: Path, *, sr: int = SAMPLE_RATE) -> RhythmFeatures:
    """Extract rhythm features from audio file."""
    y, sr = load_audio(audio_path, sr=sr)
    return compute_rhythm(y, sr)


def compute_rhythm(y: np.ndarray, sr: int) -> RhythmFeatures:
    """Compute rhythm features from audio waveform."""
    tempo_bpm = _estimate_tempo(y, sr)
    onset_env = _compute_onset_strength(y, sr)
    onset_mean, onset_std = _mean_std(onset_env)
    
    return RhythmFeatures(
        tempo_bpm=tempo_bpm,
        onset_strength_mean=onset_mean,
        onset_strength_std=onset_std,
    )


def _estimate_tempo(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def _compute_onset_strength(y: np.ndarray, sr: int) -> np.ndarray:
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    return np.asarray(onset, dtype=np.float32)


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    """Mean and std of array, returns (0, 0) if empty."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))
