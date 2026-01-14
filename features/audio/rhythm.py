from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import load_audio


@dataclass(frozen=True, slots=True)
class RhythmFeatures:
    tempo_bpm: float
    onset_strength_mean: float
    onset_strength_std: float
    
    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def extract_rhythm(audio_path: Path, *, sr: int = 22050) -> RhythmFeatures:
    y, sr_loaded = load_audio(audio_path, sr=sr)
    return compute_rhythm(y=y, sr=sr_loaded)


def compute_rhythm(*, y: np.ndarray, sr: int) -> RhythmFeatures:
    tempo_bpm = _estimate_tempo(y=y, sr=sr)
    onset_env = _compute_onset_strength(y=y, sr=sr)
    onset_mean, onset_std = _mean_std(onset_env)
    
    return RhythmFeatures(
        tempo_bpm=tempo_bpm,
        onset_strength_mean=onset_mean,
        onset_strength_std=onset_std,
    )


def _estimate_tempo(*, y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=int(sr))
    return float(tempo)


def _compute_onset_strength(*, y: np.ndarray, sr: int) -> np.ndarray:
    onset = librosa.onset.onset_strength(y=y, sr=int(sr))
    return np.asarray(onset, dtype=np.float32)


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))
