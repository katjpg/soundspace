from pathlib import Path
from typing import TypeAlias

import librosa
import numpy as np

SAMPLE_RATE = 22050

AudioData: TypeAlias = tuple[np.ndarray, int]


def load_audio(audio_path: Path, *, sr: int = SAMPLE_RATE) -> AudioData:
    """Load audio file as mono, resampled to target sample rate."""
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        raise ValueError(f"empty audio: {audio_path}")

    return y, int(sr_loaded)
