from pathlib import Path

import librosa
import numpy as np


AudioData = tuple[np.ndarray, int]


def load_audio(audio_path: Path, *, sr: int = 22050) -> AudioData:
    """Load mono audio as (y, sr)."""
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    y, sr_loaded = librosa.load(str(audio_path), sr=int(sr), mono=True)
    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        raise ValueError(f"empty audio: {audio_path}")

    return y, int(sr_loaded)
