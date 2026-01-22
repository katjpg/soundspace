from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import load_audio

ESSENTIA_SR = 44100
MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)
MINOR_DEGREES = (0, 2, 3, 5, 7, 8, 10)
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class TonalFeatures:
    chroma_entropy: float
    major_alignment: float
    minor_alignment: float
    hpcp_0: float
    hpcp_1: float
    hpcp_2: float
    hpcp_3: float
    hpcp_4: float
    hpcp_5: float
    hpcp_6: float
    hpcp_7: float
    hpcp_8: float
    hpcp_9: float
    hpcp_10: float
    hpcp_11: float
    hpcp_entropy: float
    hpcp_std: float
    hpcp_max: float
    hpcp_temporal_std: float
    key_strength: float
    is_minor: float
    key_cos: float
    key_sin: float


def extract_tonal(audio_path: Path, *, sr: int = ESSENTIA_SR) -> TonalFeatures:
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

    hpcp = _extract_hpcp(y, sr)
    key = _extract_key(y, sr)

    return TonalFeatures(
        chroma_entropy=chroma_entropy,
        major_alignment=major_alignment,
        minor_alignment=minor_alignment,
        hpcp_0=hpcp["hpcp_0"],
        hpcp_1=hpcp["hpcp_1"],
        hpcp_2=hpcp["hpcp_2"],
        hpcp_3=hpcp["hpcp_3"],
        hpcp_4=hpcp["hpcp_4"],
        hpcp_5=hpcp["hpcp_5"],
        hpcp_6=hpcp["hpcp_6"],
        hpcp_7=hpcp["hpcp_7"],
        hpcp_8=hpcp["hpcp_8"],
        hpcp_9=hpcp["hpcp_9"],
        hpcp_10=hpcp["hpcp_10"],
        hpcp_11=hpcp["hpcp_11"],
        hpcp_entropy=hpcp["hpcp_entropy"],
        hpcp_std=hpcp["hpcp_std"],
        hpcp_max=hpcp["hpcp_max"],
        hpcp_temporal_std=hpcp["hpcp_temporal_std"],
        key_strength=key["key_strength"],
        is_minor=key["is_minor"],
        key_cos=key["key_cos"],
        key_sin=key["key_sin"],
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


def _extract_hpcp(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract HPCP features using essentia SpectralPeaks + HPCP."""
    defaults = {
        "hpcp_0": 0.0,
        "hpcp_1": 0.0,
        "hpcp_2": 0.0,
        "hpcp_3": 0.0,
        "hpcp_4": 0.0,
        "hpcp_5": 0.0,
        "hpcp_6": 0.0,
        "hpcp_7": 0.0,
        "hpcp_8": 0.0,
        "hpcp_9": 0.0,
        "hpcp_10": 0.0,
        "hpcp_11": 0.0,
        "hpcp_entropy": 0.0,
        "hpcp_std": 0.0,
        "hpcp_max": 0.0,
        "hpcp_temporal_std": 0.0,
    }

    try:
        import essentia.standard as es
    except ImportError:
        return defaults

    try:
        audio = np.asarray(y, dtype=np.float32)
        if sr != ESSENTIA_SR:
            resampler = es.Resample(inputSampleRate=sr, outputSampleRate=ESSENTIA_SR)
            audio = resampler(audio)

        frame_size = 4096
        hop_size = 2048
        hpcp_frames = []

        windowing = es.Windowing(type="blackmanharris62", size=frame_size)
        spectrum = es.Spectrum(size=frame_size)
        spectral_peaks = es.SpectralPeaks(
            sampleRate=ESSENTIA_SR,
            maxPeaks=100,
            minFrequency=20,
            maxFrequency=5000,
            magnitudeThreshold=0.00001,
            orderBy="magnitude",
        )
        hpcp_algo = es.HPCP(
            size=12,
            sampleRate=ESSENTIA_SR,
            minFrequency=20,
            maxFrequency=5000,
            normalized="unitSum",
        )

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            windowed = windowing(frame)
            spec = spectrum(windowed)
            frequencies, magnitudes = spectral_peaks(spec)
            hpcp = hpcp_algo(frequencies, magnitudes)
            hpcp_frames.append(hpcp)

        if len(hpcp_frames) == 0:
            return defaults

        hpcp_matrix = np.array(hpcp_frames, dtype=np.float32)
        hpcp_mean = np.mean(hpcp_matrix, axis=0)
        hpcp_temporal_std_arr = np.std(hpcp_matrix, axis=0)

        total = float(np.sum(hpcp_mean))
        if total > 0:
            hpcp_normalized = hpcp_mean / total
            hpcp_normalized = np.clip(hpcp_normalized, EPS, 1.0)
            hpcp_entropy_val = float(-np.sum(hpcp_normalized * np.log2(hpcp_normalized)))
        else:
            hpcp_entropy_val = 0.0

        result = {}
        for i in range(12):
            result[f"hpcp_{i}"] = float(hpcp_mean[i])

        result["hpcp_entropy"] = hpcp_entropy_val
        result["hpcp_std"] = float(np.std(hpcp_mean))
        result["hpcp_max"] = float(np.max(hpcp_mean))
        result["hpcp_temporal_std"] = float(np.mean(hpcp_temporal_std_arr))

        return result

    except Exception:
        return defaults


def _extract_key(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract key features using essentia KeyExtractor."""
    defaults = {
        "key_strength": 0.0,
        "is_minor": 0.0,
        "key_cos": 0.0,
        "key_sin": 0.0,
    }

    try:
        import essentia.standard as es
    except ImportError:
        return defaults

    try:
        audio = np.asarray(y, dtype=np.float32)
        if sr != ESSENTIA_SR:
            resampler = es.Resample(inputSampleRate=sr, outputSampleRate=ESSENTIA_SR)
            audio = resampler(audio)

        key_extractor = es.KeyExtractor(sampleRate=ESSENTIA_SR)
        key, scale, strength = key_extractor(audio)

        key_to_index = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        }

        key_index = key_to_index.get(key, 0)
        angle = (key_index / 12.0) * 2 * np.pi
        is_minor_val = 1.0 if scale == "minor" else 0.0

        return {
            "key_strength": float(strength),
            "is_minor": is_minor_val,
            "key_cos": float(np.cos(angle)),
            "key_sin": float(np.sin(angle)),
        }

    except Exception:
        return defaults
