from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .io import load_audio

ESSENTIA_SR = 44100
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RhythmFeatures:
    tempo_bpm: float
    onset_strength_mean: float
    onset_strength_std: float
    beat_interval_mean: float
    beat_interval_std: float
    beat_interval_cv: float
    bpm_first_peak: float
    bpm_first_weight: float
    bpm_first_spread: float
    bpm_second_peak: float
    bpm_second_weight: float
    bpm_second_spread: float
    bpm_peak_ratio: float
    bpm_histogram_entropy: float
    rhythm_transform_mean: float
    rhythm_transform_std: float
    rhythm_transform_max: float
    rhythm_transform_entropy: float
    beats_loudness_mean: float
    beats_loudness_std: float
    beats_low_ratio: float
    beats_mid_ratio: float
    beats_high_ratio: float


def extract_rhythm(audio_path: Path) -> RhythmFeatures:
    """Extract rhythm features from audio file."""
    y, sr = load_audio(audio_path, sr=ESSENTIA_SR)
    return compute_rhythm(y, sr)


def compute_rhythm(y: np.ndarray, sr: int) -> RhythmFeatures:
    """Compute rhythm features from audio waveform."""
    tempo_bpm = _estimate_tempo(y, sr)
    onset_mean, onset_std = _compute_onset_stats(y, sr)

    beat_intervals = _extract_beat_intervals(y, sr)
    bpm_histogram = _extract_bpm_histogram(y, sr)
    rhythm_transform = _extract_rhythm_transform(y, sr)
    beats_loudness = _extract_beats_loudness(y, sr)

    return RhythmFeatures(
        tempo_bpm=tempo_bpm,
        onset_strength_mean=onset_mean,
        onset_strength_std=onset_std,
        beat_interval_mean=beat_intervals["beat_interval_mean"],
        beat_interval_std=beat_intervals["beat_interval_std"],
        beat_interval_cv=beat_intervals["beat_interval_cv"],
        bpm_first_peak=bpm_histogram["bpm_first_peak"],
        bpm_first_weight=bpm_histogram["bpm_first_weight"],
        bpm_first_spread=bpm_histogram["bpm_first_spread"],
        bpm_second_peak=bpm_histogram["bpm_second_peak"],
        bpm_second_weight=bpm_histogram["bpm_second_weight"],
        bpm_second_spread=bpm_histogram["bpm_second_spread"],
        bpm_peak_ratio=bpm_histogram["bpm_peak_ratio"],
        bpm_histogram_entropy=bpm_histogram["bpm_histogram_entropy"],
        rhythm_transform_mean=rhythm_transform["rhythm_transform_mean"],
        rhythm_transform_std=rhythm_transform["rhythm_transform_std"],
        rhythm_transform_max=rhythm_transform["rhythm_transform_max"],
        rhythm_transform_entropy=rhythm_transform["rhythm_transform_entropy"],
        beats_loudness_mean=beats_loudness["beats_loudness_mean"],
        beats_loudness_std=beats_loudness["beats_loudness_std"],
        beats_low_ratio=beats_loudness["beats_low_ratio"],
        beats_mid_ratio=beats_loudness["beats_mid_ratio"],
        beats_high_ratio=beats_loudness["beats_high_ratio"],
    )


def _estimate_tempo(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def _compute_onset_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Compute mean and std of onset strength envelope."""
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    x = np.asarray(onset, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def _extract_beat_intervals(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract beat interval statistics using essentia RhythmExtractor2013."""
    defaults = {
        "beat_interval_mean": 0.0,
        "beat_interval_std": 0.0,
        "beat_interval_cv": 0.0,
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

        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        if len(beats_intervals) < 2:
            return defaults

        intervals = np.array(beats_intervals, dtype=np.float32)
        mean_val = float(np.mean(intervals))
        std_val = float(np.std(intervals))
        cv_val = std_val / mean_val if mean_val > 0 else 0.0

        return {
            "beat_interval_mean": mean_val,
            "beat_interval_std": std_val,
            "beat_interval_cv": cv_val,
        }

    except Exception:
        return defaults


def _extract_bpm_histogram(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract BPM histogram features using essentia BpmHistogramDescriptors."""
    defaults = {
        "bpm_first_peak": 0.0,
        "bpm_first_weight": 0.0,
        "bpm_first_spread": 0.0,
        "bpm_second_peak": 0.0,
        "bpm_second_weight": 0.0,
        "bpm_second_spread": 0.0,
        "bpm_peak_ratio": 0.0,
        "bpm_histogram_entropy": 0.0,
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

        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        if len(beats_intervals) < 2:
            return defaults

        bpm_histogram_desc = es.BpmHistogramDescriptors()
        (
            first_peak_bpm,
            first_peak_weight,
            first_peak_spread,
            second_peak_bpm,
            second_peak_weight,
            second_peak_spread,
            histogram,
        ) = bpm_histogram_desc(np.array(beats_intervals, dtype=np.float32))

        peak_ratio = 0.0
        if first_peak_weight > 0:
            peak_ratio = second_peak_weight / first_peak_weight

        histogram_arr = np.array(histogram, dtype=np.float32)
        total = float(np.sum(histogram_arr))
        if total > 0:
            p = histogram_arr / total
            p = np.clip(p, EPS, 1.0)
            entropy_val = float(-np.sum(p * np.log2(p)))
        else:
            entropy_val = 0.0

        return {
            "bpm_first_peak": float(first_peak_bpm),
            "bpm_first_weight": float(first_peak_weight),
            "bpm_first_spread": float(first_peak_spread),
            "bpm_second_peak": float(second_peak_bpm),
            "bpm_second_weight": float(second_peak_weight),
            "bpm_second_spread": float(second_peak_spread),
            "bpm_peak_ratio": peak_ratio,
            "bpm_histogram_entropy": entropy_val,
        }

    except Exception:
        return defaults


def _extract_rhythm_transform(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract rhythm transform features using essentia MelBands + RhythmTransform."""
    defaults = {
        "rhythm_transform_mean": 0.0,
        "rhythm_transform_std": 0.0,
        "rhythm_transform_max": 0.0,
        "rhythm_transform_entropy": 0.0,
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

        frame_size = 2048
        hop_size = 1024
        mel_frames = []

        windowing = es.Windowing(type="hann", size=frame_size)
        spectrum = es.Spectrum(size=frame_size)
        mel_bands = es.MelBands(
            sampleRate=ESSENTIA_SR,
            numberBands=40,
            lowFrequencyBound=0,
            highFrequencyBound=ESSENTIA_SR / 2,
        )

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            windowed = windowing(frame)
            spec = spectrum(windowed)
            mels = mel_bands(spec)
            mel_frames.append(mels)

        if len(mel_frames) < 8:
            return defaults

        mel_matrix = np.array(mel_frames, dtype=np.float32)

        rhythm_transform = es.RhythmTransform(frameSize=8, hopSize=4)
        rt = rhythm_transform(mel_matrix)

        if rt.size == 0:
            return defaults

        rt_flat = rt.flatten()
        mean_val = float(np.mean(rt_flat))
        std_val = float(np.std(rt_flat))
        max_val = float(np.max(rt_flat))

        total = float(np.sum(np.abs(rt_flat)))
        if total > 0:
            p = np.abs(rt_flat) / total
            p = np.clip(p, EPS, 1.0)
            entropy_val = float(-np.sum(p * np.log2(p)))
        else:
            entropy_val = 0.0

        return {
            "rhythm_transform_mean": mean_val,
            "rhythm_transform_std": std_val,
            "rhythm_transform_max": max_val,
            "rhythm_transform_entropy": entropy_val,
        }

    except Exception:
        return defaults


def _extract_beats_loudness(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract beats loudness features using essentia BeatsLoudness."""
    defaults = {
        "beats_loudness_mean": 0.0,
        "beats_loudness_std": 0.0,
        "beats_low_ratio": 0.0,
        "beats_mid_ratio": 0.0,
        "beats_high_ratio": 0.0,
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

        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        if len(beats) < 2:
            return defaults

        beats_loudness = es.BeatsLoudness(
            sampleRate=ESSENTIA_SR,
            frequencyBands=[20, 150, 400, 3200, 7000, ESSENTIA_SR / 2],
        )
        loudness, loudness_band_ratio = beats_loudness(audio, beats)

        if len(loudness) == 0:
            return defaults

        loudness_arr = np.array(loudness, dtype=np.float32)
        mean_val = float(np.mean(loudness_arr))
        std_val = float(np.std(loudness_arr))

        band_ratio_arr = np.array(loudness_band_ratio, dtype=np.float32)
        if band_ratio_arr.ndim == 2 and band_ratio_arr.shape[1] >= 5:
            low_ratio = float(np.mean(band_ratio_arr[:, 0] + band_ratio_arr[:, 1]))
            mid_ratio = float(np.mean(band_ratio_arr[:, 2] + band_ratio_arr[:, 3]))
            high_ratio = float(np.mean(band_ratio_arr[:, 4]))
        else:
            low_ratio = 0.0
            mid_ratio = 0.0
            high_ratio = 0.0

        return {
            "beats_loudness_mean": mean_val,
            "beats_loudness_std": std_val,
            "beats_low_ratio": low_ratio,
            "beats_mid_ratio": mid_ratio,
            "beats_high_ratio": high_ratio,
        }

    except Exception:
        return defaults
