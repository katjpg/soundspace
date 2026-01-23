from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .audio.io import load_audio
from .audio.rhythm import ESSENTIA_SR, compute_rhythm
from .audio.spectral import compute_spectral
from .audio.tonal import compute_tonal


def _init_worker() -> None:
    """Suppress essentia warnings in worker process."""
    import essentia
    essentia.log.warningActive = False


def _extract_single_track(args: tuple[str, str]) -> dict[str, Any]:
    """
    Extract all audio features for a single track.

    Args
    ----
        args (tuple[str, str]) : (audio_path, song_id) tuple for pickling.

    Returns
    -------
        (dict[str, Any]) : feature dictionary with song_id and all features.
    """
    audio_path, song_id = args
    path = Path(audio_path)

    y, sr = load_audio(path, sr=ESSENTIA_SR)

    rhythm = compute_rhythm(y, sr)
    spectral = compute_spectral(y, sr)
    tonal = compute_tonal(y, sr)

    return {
        "song_id": song_id,
        **asdict(rhythm),
        **asdict(spectral),
        **asdict(tonal),
    }


def extract_features_parallel(
    audio_paths: list[str],
    song_ids: list[str],
    *,
    n_workers: int | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract audio features from multiple tracks in parallel.

    Args
    ----
        audio_paths (list[str]) : paths to audio files.
        song_ids    (list[str]) : song identifiers (same order as audio_paths).
        n_workers    (int|None) : worker processes. (Default is CPU count).
        show_progress    (bool) : display tqdm progress bar. (Default is True).

    Returns
    -------
        (list[dict[str, Any]]) : feature dictionaries in input order.
    """
    if len(audio_paths) != len(song_ids):
        raise ValueError("audio_paths and song_ids must have same length")

    tasks = list(zip(audio_paths, song_ids))
    results: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_extract_single_track, task): task[1] for task in tasks}

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Extracting audio features")

        for future in iterator:
            song_id = futures[future]
            results[song_id] = future.result()

    # preserve input order
    return [results[sid] for sid in song_ids]
