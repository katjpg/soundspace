from pathlib import Path

import librosa
import numpy as np
import torch


def preprocess_query(embedding: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Center by database mean and L2-normalize."""
    centered = embedding - mean
    return (centered / np.linalg.norm(centered)).astype(np.float32)


def embed_audio(
    audio_path: Path,
    model: torch.nn.Module,
    processor,
    device: str = "cpu",
) -> np.ndarray:
    """Embed audio file using CLAP model."""
    waveform, _ = librosa.load(audio_path, sr=48000, mono=True)
    inputs = processor(audios=[waveform], sampling_rate=48000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_audio_features(**inputs)

    return features.cpu().numpy().squeeze()


def retrieve_top_k(
    query: np.ndarray,
    embeddings: np.ndarray,
    track_ids: np.ndarray,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k tracks by cosine similarity."""
    scores = embeddings @ query

    # argpartition for O(n) candidate selection -> sort top-k
    k = min(k, len(scores))
    candidate_idx = np.argpartition(scores, -k)[-k:]
    candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

    return [(str(track_ids[i]), float(scores[i])) for i in candidate_idx]
