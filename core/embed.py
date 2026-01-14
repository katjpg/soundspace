from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from models.clap import ClapEmbedder


@dataclass(frozen=True, slots=True)
class AudioEmbedding:
    """Container for track ID and its corresponding CLAP embedding vector."""
    track_id: str
    embedding: np.ndarray
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "embedding": self.embedding.tolist(),
        }


def extract_clap_embedding(
    audio_path: Path,
    embedder: ClapEmbedder,
    *,
    track_id: str | None = None,
) -> AudioEmbedding:
    """
    Extract 512-dim CLAP embedding from single audio file.
    
    Pipeline:
    1. Load audio at 48kHz (CLAP's native sample rate)
    2. Process audio through ClapProcessor (mel-spectrogram + feature fusion)
    3. Extract embedding via model.get_audio_features()
    4. Return L2-normalized 512-dim vector
    """
    audio_waveform, _ = librosa.load(audio_path, sr=embedder.sample_rate, mono=True)
    
    processor_inputs = embedder.processor(
        audio=[audio_waveform],
        sampling_rate=embedder.sample_rate
    )
    
    model_inputs = _convert_to_tensors(processor_inputs, embedder.device)
    
    with torch.no_grad():
        audio_features = embedder.model.get_audio_features(**model_inputs)
    
    embedding_vector = audio_features.cpu().numpy()[0]
    
    if track_id is None:
        track_id = audio_path.stem
    
    return AudioEmbedding(track_id=track_id, embedding=embedding_vector)


def extract_clap_embeddings_batch(
    audio_paths: Sequence[Path],
    embedder: ClapEmbedder,
    *,
    track_ids: Sequence[str] | None = None,
    batch_size: int = 8,
) -> list[AudioEmbedding]:
    """
    Extract CLAP embeddings for multiple audio files in batches.
    
    Processes files in chunks of batch_size, skipping files that fail to load.
    Returns embeddings in same order as input paths (excluding failed files).
    """
    if track_ids is None:
        track_ids = [path.stem for path in audio_paths]
    
    if len(audio_paths) != len(track_ids):
        raise ValueError(
            f"Length mismatch: {len(audio_paths)} audio_paths, {len(track_ids)} track_ids"
        )
    
    all_embeddings: list[AudioEmbedding] = []
    
    for batch_start in range(0, len(audio_paths), batch_size):
        batch_end = batch_start + batch_size
        batch_paths = audio_paths[batch_start:batch_end]
        batch_ids = track_ids[batch_start:batch_end]
        
        batch_embeddings = _process_batch(batch_paths, batch_ids, embedder)
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def _process_batch(
    paths: Sequence[Path],
    track_ids: Sequence[str],
    embedder: ClapEmbedder,
) -> list[AudioEmbedding]:
    """Load audio files, extract embeddings for valid files only."""
    audio_waveforms: list[np.ndarray] = []
    valid_track_ids: list[str] = []
    
    for path, tid in zip(paths, track_ids):
        try:
            waveform, _ = librosa.load(path, sr=embedder.sample_rate, mono=True)
            audio_waveforms.append(waveform)
            valid_track_ids.append(tid)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
    
    if len(audio_waveforms) == 0:
        return []
    
    processor_inputs = embedder.processor(
        audio=audio_waveforms,
        sampling_rate=embedder.sample_rate
    )
    
    model_inputs = _convert_to_tensors(processor_inputs, embedder.device)
    
    with torch.no_grad():
        audio_features = embedder.model.get_audio_features(**model_inputs)
    
    embedding_matrix = audio_features.cpu().numpy()
    
    embeddings: list[AudioEmbedding] = []
    for tid, emb_vector in zip(valid_track_ids, embedding_matrix):
        embeddings.append(AudioEmbedding(track_id=tid, embedding=emb_vector))
    
    return embeddings


def _convert_to_tensors(
    processor_output: dict[str, Any],
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Convert ClapProcessor output (numpy arrays or lists) to PyTorch tensors.
    
    Handles both numpy arrays and nested lists from processor.
    Moves all tensors to specified device (CPU or CUDA).
    """
    tensors: dict[str, torch.Tensor] = {}
    
    for key, value in processor_output.items():
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            array = np.array(value)
            tensor = torch.from_numpy(array)
        
        tensors[key] = tensor.to(device)
    
    return tensors


def save_embeddings(embeddings: Sequence[AudioEmbedding], output_path: Path) -> None:
    """
    Save embeddings to compressed numpy archive (.npz).
    
    Format:
    - track_ids: array of strings
    - embeddings: (n_tracks, 512) float32 matrix
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    track_ids = [emb.track_id for emb in embeddings]
    embedding_matrix = np.vstack([emb.embedding for emb in embeddings])
    
    np.savez_compressed(
        output_path,
        track_ids=track_ids,
        embeddings=embedding_matrix,
    )


def load_embeddings(input_path: Path) -> list[AudioEmbedding]:
    """Load embeddings from .npz archive created by save_embeddings()."""
    data = np.load(input_path, allow_pickle=False)
    
    track_ids = data["track_ids"]
    embedding_matrix = data["embeddings"]
    
    embeddings: list[AudioEmbedding] = []
    for tid, emb_vector in zip(track_ids, embedding_matrix):
        embeddings.append(AudioEmbedding(track_id=str(tid), embedding=emb_vector))
    
    return embeddings

def center_embeddings(embeddings: Sequence[AudioEmbedding]) -> list[AudioEmbedding]:
    """
    Center embeddings by subtracting mean vector.
    
    Reduces anisotropy by removing dominant direction.
    Centered embeddings still support cosine similarity comparisons.
    """
    emb_matrix = np.vstack([e.embedding for e in embeddings])
    
    mean_vector = emb_matrix.mean(axis=0)
    centered_matrix = emb_matrix - mean_vector
    
    norms = np.linalg.norm(centered_matrix, axis=1, keepdims=True)
    normalized_matrix = centered_matrix / norms
    
    centered_embeddings: list[AudioEmbedding] = []
    for emb, centered_vec in zip(embeddings, normalized_matrix):
        centered_embeddings.append(
            AudioEmbedding(track_id=emb.track_id, embedding=centered_vec)
        )
    
    return centered_embeddings
