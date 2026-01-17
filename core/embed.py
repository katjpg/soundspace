from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import librosa
import numpy as np
import torch

from models.clap import ClapEmbedder

CLAP_BATCH_SIZE = 8

EmbeddingList: TypeAlias = list["AudioEmbedding"]
ProcessorTensors: TypeAlias = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AudioEmbedding:
    """Single track's CLAP embedding + metadata."""

    track_id: str
    embedding: np.ndarray

    def __str__(self) -> str:
        dim = len(self.embedding)
        norm = float(np.linalg.norm(self.embedding))
        return f"{self.track_id} (dim={dim}, norm={norm:.3f})"

    def __len__(self) -> int:
        return len(self.embedding)

    @property
    def dim(self) -> int:
        return len(self.embedding)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.embedding))

    def dot_product(self, other: "AudioEmbedding") -> float:
        """Dot product, equals cosine sim if both L2-normalized."""
        return float(self.embedding @ other.embedding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "embedding": self.embedding.tolist(),
        }


def extract(
    audio_paths: Path | Sequence[Path],
    embedder: ClapEmbedder,
    *,
    track_ids: str | Sequence[str] | None = None,
    batch_size: int = CLAP_BATCH_SIZE,
) -> EmbeddingList:
    """
    Extract CLAP embeddings from audio file(s).

    Args:
        audio_paths (Path | Sequence[Path]) : single path or list of paths to audio files.
        embedder      (ClapEmbedder)        : loaded CLAP model.
        track_ids     (str | Sequence[str]) : optional track IDs, defaults to filenames.
        batch_size             (int)        : num files to process per batch.
                                              (Default is 8).

    Returns:
        (list[AudioEmbedding]) : one embedding per audio file.
    """
    if isinstance(audio_paths, Path):
        audio_paths = [audio_paths]
    if isinstance(track_ids, str):
        track_ids = [track_ids]
    if track_ids is None:
        track_ids = [path.stem for path in audio_paths]

    if len(audio_paths) != len(track_ids):
        raise ValueError(
            f"Length mismatch: {len(audio_paths)} paths, {len(track_ids)} ids"
        )

    all_embeddings: EmbeddingList = []

    for batch_start in range(0, len(audio_paths), batch_size):
        batch_end = batch_start + batch_size
        batch_paths = audio_paths[batch_start:batch_end]
        batch_ids = track_ids[batch_start:batch_end]

        audio_waveforms: list[np.ndarray] = []
        valid_track_ids: list[str] = []

        for path, tid in zip(batch_paths, batch_ids, strict=True):
            try:
                waveform, _ = librosa.load(
                    path, sr=embedder.sample_rate, mono=True
                )
                audio_waveforms.append(waveform)
                valid_track_ids.append(tid)
            except Exception as e:
                raise RuntimeError(f"Failed to load {path}: {e}") from e

        if len(audio_waveforms) == 0:
            continue

        processor_inputs = embedder.processor(
            audios=audio_waveforms, sampling_rate=embedder.sample_rate
        )
        model_inputs = _convert_to_tensors(processor_inputs, embedder.device)

        with torch.no_grad():
            audio_features = embedder.model.get_audio_features(**model_inputs)

        embedding_matrix = audio_features.cpu().numpy()

        for tid, emb_vector in zip(valid_track_ids, embedding_matrix, strict=True):
            all_embeddings.append(
                AudioEmbedding(track_id=tid, embedding=emb_vector)
            )

    return all_embeddings


def load_embeddings(input_path: Path) -> EmbeddingList:
    """Load embeddings from .npz file."""
    data = np.load(input_path, allow_pickle=False)
    track_ids = data["track_ids"]
    embedding_matrix = data["embeddings"]

    return [
        AudioEmbedding(track_id=str(tid), embedding=emb_vector)
        for tid, emb_vector in zip(track_ids, embedding_matrix, strict=True)
    ]


def save_embeddings(
    embeddings: Sequence[AudioEmbedding],
    output_path: Path,
) -> None:
    """Save embeddings to .npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_ids = [emb.track_id for emb in embeddings]
    embedding_matrix = np.vstack([emb.embedding for emb in embeddings])

    np.savez_compressed(
        output_path,
        track_ids=track_ids,
        embeddings=embedding_matrix,
    )


def center_and_normalize(
    embeddings: Sequence[AudioEmbedding],
) -> EmbeddingList:
    """
    Center at origin then L2-normalize each vector.

    Breaks cone geometry in CLAP embeddings to improve isotropy.

    Args:
        embeddings (Sequence[AudioEmbedding]) : list of raw embeddings.

    Returns:
        (list[AudioEmbedding]) : centered + normalized embeddings.
    """
    emb_matrix = np.vstack([e.embedding for e in embeddings])
    mean_vector = emb_matrix.mean(axis=0)
    centered_matrix = emb_matrix - mean_vector

    norms = np.linalg.norm(centered_matrix, axis=1, keepdims=True)
    if float(norms.min()) <= 0.0:
        raise ValueError("zero norm vector after centering")

    normalized_matrix = centered_matrix / norms

    return [
        AudioEmbedding(track_id=emb.track_id, embedding=centered_vec)
        for emb, centered_vec in zip(embeddings, normalized_matrix, strict=True)
    ]


def _convert_to_tensors(
    processor_output: dict[str, Any],
    device: str,
) -> ProcessorTensors:
    """Convert HuggingFace processor output to torch tensors on device."""
    tensors: ProcessorTensors = {}
    for key, value in processor_output.items():
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            array = np.array(value)
            tensor = torch.from_numpy(array)
        tensors[key] = tensor.to(device)
    return tensors
