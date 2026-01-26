from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RetrievalIndex:
    """Retrieval-ready embedding index with centered, L2-normalized vectors."""

    track_ids: NDArray
    embeddings: NDArray[np.floating]
    mean: NDArray[np.floating]

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]

    def save(self, path: Path) -> None:
        """Write index to compressed .npz archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            track_ids=self.track_ids,
            embeddings=self.embeddings,
            mean=self.mean,
        )

    @classmethod
    def load(cls, path: Path) -> "RetrievalIndex":
        """Read index from .npz archive."""
        data = np.load(path, allow_pickle=False)
        return cls(
            track_ids=data["track_ids"],
            embeddings=data["embeddings"],
            mean=data["mean"],
        )


def build_retrieval_index(
    embeddings: NDArray[np.floating],
    track_ids: NDArray,
) -> RetrievalIndex:
    """
    Build a retrieval index by centering and L2-normalizing embeddings.

    Subtracts the database mean to break cone geometry, then normalizes
    each vector to unit length so dot product equals cosine similarity.
    
    """
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if float(norms.min()) <= 0.0:
        raise ValueError("zero-norm vector after centering")

    normalized = (centered / norms).astype(np.float32)

    return RetrievalIndex(
        track_ids=np.asarray(track_ids),
        embeddings=normalized,
        mean=mean.astype(np.float32),
    )
