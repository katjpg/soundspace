from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import librosa
import numpy as np
import torch

from models.clap import ClapEmbedder


CLAP_BATCH_SIZE = 8
KNN_NEIGHBORS = 15
PAIRWISE_SAMPLE_SIZE = 10000
PROBE_SAMPLE_SIZE = 200
RANDOM_SEED = 42


EmbeddingList: TypeAlias = list["AudioEmbedding"]
ValidationMetrics: TypeAlias = dict[str, Any]
SimilarityStats: TypeAlias = dict[str, float]
ProcessorTensors: TypeAlias = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AudioEmbedding:
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
    """Extract CLAP embeddings from audio file(s)."""
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
    """Center at origin then L2-normalize each vector."""
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


def validate_embeddings(
    embeddings: Sequence[AudioEmbedding],
    *,
    k: int = KNN_NEIGHBORS,
) -> ValidationMetrics:
    E = np.vstack([e.embedding for e in embeddings]).astype(np.float32)

    return {
        "validity": check_validity(E),
        "pairwise_cosine": sample_pairwise_cosine(E),
        "dimension_spread": measure_dimension_spread(E),
        "neighbor_gap": measure_neighbor_separation(E, k=k),
        "degrees": measure_graph_degrees(E, k=k),
    }


def check_validity(E: np.ndarray) -> SimilarityStats:
    """Check for NaN, inf, zero-norm, duplicate vectors."""
    nan_count = int(np.isnan(E).any(axis=1).sum())
    inf_count = int(np.isinf(E).any(axis=1).sum())

    norms = np.linalg.norm(E, axis=1)
    zero_count = int(np.sum(norms < 1e-9))

    unique_rows = np.unique(E, axis=0)
    duplicate_count = len(E) - len(unique_rows)

    return {
        "total": float(len(E)),
        "nan_count": float(nan_count),
        "inf_count": float(inf_count),
        "zero_count": float(zero_count),
        "duplicate_count": float(duplicate_count),
    }


def sample_pairwise_cosine(
    E: np.ndarray,
    n_samples: int = PAIRWISE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """Sample random pairs -> cosine sim stats."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]

    if n < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
        }

    n_actual = min(n_samples, (n * (n - 1)) // 2)
    sims = np.zeros(n_actual, dtype=np.float32)

    for idx in range(n_actual):
        i, j = rng.choice(n, size=2, replace=False)
        sims[idx] = float(E[i] @ E[j])

    return {
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "q1": float(np.percentile(sims, 25)),
        "median": float(np.median(sims)),
        "q3": float(np.percentile(sims, 75)),
        "max": float(sims.max()),
    }


def measure_dimension_spread(E: np.ndarray) -> SimilarityStats:
    """Check if embedding dims have similar variance (isotropy)."""
    d = E.shape[1]
    target = 1.0 / np.sqrt(float(d))

    stds = E.std(axis=0).astype(np.float32)

    return {
        "dim": float(d),
        "target_isotropic": float(target),
        "mean_std": float(stds.mean()),
        "min_std": float(stds.min()),
        "max_std": float(stds.max()),
    }


def measure_neighbor_separation(
    E: np.ndarray,
    k: int = KNN_NEIGHBORS,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """kNN sim vs random-pair sim -> gap metric."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]

    if n < k + 1:
        return {"mean_nn_sim": 0.0, "mean_rand_sim": 0.0, "gap": 0.0}

    probes = rng.choice(n, size=min(n_probe, n), replace=False)

    nn_sims = []
    rand_sims = []

    for i in probes:
        sims = E @ E[int(i)]
        sims[int(i)] = -np.inf

        nn_idx = np.argpartition(-sims, kth=k - 1)[:k]
        nn_sims.append(float(np.mean(sims[nn_idx])))

        j = int(rng.choice(n))
        while j == int(i):
            j = int(rng.choice(n))
        rand_sims.append(float(sims[j]))

    mean_nn = float(np.mean(nn_sims))
    mean_rand = float(np.mean(rand_sims))

    return {
        "mean_nn_sim": mean_nn,
        "mean_rand_sim": mean_rand,
        "gap": mean_nn - mean_rand,
    }


def measure_graph_degrees(
    E: np.ndarray,
    k: int = KNN_NEIGHBORS,
) -> SimilarityStats:
    """kNN graph degree stats (O(n²) memory)."""
    n = E.shape[0]

    S = E @ E.T
    np.fill_diagonal(S, -np.inf)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argpartition(-S[i], kth=k - 1)[:k]
        A[i, idx] = 1.0

    A_sym = np.maximum(A, A.T)
    deg = (A_sym != 0).sum(axis=1).astype(np.float32)

    return {
        "mean": float(deg.mean()),
        "std": float(deg.std()),
        "min": float(deg.min()),
        "q1": float(np.percentile(deg, 25)),
        "median": float(np.median(deg)),
        "q3": float(np.percentile(deg, 75)),
        "max": float(deg.max()),
        "isolated_count": float(np.sum(deg == 0)),
    }


def compare_neighborhoods(
    E_before: np.ndarray,
    E_after: np.ndarray,
    k: int = KNN_NEIGHBORS,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """Jaccard overlap: kNN neighborhoods before vs after transformation."""
    rng = np.random.default_rng(seed)
    n = E_before.shape[0]

    if n < k + 1:
        return {"mean_overlap": 0.0, "median_overlap": 0.0}

    probes = rng.choice(n, size=min(n_probe, n), replace=False)

    overlaps = []

    for i in probes:
        sims_before = E_before @ E_before[int(i)]
        sims_before[int(i)] = -np.inf
        nn_before = set(np.argpartition(-sims_before, kth=k - 1)[:k])

        sims_after = E_after @ E_after[int(i)]
        sims_after[int(i)] = -np.inf
        nn_after = set(np.argpartition(-sims_after, kth=k - 1)[:k])

        jaccard = len(nn_before & nn_after) / float(len(nn_before | nn_after))
        overlaps.append(jaccard)

    overlaps_arr = np.array(overlaps, dtype=np.float32)

    return {
        "mean_overlap": float(overlaps_arr.mean()),
        "median_overlap": float(np.median(overlaps_arr)),
        "min_overlap": float(overlaps_arr.min()),
        "max_overlap": float(overlaps_arr.max()),
    }


def _convert_to_tensors(
    processor_output: dict[str, Any],
    device: str,
) -> ProcessorTensors:
    tensors: ProcessorTensors = {}

    for key, value in processor_output.items():
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            array = np.array(value)
            tensor = torch.from_numpy(array)

        tensors[key] = tensor.to(device)

    return tensors
