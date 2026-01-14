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

ALIGNMENT_ALPHA = 2.0
UNIFORMITY_TEMP = 2.0

EmbeddingList: TypeAlias = list["AudioEmbedding"]
ValidationMetrics: TypeAlias = dict[str, Any]
SimilarityStats: TypeAlias = dict[str, float]
SpectrumData: TypeAlias = dict[str, Any]
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
        audio_paths: single path or list of paths to audio files
        embedder: loaded CLAP model
        track_ids: optional track IDs, defaults to filenames
        batch_size: num files to process per batch

    Returns:
        list[AudioEmbedding]: one embedding per audio file
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
    """
    Load embeddings from .npz file.

    Args:
        input_path: path to .npz file w/ 'track_ids' + 'embeddings' arrays

    Returns:
        list[AudioEmbedding]: loaded embeddings
    """
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
    """
    Save embeddings to .npz file.

    Args:
        embeddings: list of embeddings to save
        output_path: path to .npz file (will create parent dirs)
    """
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

    Breaks cone geometry in CLAP embeddings -> improves isotropy.

    Args:
        embeddings: list of raw embeddings

    Returns:
        list[AudioEmbedding]: centered + normalized embeddings
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


def validate_embeddings(
    embeddings: Sequence[AudioEmbedding],
    *,
    k: int = KNN_NEIGHBORS,
) -> ValidationMetrics:
    """
    Validate embedding quality across multiple metrics.

    Checks: validity (NaN/inf/zeros), pairwise sim, dim spread,
    neighbor separation, graph connectivity, effective rank,
    information abundance (IA), singular spectrum.

    Args:
        embeddings: list of embeddings to validate
        k: num neighbors for kNN graph

    Returns:
        dict[str, dict]: nested dict w/ metric categories + values
    """
    E = np.vstack([e.embedding for e in embeddings]).astype(np.float32)

    return {
        "validity": check_validity(E),
        "pairwise_cosine": sample_pairwise_cosine(E),
        "dimension_spread": measure_dimension_spread(E),
        "neighbor_gap": measure_neighbor_separation(E, k=k),
        "degrees": measure_graph_degrees(E, k=k),
        "effective_rank": {
            "erank": compute_effective_rank(E),
            "max_rank": float(E.shape[1]),
        },
        "information_abundance": {
            "ia": compute_information_abundance(E),
            "max_ia": float(E.shape[1]),
        },
        "spectrum": compute_singular_spectrum(E),
    }


def check_validity(E: np.ndarray) -> SimilarityStats:
    """
    Check for NaN, inf, zero-norm, duplicate vectors.

    Args:
        E: embedding matrix (n_samples, n_dims)

    Returns:
        dict[str, float]: counts for each validity issue
    """
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
    """
    Sample random pairs -> cosine sim stats.

    Args:
        E: embedding matrix (n_samples, n_dims)
        n_samples: num pairs to sample
        seed: random seed

    Returns:
        dict[str, float]: mean, std, min, q1, median, q3, max
    """
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
    """
    Check if embedding dims have similar variance (isotropy).

    Target variance for isotropic d-dim space: 1/sqrt(d).

    Args:
        E: embedding matrix (n_samples, n_dims)

    Returns:
        dict[str, float]: dim, target_isotropic, mean/min/max std
    """
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
    """
    kNN sim vs random-pair sim -> gap metric.

    Larger gap = better neighbor structure.

    Args:
        E: embedding matrix (n_samples, n_dims)
        k: num neighbors for kNN
        n_probe: num probe points to sample
        seed: random seed

    Returns:
        dict[str, float]: mean_nn_sim, mean_rand_sim, gap
    """
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
    """
    kNN graph degree stats (O(n²) memory).

    Computes bidirectional kNN graph -> degree distribution.

    Args:
        E: embedding matrix (n_samples, n_dims)
        k: num neighbors for kNN

    Returns:
        dict[str, float]: mean, std, min, q1, median, q3, max, isolated_count
    """
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


def compute_effective_rank(E: np.ndarray) -> float:
    """
    Compute effective rank via Shannon entropy of normalized singular values.

    Measures how many dims are effectively used. Returns value between
    1 (complete collapse) and d (full dim usage).

    Args:
        E: embedding matrix (n_samples, n_dims)

    Returns:
        float: effective rank, should approach d for well-distributed embeddings
    """
    singular_values = np.linalg.svd(E, full_matrices=False, compute_uv=False)
    singular_values_sq = singular_values**2
    total = singular_values_sq.sum()

    if total < 1e-12:
        return 1.0

    p = singular_values_sq / total
    entropy = -(p * np.nan_to_num(np.log(p), neginf=0.0)).sum()
    erank = float(np.exp(entropy))

    return erank


def compute_information_abundance(E: np.ndarray) -> float:
    """
    Compute information abundance (IA): sum(σ) / max(σ).

    Measures dim utilization. 1 = complete collapse, d = uniform distribution.

    Args:
        E: embedding matrix (n_samples, n_dims)

    Returns:
        float: IA in range [1, d]
    """
    singular_values = np.linalg.svd(E, full_matrices=False, compute_uv=False)

    if singular_values.max() < 1e-12:
        return 1.0

    ia = float(singular_values.sum() / singular_values.max())
    return ia


def compute_singular_spectrum(E: np.ndarray) -> SpectrumData:
    """
    Compute singular value spectrum of normalized embedding covariance.

    Returns singular values in descending order with raw + log-scaled
    + normalized versions for analyzing dim collapse.

    Args:
        E: embedding matrix (n_samples, n_dims)

    Returns:
        dict with keys: singular_values, singular_values_log,
        singular_values_normalized, n_dims, n_nonzero
    """
    z = E / np.linalg.norm(E, axis=1, keepdims=True)
    C = np.cov(z.T)
    singular_values = np.linalg.svd(C, compute_uv=False)

    singular_values_log = np.log(singular_values + 1e-12)

    sv_max = singular_values.max()
    if sv_max > 1e-12:
        singular_values_normalized = singular_values / sv_max
    else:
        singular_values_normalized = np.zeros_like(singular_values)

    n_nonzero = int(np.sum(singular_values > 1e-9))

    return {
        "singular_values": singular_values,
        "singular_values_log": singular_values_log,
        "singular_values_normalized": singular_values_normalized,
        "n_dims": len(singular_values),
        "n_nonzero": n_nonzero,
    }


def compute_alignment_uniformity(
    E: np.ndarray,
    positive_pairs: list[tuple[int, int]],
    alpha: float = ALIGNMENT_ALPHA,
    t: float = UNIFORMITY_TEMP,
) -> SimilarityStats:
    """
    Compute alignment + uniformity for contrastive learning.

    Alignment: how close positive pairs are mapped.
    Uniformity: how uniformly embeddings are distributed on unit sphere.

    Args:
        E: embedding matrix (n_samples, n_dims)
        positive_pairs: list of (i, j) index pairs that should be similar
        alpha: exponent for alignment loss
        t: temperature param for uniformity

    Returns:
        dict[str, float]: alignment, uniformity scores
    """
    if len(positive_pairs) == 0:
        return {"alignment": 0.0, "uniformity": 0.0}

    alignment_dists = []
    for i, j in positive_pairs:
        dist = np.linalg.norm(E[i] - E[j])
        alignment_dists.append(float(dist**alpha))

    alignment = float(np.mean(alignment_dists))

    n = E.shape[0]
    if n < 2:
        return {"alignment": alignment, "uniformity": 0.0}

    pairwise_sq_dists = []
    for i in range(min(n, 1000)):
        for j in range(i + 1, min(n, 1000)):
            sq_dist = float(np.sum((E[i] - E[j]) ** 2))
            pairwise_sq_dists.append(sq_dist)

    if len(pairwise_sq_dists) == 0:
        uniformity = 0.0
    else:
        exp_terms = np.exp(-t * np.array(pairwise_sq_dists))
        uniformity = float(np.log(np.mean(exp_terms)))

    return {
        "alignment": alignment,
        "uniformity": uniformity,
    }


def compare_neighborhoods(
    E_before: np.ndarray,
    E_after: np.ndarray,
    k: int = KNN_NEIGHBORS,
    n_probe: int = PROBE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> SimilarityStats:
    """
    Jaccard overlap: kNN neighborhoods before vs after transformation.

    Measures how much neighborhood structure is preserved.

    Args:
        E_before: embedding matrix before transformation
        E_after: embedding matrix after transformation
        k: num neighbors for kNN
        n_probe: num probe points
        seed: random seed

    Returns:
        dict[str, float]: mean/median/min/max overlap
    """
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
    """Convert HuggingFace processor output -> torch tensors on device."""
    tensors: ProcessorTensors = {}

    for key, value in processor_output.items():
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            array = np.array(value)
            tensor = torch.from_numpy(array)
        tensors[key] = tensor.to(device)

    return tensors
