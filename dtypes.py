from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


# N-dimensional array of float values
FloatArray: TypeAlias = NDArray[np.floating]

# "N-dimensional array of int values
IntArray: TypeAlias = NDArray[np.integer]

# adj matrix as sparse CSR or dense ndarray
Adjacency: TypeAlias = csr_matrix | np.ndarray

IgraphGraph: TypeAlias = Any

# mapping from (source, target) node pairs to edge weights
EdgeWeights: TypeAlias = dict[tuple[str, str], float]

# for symmetrizing directed adjacency matrices
SymmetrizeMode: TypeAlias = Literal["max", "mean", "min"]

# sklearn NearestNeighbors
NNAlgorithm: TypeAlias = Literal["auto", "ball_tree", "kd_tree", "brute"]

# train/test split
SplitName: TypeAlias = Literal["train", "val", "test"]

# metadata tag group
TagGroup: TypeAlias = Literal["mood", "genre", "theme", "style"]

SamplingStrategy: TypeAlias = Literal["none", "uniform"]
