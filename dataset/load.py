from dataclasses import dataclass
from pathlib import Path

from configs.dataset import AppConfig


@dataclass(frozen=True, slots=True)
class Dataset:
    """Store a dataset name and its root folder."""
    name: str
    root: Path


def load_dataset(config: AppConfig, project_root: Path, name: str) -> Dataset:
    """Load a dataset by name."""
    try:
        loader = _LOADERS[name]
    except KeyError as e:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_LOADERS)}") from e
    return loader(config, project_root)


def _load_mtg_jamendo(config: AppConfig, project_root: Path) -> Dataset:
    """Build paths for the MTG-Jamendo dataset."""
    root = (project_root / config.datasets.mtg_jamendo.root).resolve()
    return Dataset(name="mtg_jamendo", root=root)


def _load_deam(config: AppConfig, project_root: Path) -> Dataset:
    """Build paths for the DEAM dataset."""
    root = (project_root / config.datasets.deam.root).resolve()
    return Dataset(name="deam", root=root)


_LOADERS = {
    "mtg_jamendo": _load_mtg_jamendo,
    "deam": _load_deam,
}
