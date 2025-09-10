"""Utility functions for loading saved datasets from nas_cnn1d_v3.4.

This module provides helpers to reconstruct the training and validation
splits that were exported as individual NumPy ``.npy`` files. The expected
folder structure mirrors the one produced by the notebook after selecting
the optimal ``train_frac_s009``.

Notes:
    The directory should contain ``train`` and ``valid`` subdirectories with
    ``labels.npy``, ``coords.npy`` and ``lidar.npy`` files inside each.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def load_best_datasets(base_dir: str | Path = "best_datasets") -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Load the exported train and validation datasets.

    Detailed description:
        Loads label, coordinate and LiDAR arrays that were previously saved as
        individual ``.npy`` files for both the training and validation splits.

    Args:
        base_dir: Root directory containing the ``train`` and ``valid``
            subdirectories. Defaults to ``"best_datasets"``.

    Returns:
        A tuple of two tuples:

        - ``(train_labels, train_coords, train_lidar)``
        - ``(valid_labels, valid_coords, valid_lidar)``

    Raises:
        FileNotFoundError: If any of the expected ``.npy`` files is missing.

    Warnings:
        Loading very large datasets may exhaust available memory.

    Notes:
        The returned arrays follow the order ``labels``, ``coords``, ``lidar``.
    """

    base_path = Path(base_dir)
    train_dir = base_path / "train"
    valid_dir = base_path / "valid"

    files = {"labels": "labels.npy", "coords": "coords.npy", "lidar": "lidar.npy"}

    def _load_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = {}
        for key, name in files.items():
            file_path = split_dir / name
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file: {file_path}")
            arrays[key] = np.load(file_path)
        return arrays["labels"], arrays["coords"], arrays["lidar"]

    train_data = _load_split(train_dir)
    valid_data = _load_split(valid_dir)
    return train_data, valid_data
