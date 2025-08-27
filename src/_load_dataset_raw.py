"""
raymobtime_loader.py
====================

This module provides utilities to load the raw HDF5 channel data from the
Raymobtime datasets and organise it into separate Line‑of‑Sight (LOS) and
Non‑Line‑of‑Sight (NLOS) subsets.  Each Raymobtime dataset is composed of
multiple HDF5 files (one per episode).  The file format is described in the
Raymobtime wiki: the `allEpisodeData` array stored in each file is a 4‑D
tensor with dimensions corresponding to the number of scenes, number of
transmitter/receiver pairs, maximum number of rays (paths) and the number of
path parameters【664931960144665†L276-L286】.  The path parameters include
received power, time of arrival, elevation and azimuth angles of departure
and arrival, a flag indicating whether a ray is line of sight, the ray
phase (available only when the dataset was generated with Wireless Insite
version ≥3.3) and, for some mobile datasets like ``s008`` and ``s009``, the
orientation of the vehicle【664931960144665†L294-L299】.

The loading functions defined here iterate over all episodes in a dataset,
extract every channel (scene × receiver) and classify it as LOS or NLOS
depending on whether at least one valid path has the LOS flag equal to 1.  All
channels containing only invalid rays (i.e. rows filled with NaNs) are
discarded.  Channels are flattened into a one‑dimensional feature vector to
facilitate subsequent use in machine learning pipelines.  NaN entries are
replaced with zeros.

Example
-------

```python
from raymobtime_loader import load_raymobtime_dataset

# Replace these with the local paths where you have extracted the HDF5 files
path_s008 = "/data/raymobtime/ray_tracing_data_s008_carrier60GHz"
path_s009 = "/data/raymobtime/ray_tracing_data_s009_carrier60GHz"

# Process the s008 dataset
(X_s008_los, y_s008_los), (X_s008_nlos, y_s008_nlos) = load_raymobtime_dataset(path_s008)
print(f"s008: {len(X_s008_los)} LOS channels, {len(X_s008_nlos)} NLOS channels")

# Process the s009 dataset
(X_s009_los, y_s009_los), (X_s009_nlos, y_s009_nlos) = load_raymobtime_dataset(path_s009)
print(f"s009: {len(X_s009_los)} LOS channels, {len(X_s009_nlos)} NLOS channels")
```

The resulting feature arrays ``X_s008_los``, ``X_s008_nlos``, etc., contain
flattened channel descriptions where the first axis enumerates the channels
and the second axis enumerates all rays (including invalid ones) and
parameters.  The corresponding label arrays ``y_*`` are filled with ones for
LOS samples and zeros for NLOS samples.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import h5py
import numpy as np

def _load_episode(file_path: str) -> Tuple[np.ndarray, int, int, int, int]:
    """Load a single HDF5 episode file and return its channel tensor.

    The HDF5 file is expected to contain an array named ``allEpisodeData``
    following the Raymobtime format.  According to the dataset documentation,
    this array has the shape ``(num_scenes, num_pairs, max_num_rays, num_params)``
    where the last dimension enumerates the path parameters in the order given
    in the wiki【664931960144665†L276-L286】.  NaN values are used to mark
    invalid rays.

    Parameters
    ----------
    file_path:
        Path to an HDF5 file.

    Returns
    -------
    tuple
        A tuple containing the array of channel data and the numbers of
        scenes, pairs, rays and parameters.  The array is loaded into
        memory as a NumPy array of type ``float64``.
    """
    with h5py.File(file_path, "r") as h5_file:
        # Load the entire `allEpisodeData` dataset.  The indexing order is
        # (scenes, pairs, rays, parameters) as documented.
        data = h5_file["allEpisodeData"][()]

    # Extract dimensions for convenience
    num_scenes, num_pairs, max_num_rays, num_params = data.shape
    return data, num_scenes, num_pairs, max_num_rays, num_params


def _flatten_channel(channel: np.ndarray) -> np.ndarray:
    """Replace NaN entries with zeros and flatten a channel into a 1‑D vector.

    Each channel is a 2‑D matrix of shape ``(max_num_rays, num_params)``.  NaNs
    denote invalid rays or missing parameters.  They are replaced by zeros
    before flattening.

    Parameters
    ----------
    channel:
        Array of shape ``(max_num_rays, num_params)`` containing the raw
        path parameters for a given scene/receiver pair.

    Returns
    -------
    numpy.ndarray
        A one‑dimensional vector of length ``max_num_rays * num_params``.
    """
    # Replace NaNs with zeros
    channel_clean = np.nan_to_num(channel, nan=0.0)
    return channel_clean.flatten()


def _is_los(channel: np.ndarray, los_index: int = 6) -> bool:
    """Determine if a channel contains at least one line‑of‑sight path.

    Parameters
    ----------
    channel:
        Array of shape ``(max_num_rays, num_params)`` containing the path
        parameters for a channel.  It may include NaN rows for invalid rays.

    los_index:
        Column index of the LOS flag in the parameter dimension.  By
        convention this is the 7th parameter (index 6)【664931960144665†L276-L286】.

    Returns
    -------
    bool
        ``True`` if there is at least one valid ray with ``isLOS == 1`` and
        ``False`` otherwise.  Rays with NaN entries in the LOS column are
        ignored.
    """
    # Extract the LOS flag column
    los_column = channel[:, los_index]
    # Identify valid flags (exclude NaNs)
    valid_mask = ~np.isnan(los_column)
    if not np.any(valid_mask):
        # No valid rays in this channel
        return False
    # Determine if any ray is LOS
    return np.any(los_column[valid_mask] == 1)


def load_raymobtime_dataset(dataset_dir: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load all HDF5 episodes from a Raymobtime dataset and separate channels by LOS/NLOS.

    This function iterates over all ``.hdf5`` files in ``dataset_dir`` (the
    directory containing the per‑episode HDF5 files such as
    ``ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e0.hdf5``) and
    collects every channel (scene × receiver) into one of two lists depending
    on the presence of a line‑of‑sight path.  Invalid channels (where all
    rays are NaN) are skipped.

    Parameters
    ----------
    dataset_dir:
        Path to the directory containing the HDF5 episode files for a given
        Raymobtime simulation (e.g. the extracted ``ray_tracing_data_s008_carrier60GHz`` directory).

    Returns
    -------
    tuple
        A pair ``((X_los, y_los), (X_nlos, y_nlos))``.  ``X_los`` and
        ``X_nlos`` are two‑dimensional NumPy arrays where each row is a
        flattened channel.  ``y_los`` and ``y_nlos`` are one‑dimensional
        arrays filled with ones and zeros respectively.
    """
    # Prepare containers for features and labels
    los_features: List[np.ndarray] = []
    nlos_features: List[np.ndarray] = []

    # Gather all HDF5 files in the directory.  Different datasets may use
    # different extensions (e.g. `.hdf5` or `.h5`), so we accept any of
    # these.
    hdf5_files = [fname for fname in os.listdir(dataset_dir) if fname.lower().endswith((".hdf5", ".h5"))]
    hdf5_files.sort()

    for fname in hdf5_files:
        file_path = os.path.join(dataset_dir, fname)
        try:
            episode_data, num_scenes, num_pairs, max_num_rays, num_params = _load_episode(file_path)
        except Exception as exc:
            # Skip files that cannot be read
            print(f"Warning: could not process {file_path}: {exc}")
            continue
        # Iterate through scenes and receivers
        for scene_idx in range(num_scenes):
            for pair_idx in range(num_pairs):
                channel = episode_data[scene_idx, pair_idx]  # shape (max_num_rays, num_params)
                # Skip channels where all entries are NaN
                if np.isnan(channel).all():
                    continue
                # Determine LOS/NLOS status
                if _is_los(channel, los_index=6):
                    los_features.append(_flatten_channel(channel))
                else:
                    nlos_features.append(_flatten_channel(channel))

    # Convert lists to arrays.  We use object dtype because the flattened
    # channels can have different lengths if max_num_rays varies across
    # episodes.  If all episodes share the same max number of rays (as is
    # typical for a single simulation), the resulting array will have
    # homogeneous shape and NumPy will upcast automatically.
    X_los = np.array(los_features)
    y_los = np.ones(len(los_features), dtype=int)
    X_nlos = np.array(nlos_features)
    y_nlos = np.zeros(len(nlos_features), dtype=int)

    return (X_los, y_los), (X_nlos, y_nlos)