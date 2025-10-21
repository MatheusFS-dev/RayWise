"""Utilities for reconstructing Raymobtime baseline arrays from raw data files.

This module reads the raw Raymobtime s008 dataset (CSV metadata, LIDAR tensors,
and beam outputs) and recreates the baseline arrays used by the learning
pipelines. The resulting tensors replicate the shapes expected by legacy
experiments: coordinates ``(num_samples, 2)``, LIDAR data ``(num_samples, 20,
200, 10)``, and beam tensors ``(num_samples, 8, 32)``.

The implementation prioritises transparency and debuggability, so every
intermediate step is wrapped by single-purpose helper functions and prints
status information. This design makes it easier to audit the reconstruction
process and to troubleshoot missing or malformed files in the massive Raymobtime
archives.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------------------


def _find_episode_file(folder: str, episode_id: int, extension: str) -> str:
    """Locate a dataset file that corresponds to a specific episode.

    The helper searches for files following Raymobtime's ``*_e_<episode>`` naming
    pattern while guarding against partial matches. It prints every major step to
    make debugging missing-episode issues straightforward.

    Args:
        folder (str): Directory that stores the per-episode files.
        episode_id (int): Episode identifier provided by Raymobtime metadata.
        extension (str): File extension without the leading dot (for example,
            ``"npz"``).

    Returns:
        str: Absolute path to the episode file.

    Raises:
        FileNotFoundError: If the folder does not exist or no file matches the
            naming pattern ``*_e_<episode_id>.<extension>``.
    """

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Data folder not found: {folder}. Please double-check the dataset path."
        )

    escaped_extension = re.escape(extension)
    pattern = re.compile(rf".*_e_{episode_id}\.{escaped_extension}")
    print(
        f"Searching for episode {episode_id} inside '{folder}' using extension '.{extension}'..."
    )

    for candidate in os.listdir(folder):
        if pattern.fullmatch(candidate):
            episode_path = os.path.join(folder, candidate)
            print(f"  ✔ Found episode file: {episode_path}")
            return episode_path

    raise FileNotFoundError(
        "Could not locate an episode file for episode "
        f"{episode_id} inside {folder!r}."
    )


# --------------------------------------------------------------------------------------
# Coordinate metadata loading
# --------------------------------------------------------------------------------------


def load_coordinate_dataframe(csv_path: str) -> pd.DataFrame:
    """Load the Raymobtime coordinate CSV file and return it as a DataFrame.

    The CSV stores one row per channel and includes metadata such as the
    episode, scene, receiver identifier, and 3D coordinates. Only a subset of
    the columns is required for the baseline dataset, but the helper retains the
    full structure to simplify auditing and future feature engineering.

    Args:
        csv_path (str): Absolute path to ``CoordVehiclesRxPerScene_sXXX.csv``.

    Returns:
        pd.DataFrame: Coordinate metadata with original dtypes preserved. The
        caller is responsible for selecting valid channels.

    Raises:
        FileNotFoundError: If the CSV file is missing.
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Coordinate CSV not found: {csv_path}. Ensure the raw dataset is available."
        )

    print(f"Loading coordinate metadata from: {csv_path}")
    df_coord = pd.read_csv(csv_path)
    print(
        "  ✔ Coordinate DataFrame loaded with shape "
        f"{df_coord.shape} and columns: {list(df_coord.columns)}"
    )
    return df_coord


def filter_valid_channels(df_coord: pd.DataFrame) -> pd.DataFrame:
    """Filter the coordinate metadata to retain only valid communication channels.

    Notes:
        Raymobtime encodes valid channels with the letter ``"V"`` in the ``Val``
        column. The filter is case-insensitive to gracefully handle potential
        formatting differences.

    Args:
        df_coord (pd.DataFrame): Complete coordinate DataFrame returned by
            :func:`load_coordinate_dataframe`.

    Returns:
        pd.DataFrame: Filtered DataFrame that contains only the valid rows with
        a reset positional index.

    Raises:
        ValueError: If any required columns for downstream processing are
        missing from ``df_coord``.
    """

    print("Filtering DataFrame to keep only valid channels (Val == 'V')...")
    required_columns = {"EpisodeID", "SceneID", "VehicleArrayID", "x", "y", "Val"}
    if not required_columns.issubset(df_coord.columns):
        missing = sorted(required_columns.difference(df_coord.columns))
        raise ValueError(
            "Coordinate DataFrame is missing required columns: "
            f"{missing}. Please verify the CSV integrity."
        )

    valid_mask = df_coord["Val"].astype(str).str.upper() == "V"
    df_valid = df_coord.loc[valid_mask].reset_index(drop=True)
    print(
        f"  ✔ Retained {len(df_valid)} valid rows out of {len(df_coord)} total entries."
    )
    return df_valid


# --------------------------------------------------------------------------------------
# LIDAR tensor processing
# --------------------------------------------------------------------------------------


def load_lidar_episode(
    lidar_folder: str,
    episode_id: int,
    array_key: str = "obstacles_matrix_array",
) -> np.ndarray:
    """Load the per-episode LIDAR tensor used to recreate the baseline input.

    Args:
        lidar_folder (str): Directory that stores ``obstacles_e_<episode>.npz``
            files.
        episode_id (int): Episode identifier used by Raymobtime.
        array_key (str): Key inside the ``.npz`` archive that stores the obstacle
            tensor. The default matches the official dataset.

    Returns:
        np.ndarray: Four-dimensional NumPy array with shape ``(num_pairs, 20,
        200, 10)`` where the first dimension enumerates the TX/RX pairs observed
        during the episode.

    Raises:
        FileNotFoundError: If the expected ``.npz`` file is missing.
        KeyError: If the provided key is not part of the archive.
    """

    episode_path = _find_episode_file(lidar_folder, episode_id, "npz")
    print(f"Loading LIDAR data from: {episode_path}")
    with np.load(episode_path, allow_pickle=True) as data:
        print(f"  • Available keys inside LIDAR archive: {list(data.files)}")
        if array_key not in data:
            raise KeyError(
                f"LIDAR key '{array_key}' missing in {episode_path}. Available keys:"
                f" {list(data.files)}"
            )
        lidar_episode = data[array_key]

    print(
        f"  ✔ Loaded LIDAR tensor with shape {lidar_episode.shape} for episode {episode_id}"
    )
    return lidar_episode


def extract_lidar_tensor(lidar_episode: np.ndarray, pair_index: int) -> np.ndarray:
    """Select the obstacle tensor for a specific transmitter/receiver pair.

    Args:
        lidar_episode (np.ndarray): Output of :func:`load_lidar_episode`.
        pair_index (int): Index of the TX/RX pair. The helper automatically
            clamps values that fall outside the valid range to accommodate
            off-by-one metadata issues.

    Returns:
        np.ndarray: Three-dimensional NumPy array with shape ``(20, 200, 10)``
        ready to be stacked into the global baseline tensor.

    Raises:
        ValueError: If the extracted slice does not follow the expected
        three-dimensional structure.
    """

    num_pairs = lidar_episode.shape[0]
    print(
        f"Selecting LIDAR slice for pair index {pair_index} (available pairs: {num_pairs})"
    )

    # Guard against 1-based indexing in the CSV by clamping into the valid range.
    if pair_index < 0:
        print(
            "  ! Negative pair index detected. Clamping to 0 to avoid indexing errors."
        )
        pair_index = 0
    if pair_index >= num_pairs:
        print(
            "  ! Pair index outside available range. Assuming 1-based indexing and"
            " decrementing by one."
        )
        pair_index = min(num_pairs - 1, pair_index - 1)

    lidar_slice = lidar_episode[pair_index]
    if lidar_slice.ndim != 3:
        raise ValueError(
            "Unexpected LIDAR slice dimensionality. Expected a 3D array but received "
            f"{lidar_slice.shape}."
        )
    print(f"  ✔ LIDAR slice shape: {lidar_slice.shape}")
    return lidar_slice


# --------------------------------------------------------------------------------------
# Beam output handling
# --------------------------------------------------------------------------------------


def load_beam_output_array(
    beam_output_path: str,
    array_key: Optional[str] = None,
) -> np.ndarray:
    """Load the beam output tensor that stores optimal beam selections.

    Args:
        beam_output_path (str): Path to the ``beams_output_sXXX.npz`` archive.
        array_key (Optional[str]): Optional override for the array name stored
            inside the archive. When ``None`` the function automatically selects
            the first available array.

    Returns:
        np.ndarray: Tensor with shape ``(num_samples, num_rx_beams, num_tx_beams)``
        containing the pre-computed beam data aligned with valid channels.

    Raises:
        FileNotFoundError: If the ``beam_output`` archive is missing.
        KeyError: If ``array_key`` does not exist in the archive.
    """

    if not os.path.isfile(beam_output_path):
        raise FileNotFoundError(
            f"Beam output archive not found: {beam_output_path}."
        )

    print(f"Loading beam output data from: {beam_output_path}")
    with np.load(beam_output_path, allow_pickle=True) as data:
        print(f"  • Available keys inside beam output archive: {list(data.files)}")
        resolved_key = array_key or data.files[0]
        if resolved_key not in data:
            raise KeyError(
                f"Beam output key '{resolved_key}' missing in {beam_output_path}."
                f" Available keys: {list(data.files)}"
            )
        beam_output = data[resolved_key]

    print(f"  ✔ Beam output tensor loaded with shape {beam_output.shape}")
    return beam_output
# --------------------------------------------------------------------------------------
# Dataset assembly pipeline
# --------------------------------------------------------------------------------------


def load_raymobtime_dataset(
    coord_csv_path: str,
    lidar_folder: str,
    beam_output_path: str,
    lidar_key: str = "obstacles_matrix_array",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recreate the baseline Raymobtime arrays from raw dataset folders.

    The loader iterates over every valid channel described in the coordinate CSV
    and collects three aligned arrays:

    * ``coord_input`` containing the (x, y) coordinates of the receiver.
    * ``lidar_input`` containing the voxelised point cloud for the corresponding
      TX/RX pair.
    * ``beam_output`` storing the original ``(num_samples, 8, 32)`` beam tensor
      from the baseline dataset without reducing it to sparse labels.

    Args:
        coord_csv_path (str): Absolute path to ``CoordVehiclesRxPerScene_sXXX.csv``.
        lidar_folder (str): Folder that stores ``obstacles_e_<episode>.npz`` files.
        beam_output_path (str): Path to the ``beam_output`` archive containing
            the pre-computed beam selections.
        lidar_key (str): Array key inside the LIDAR archive. Defaults to
            ``"obstacles_matrix_array"``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple ``(coord_input,
        lidar_input, beam_output)`` with NumPy arrays ready to be fed into the
        training pipelines.

    Raises:
        FileNotFoundError: If any of the required dataset files is missing.
        KeyError: When specified keys are absent in the underlying ``.npz`` files.
        ValueError: For inconsistent shapes or missing metadata columns.
    """

    print("================ Raymobtime raw dataset loader ================")

    df_coord = load_coordinate_dataframe(coord_csv_path)
    df_valid = filter_valid_channels(df_coord)
    beam_output_tensor = load_beam_output_array(beam_output_path)

    if len(beam_output_tensor) != len(df_valid):
        raise ValueError(
            "The number of beam samples in the beam output tensor does not match"
            " the number of valid channels in the coordinate CSV."
            f" Beam samples: {len(beam_output_tensor)}, valid rows: {len(df_valid)}."
        )
    print(
        "  ✔ Beam tensor sample count matches the number of valid coordinate rows."
    )

    coord_list: List[np.ndarray] = []
    lidar_list: List[np.ndarray] = []
    beam_tensor_list: List[np.ndarray] = []

    lidar_cache: Dict[int, np.ndarray] = {}

    for row_id, row in df_valid.iterrows():
        episode_id = int(row["EpisodeID"])
        scene_id = int(row.get("SceneID", 0))
        pair_index = int(row.get("VehicleArrayID", row.get("PairID", 0)))
        print(
            "--------------------------------------------------------------"
        )
        print(
            f"Processing row {row_id}: episode={episode_id}, scene={scene_id}, pair={pair_index}"
        )

        # Coordinates -----------------------------------------------------------------
        coord_xy = np.array([float(row["x"]), float(row["y"])], dtype=np.float32)
        coord_list.append(coord_xy)
        print(f"  ✔ Appended coordinates: {coord_xy} (shape: {coord_xy.shape})")

        # LIDAR -----------------------------------------------------------------------
        if episode_id not in lidar_cache:
            lidar_cache[episode_id] = load_lidar_episode(
                lidar_folder, episode_id, array_key=lidar_key
            )
        lidar_episode = lidar_cache[episode_id]

        lidar_pair_tensor = extract_lidar_tensor(lidar_episode, pair_index)
        lidar_list.append(lidar_pair_tensor.astype(np.float32))
        print(
            f"  ✔ Appended LIDAR tensor with shape {lidar_pair_tensor.shape} to dataset."
        )

        # Beam tensors -------------------------------------------------------------
        beam_slice = np.asarray(beam_output_tensor[row_id])
        beam_tensor_list.append(beam_slice)
        print(
            f"  ✔ Appended beam tensor with shape {beam_slice.shape} for sample index {row_id}"
        )

    coord_input = np.stack(coord_list, axis=0).astype(np.float32)
    lidar_input = np.stack(lidar_list, axis=0).astype(np.float32)
    beam_output_array = np.stack(beam_tensor_list, axis=0)

    print("================ Reconstruction summary =======================")
    print(f"coord_input shape: {coord_input.shape}")
    print(f"lidar_input shape: {lidar_input.shape}")
    print(f"beam_output shape: {beam_output_array.shape}")
    print("==============================================================")

    return coord_input, lidar_input, beam_output_array


# --------------------------------------------------------------------------------------
# Script entry point for manual testing
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
    DEFAULT_COORD_CSV = (
        "/home/matheus/src/datasets/RayWise/Raymobtime_s008/raw_data/"
        "CoordVehiclesRxPerScene_s008.csv"
    )
    DEFAULT_LIDAR_FOLDER = (
        "/home/matheus/src/datasets/RayWise/Raymobtime_s008/processed_raw_data/"
        "lidar_data_s008"
    )
    DEFAULT_BEAM_OUTPUT = (
        "/home/matheus/src/datasets/RayWise/Raymobtime_s008/baseline_data/beam_output/"
        "beams_output_s008.npz"
    )

    print("Running standalone reconstruction using default dataset paths...")

    try:
        coord_input, lidar_input, beam_output = load_raymobtime_dataset(
            coord_csv_path=DEFAULT_COORD_CSV,
            lidar_folder=DEFAULT_LIDAR_FOLDER,
            beam_output_path=DEFAULT_BEAM_OUTPUT,
        )
    except Exception as exc:  # pragma: no cover - manual debugging aid
        print("An error occurred during reconstruction:")
        print(exc)
    else:
        print("Baseline arrays successfully reconstructed!")
        print(f"coord_input dtype: {coord_input.dtype}")
        print(f"lidar_input dtype: {lidar_input.dtype}")
        print(f"beam_output dtype: {beam_output.dtype}")
