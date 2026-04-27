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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from _load_dataset import convert_to_sparse_labels


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
        raise FileNotFoundError(f"Data folder not found: {folder}. Please double-check the dataset path.")

    escaped_extension = re.escape(extension)
    pattern = re.compile(rf".*_e_{episode_id}\.{escaped_extension}")
    print(f"Searching for episode {episode_id} inside '{folder}' using extension '.{extension}'...")

    for candidate in os.listdir(folder):
        if pattern.fullmatch(candidate):
            episode_path = os.path.join(folder, candidate)
            print(f"  ✔ Found episode file: {episode_path}")
            return episode_path

    raise FileNotFoundError(
        "Could not locate an episode file for episode " f"{episode_id} inside {folder!r}."
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
        raise FileNotFoundError(f"Coordinate CSV not found: {csv_path}. Ensure the raw dataset is available.")

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
    print(f"  ✔ Retained {len(df_valid)} valid rows out of {len(df_coord)} total entries.")
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
                f"LIDAR key '{array_key}' missing in {episode_path}. Available keys:" f" {list(data.files)}"
            )
        lidar_episode = data[array_key]

    print(f"  ✔ Loaded LIDAR tensor with shape {lidar_episode.shape} for episode {episode_id}")
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
    print(f"Selecting LIDAR slice for pair index {pair_index} (available pairs: {num_pairs})")

    # Guard against 1-based indexing in the CSV by clamping into the valid range.
    if pair_index < 0:
        print("  ! Negative pair index detected. Clamping to 0 to avoid indexing errors.")
        pair_index = 0
    if pair_index >= num_pairs:
        print("  ! Pair index outside available range. Assuming 1-based indexing and" " decrementing by one.")
        pair_index = min(num_pairs - 1, pair_index - 1)

    lidar_slice = lidar_episode[pair_index]
    if lidar_slice.ndim != 3:
        raise ValueError(
            "Unexpected LIDAR slice dimensionality. Expected a 3D array but received " f"{lidar_slice.shape}."
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
        raise FileNotFoundError(f"Beam output archive not found: {beam_output_path}.")

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


def _normalise_channel_condition(raw_label: Union[str, int, float]) -> str:
    """Translate Raymobtime channel condition labels into canonical forms.

    The Raymobtime coordinate CSV stores line-of-sight annotations with
    different encodings across releases. Besides the straightforward ``"LOS"``
    and ``"NLOS"`` strings, some datasets expose the flag as ``"LOS=1"`` / ``"LOS=0"``
    or as bare integers. This helper harmonises those variants so downstream
    code can operate exclusively on the canonical labels.

    Args:
        raw_label (Union[str, int, float]): Value read from the coordinate CSV.

    Returns:
        str: ``"LOS"`` when the channel is line-of-sight and ``"NLOS"`` otherwise.

    Raises:
        ValueError: If the label cannot be interpreted as LOS or NLOS.
    """

    text = str(raw_label).strip()
    if not text:
        raise ValueError("Empty channel condition label encountered.")

    upper_text = text.upper()
    if upper_text in {"LOS", "NLOS"}:
        return upper_text

    match = re.fullmatch(r"(LOS|NLOS)\s*=\s*([01])", upper_text)
    if match:
        prefix, numeric = match.groups()
        if prefix == "LOS":
            return "LOS" if numeric == "1" else "NLOS"
        return "NLOS" if numeric == "1" else "LOS"

    if upper_text in {"0", "1"}:
        return "LOS" if upper_text == "1" else "NLOS"

    raise ValueError(f"Unsupported channel condition label: {raw_label!r}")


def _identify_channel_condition_column(df_coord: pd.DataFrame) -> str:
    """Identify the column in the coordinate DataFrame with LOS/NLOS labels.

    The Raymobtime metadata may expose the channel condition through different
    column names depending on the dataset release. This helper scans the
    DataFrame and selects the first column whose values can be normalised to
    ``"LOS"`` and ``"NLOS"`` using :func:`_normalise_channel_condition`, allowing
    the main loader to provide deterministic splits when requested.

    Args:
        df_coord (pd.DataFrame): Coordinate metadata returned by
            :func:`load_coordinate_dataframe` or a filtered subset of it.

    Returns:
        str: Name of the column that stores LOS/NLOS annotations.

    Raises:
        ValueError: If the DataFrame does not expose LOS/NLOS information in an
            identifiable column.
    """

    candidate_columns = [
        column
        for column in df_coord.columns
        if any(keyword in column.lower() for keyword in ("los", "condition"))
    ]

    for column in candidate_columns:
        values = df_coord[column].dropna()
        normalised_values = set()
        try:
            for value in values:
                normalised_values.add(_normalise_channel_condition(value))
        except ValueError:
            continue

        if normalised_values and normalised_values.issubset({"LOS", "NLOS"}):
            print(
                "  ✔ Identified LOS/NLOS column: "
                f"'{column}' with values {sorted(normalised_values)}"
            )
            return column

    raise ValueError(
        "Could not detect a column with LOS/NLOS labels. Ensure the coordinate CSV "
        "contains channel condition information."
    )


def load_raymobtime_dataset(
    coord_csv_path: str,
    lidar_folder: str,
    beam_output_path: str,
    lidar_key: str = "obstacles_matrix_array",
    split_los_nlos: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
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
        split_los_nlos (bool): If ``True``, the loader additionally separates the
            reconstructed arrays into line-of-sight (LOS) and non-line-of-sight
            (NLOS) subsets using the channel condition specified in the
            coordinate CSV.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple ``(coord_input,
        lidar_input, beam_output)`` with NumPy arrays ready to be fed into the
        training pipelines when ``split_los_nlos`` is ``False``.
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            Nested tuple ``((coord_los, lidar_los, beam_los), (coord_nlos,
            lidar_nlos, beam_nlos))`` containing the LOS and NLOS partitions when
            ``split_los_nlos`` is ``True``. Each inner tuple preserves the sample
            order from the coordinate CSV.

    Raises:
        FileNotFoundError: If any of the required dataset files is missing.
        KeyError: When specified keys are absent in the underlying ``.npz`` files.
        ValueError: For inconsistent shapes, missing metadata columns, or
        absent LOS/NLOS annotations when ``split_los_nlos`` is ``True``.
    """

    print("================ Raymobtime raw dataset loader ================")

    df_coord = load_coordinate_dataframe(coord_csv_path)
    df_valid = filter_valid_channels(df_coord)
    beam_output_tensor = load_beam_output_array(beam_output_path)

    los_column: Optional[str] = None
    condition_labels: List[str] = []
    if split_los_nlos:
        los_column = _identify_channel_condition_column(df_valid)

    if len(beam_output_tensor) != len(df_valid):
        raise ValueError(
            "The number of beam samples in the beam output tensor does not match"
            " the number of valid channels in the coordinate CSV."
            f" Beam samples: {len(beam_output_tensor)}, valid rows: {len(df_valid)}."
        )
    print("  ✔ Beam tensor sample count matches the number of valid coordinate rows.")

    coord_list: List[np.ndarray] = []
    lidar_list: List[np.ndarray] = []
    beam_tensor_list: List[np.ndarray] = []

    lidar_cache: Dict[int, np.ndarray] = {}

    for row_id, row in df_valid.iterrows():
        episode_id = int(row["EpisodeID"])
        scene_id = int(row.get("SceneID", 0))
        pair_index = int(row.get("VehicleArrayID", row.get("PairID", 0)))
        print("--------------------------------------------------------------")
        print(f"Processing row {row_id}: episode={episode_id}, scene={scene_id}, pair={pair_index}")

        # Coordinates -----------------------------------------------------------------
        coord_xy = np.array([float(row["x"]), float(row["y"])], dtype=np.float32)
        coord_list.append(coord_xy)
        print(f"  ✔ Appended coordinates: {coord_xy} (shape: {coord_xy.shape})")

        # LIDAR -----------------------------------------------------------------------
        if episode_id not in lidar_cache:
            lidar_cache[episode_id] = load_lidar_episode(lidar_folder, episode_id, array_key=lidar_key)
        lidar_episode = lidar_cache[episode_id]

        lidar_pair_tensor = extract_lidar_tensor(lidar_episode, pair_index)
        lidar_list.append(lidar_pair_tensor.astype(np.float32))
        print(f"  ✔ Appended LIDAR tensor with shape {lidar_pair_tensor.shape} to dataset.")

        # Beam tensors -------------------------------------------------------------
        beam_slice = np.asarray(beam_output_tensor[row_id])
        beam_tensor_list.append(beam_slice)
        print(f"  ✔ Appended beam tensor with shape {beam_slice.shape} for sample index {row_id}")

        if split_los_nlos:
            assert los_column is not None  # For type checkers only
            try:
                condition_normalised = _normalise_channel_condition(row[los_column])
            except ValueError as exc:
                raise ValueError(
                    "Encountered an unknown channel condition label while splitting the dataset: "
                    f"{row[los_column]!r}."
                ) from exc
            condition_labels.append(condition_normalised)
            print(f"  ✔ Recorded channel condition: {condition_normalised}")

    coord_input = np.stack(coord_list, axis=0).astype(np.float32)
    lidar_input = np.stack(lidar_list, axis=0).astype(np.float32)
    beam_output_array = np.stack(beam_tensor_list, axis=0)

    print("================ Reconstruction summary =======================")
    print(f"coord_input shape: {coord_input.shape}")
    print(f"lidar_input shape: {lidar_input.shape}")
    print(f"beam_output shape: {beam_output_array.shape}")
    print("==============================================================")

    if not split_los_nlos:
        return coord_input, lidar_input, beam_output_array

    condition_array = np.asarray(condition_labels, dtype=object)
    los_mask = condition_array == "LOS"
    nlos_mask = condition_array == "NLOS"

    coord_los = coord_input[los_mask]
    lidar_los = lidar_input[los_mask]
    beam_los = beam_output_array[los_mask]

    coord_nlos = coord_input[nlos_mask]
    lidar_nlos = lidar_input[nlos_mask]
    beam_nlos = beam_output_array[nlos_mask]

    print("================ LOS/NLOS split summary ======================")
    print(f"LOS samples: {coord_los.shape[0]}, NLOS samples: {coord_nlos.shape[0]}")
    print("==============================================================")

    return (coord_los, lidar_los, beam_los), (coord_nlos, lidar_nlos, beam_nlos)


def load_dataset_raw_sparse_labels(
    *,
    s008_coord_csv: str,
    s008_lidar_folder: str,
    s008_beam_output_path: str,
    s009_coord_csv: str,
    s009_lidar_folder: str,
    s009_beam_output_path: str,
    data_seed: int,
    stratify: bool = False,
    stratify_by_class: bool = False,
    report_label_coverage: bool = True,
    remove_null_labels: bool = False,
    return_conditions: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and balance Raymobtime raw datasets with sparse beam labels.

    The function reconstructs the raw Raymobtime S008 and S009 datasets using
    :func:`load_raymobtime_dataset`, concatenates their LOS and NLOS subsets, and
    converts the dense beam tensors into sparse labels. The combined dataset is
    shuffled with the provided ``data_seed`` and split into training,
    validation, and testing partitions using a 70/20/10 ratio. Training and
    validation splits are explicitly balanced to contain the same number of LOS
    and NLOS samples unless ``stratify`` is enabled. When ``stratify=True``,
    the split preserves the LOS/NLOS proportions from the combined S008 and
    S009 datasets instead of enforcing a 50/50 balance. Extensive print
    statements document every major step as well as the resulting array shapes
    to simplify debugging.

    Notes:
        The balancing constraint may reduce the usable sample count if the LOS
        and NLOS subsets differ significantly in size (when ``stratify=False``).
        Any remaining samples are included in the test partition, which can
        therefore be imbalanced. With ``stratify=True``, all samples are used
        and class proportions are preserved across splits.

    Args:
        s008_coord_csv (str): Path to the S008 coordinate CSV file.
        s008_lidar_folder (str): Directory containing the S008 LIDAR ``.npz`` files.
        s008_beam_output_path (str): Path to the S008 beam output archive.
        s009_coord_csv (str): Path to the S009 coordinate CSV file.
        s009_lidar_folder (str): Directory containing the S009 LIDAR ``.npz`` files.
        s009_beam_output_path (str): Path to the S009 beam output archive.
        data_seed (int): Deterministic seed applied to all shuffling operations.
        stratify (bool): If ``True``, apply a stratified 70/20/10 split that
            preserves the LOS/NLOS proportion of the combined S008 and S009
            datasets. If ``False`` (default), enforce a 50/50 LOS/NLOS balance
            for training and validation splits.
        stratify_by_class (bool): If ``True``, ignore LOS/NLOS balancing and
            perform a per-label 70/20/10 split across all samples.
        report_label_coverage (bool): If ``True``, print per-split label counts
            and percentages.
        remove_null_labels (bool): If ``True``, remap labels to a compact
            ``0..N-1`` range based only on the labels present in the data,
            effectively removing gaps for absent labels (e.g., unused IDs in
            the 0–255 range).
        return_conditions (bool): If ``True``, also return arrays indicating the
            LOS/NLOS condition for each sample in the train/val/test splits. The
            condition arrays follow the same ordering as the returned feature
            and label arrays.

    Returns:
        Tuple[np.ndarray, ...]: Tuple containing nine NumPy arrays in the
        following order: ``x_lidar_train``, ``x_lidar_val``, ``x_lidar_test``,
        ``x_coord_train``, ``x_coord_val``, ``x_coord_test``, ``y_train``,
        ``y_val``, ``y_test``.

        When ``return_conditions`` is ``True``, three additional arrays are
        appended: ``condition_train``, ``condition_val``, and
        ``condition_test``. Each is a one-dimensional object array of strings
        with values ``"LOS"`` or ``"NLOS"`` aligned with the corresponding
        split.

    Raises:
        ValueError: If the datasets do not provide both LOS and NLOS samples or
        if the balancing and split configuration would result in empty
        partitions.
    """

    """
    Numeric example of LOS/NLOS balancing (stratify=False):

    Source counts:
        S008 -> LOS: 6,482, NLOS: 4,712
        S009 -> LOS: 1,473, NLOS: 8,165
        Combined -> LOS: 7,955, NLOS: 12,877

    1) Compute min_samples:
        min_samples = min(7,955, 12,877) = 7,955

    2) Per-class split from min_samples:
        train_count = floor(0.7 * 7,955) = 5,568
        val_count   = floor(0.2 * 7,955) = 1,591
        test_count  = 7,955 - 5,568 - 1,591 = 796

    3) LOS class:
        - Use first 7,955 shuffled LOS samples:
            - 5,568 -> train
            - 1,591 -> val
            -   796 -> balanced test
        - Leftover LOS:
            7,955 total - 7,955 used = 0 leftover

    4) NLOS class:
        - Use first 7,955 shuffled NLOS samples:
            - 5,568 -> train
            - 1,591 -> val
            -   796 -> balanced test
        - Leftover NLOS:
            12,877 total - 7,955 used = 4,922 extra NLOS samples
            These 4,922 go only to test.

    5) Final splits:
        - Train: 5,568 LOS + 5,568 NLOS = 11,136 (balanced 50/50)
        - Val  : 1,591 LOS + 1,591 NLOS = 3,182 (balanced 50/50)
        - Test : 796 LOS + (796 balanced + 4,922 leftover) NLOS
                 = 796 LOS + 5,718 NLOS = 6,514 (imbalanced)

    Stratified example (stratify=True):

    Source counts:
        S008 -> LOS: 6,482, NLOS: 4,712
        S009 -> LOS: 1,473, NLOS: 8,165
        Combined -> LOS: 7,955, NLOS: 12,877 (Total: 20,832)

    1) 70/20/10 global split (no forced balancing):
        total_train = floor(0.7 * 20,832) = 14,582
        total_val   = floor(0.2 * 20,832) = 4,166
        total_test  = 20,832 - 14,582 - 4,166 = 2,084

    2) Class counts follow combined proportions (about 38.2% LOS / 61.8% NLOS):
        Train:
            los_train  = floor(14,582 * 7,955 / 20,832) = 5,568
            nlos_train = 14,582 - 5,568 = 9,014

        Validation:
            los_val  = floor(4,166 * 7,955 / 20,832) = 1,590
            nlos_val = 4,166 - 1,590 = 2,576

        Test (remainders):
            los_test  = 7,955  - 5,568 - 1,590 = 797
            nlos_test = 12,877 - 9,014 - 2,576 = 1,287

    3) Final splits (proportional):
        - Train: 5,568 LOS + 9,014 NLOS = 14,582
        - Val  : 1,590 LOS + 2,576 NLOS = 4,166
        - Test : 797 LOS + 1,287 NLOS  = 2,084
    """
    
    """
    Class level stratification example (``stratify_by_class=True``)

    When ``stratify_by_class`` is enabled, LOS and NLOS subsets are first
    concatenated into a single dataset::

        lidar_all = concat(lidar_los, lidar_nlos)
        coord_all = concat(coord_los, coord_nlos)
        y_all    = concat(y_los, y_nlos)

    The split then ignores the LOS/NLOS distinction and operates only on the
    beam labels in ``y_all``. For each distinct label ``c`` in ``y_all`` the
    procedure is:

    1. Collect indices of all samples with that label::

           class_idx = np.where(y_all == c)[0]

    2. Shuffle these indices with the global RNG so that each class is
       internally randomised.

    3. If the class has size ``total = class_idx.size``:

       * If ``total < 2``, all samples of this label go to the training set.
         This guarantees that every label present in the dataset has at least
         one training example, even for extremely rare classes.

       * Otherwise, compute a per class 70/20/10 split::

             train_n = floor(0.7 * total)
             val_n   = floor(0.2 * total)
             test_n  = total - train_n - val_n

         Apply safeguards so that counts are valid:

         * Ensure there is at least one training sample::

               if train_n < 1:
                   train_n = 1

         * If validation was floored to zero but there is remaining budget,
           give validation one sample before assigning to test::

               remaining = total - train_n
               if val_n < 1 and remaining > 0:
                   val_n = 1

         * Clamp the validation count to the remaining budget and recompute the
           test count so that all three are non negative and their sum does not
           exceed ``total``::

               remaining = total - train_n
               if val_n > remaining:
                   val_n = remaining
               test_n = total - train_n - val_n
               if test_n < 0:
                   test_n = 0

       The final per class indices are then taken as contiguous slices of the
       shuffled ``class_idx``::

           train_indices.extend(class_idx[:train_n])
           val_indices.extend(class_idx[train_n : train_n + val_n])
           test_indices.extend(class_idx[train_n + val_n : train_n + val_n + test_n])

    After looping over all labels, the three global index lists are used to
    index ``lidar_all``, ``coord_all`` and ``y_all`` to obtain
    ``(x_lidar_train, x_coord_train, y_train)``,
    ``(x_lidar_val, x_coord_val, y_val)`` and
    ``(x_lidar_test, x_coord_test, y_test)``. Each split is then shuffled
    independently to remove any residual ordering.

    Numerical example
    -----------------

    Consider a toy dataset after concatenating LOS and NLOS:

    * Label 0: 10 samples
    * Label 1: 3 samples
    * Label 2: 1 sample

    Total samples: 14.

    For label 0 (``total = 10``):

    * Initial counts:
      * ``train_n = floor(0.7 * 10) = 7``
      * ``val_n   = floor(0.2 * 10) = 2``
      * ``test_n  = 10 - 7 - 2 = 1``
    * Safeguards do not change these values.
    * Result for label 0:
      * 7 samples in train
      * 2 samples in val
      * 1 sample in test

    For label 1 (``total = 3``):

    * Initial counts:
      * ``train_n = floor(0.7 * 3) = 2``
      * ``val_n   = floor(0.2 * 3) = 0``
      * ``test_n  = 3 - 2 - 0 = 1``
    * Safeguards still keep ``train_n = 2``, ``val_n = 0``, ``test_n = 1``.
    * Result for label 1:
      * 2 samples in train
      * 0 samples in val
      * 1 sample in test

    For label 2 (``total = 1``):

    * Since ``total < 2``, the single sample is assigned entirely to train.
    * Result for label 2:
      * 1 sample in train
      * 0 samples in val
      * 0 samples in test

    The final per split label counts are:

    * Train:
      * 7 samples of label 0
      * 2 samples of label 1
      * 1 sample of label 2
      * 10 samples in total
    * Validation:
      * 2 samples of label 0
      * 0 samples of labels 1 and 2
      * 2 samples in total
    * Test:
      * 1 sample of label 0
      * 1 sample of label 1
      * 0 samples of label 2
      * 2 samples in total

    Every sample appears exactly once in one of the splits. Each label has at
    least one training example. Larger classes follow the 70/20/10 proportion
    closely, while rare classes are kept primarily or exclusively in the
    training set.
    """

    def _shuffle_together(rng: np.random.Generator, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Shuffle multiple arrays in unison using the provided random generator.
        Consider that all arrays must have the same first dimension size (rows).
        It will shuffle all arrays in the same way so that corresponding rows stay aligned.
        """
        indices = np.arange(arrays[0].shape[0])
        rng.shuffle(indices)
        return tuple(array[indices] for array in arrays)

    print("============= load_dataset_raw_sparse_labels =============")
    print("Loading S008 dataset with LOS/NLOS split...")
    s008_result = load_raymobtime_dataset(
        coord_csv_path=s008_coord_csv,
        lidar_folder=s008_lidar_folder,
        beam_output_path=s008_beam_output_path,
        split_los_nlos=True,
    )
    print("Loading S009 dataset with LOS/NLOS split...")
    s009_result = load_raymobtime_dataset(
        coord_csv_path=s009_coord_csv,
        lidar_folder=s009_lidar_folder,
        beam_output_path=s009_beam_output_path,
        split_los_nlos=True,
    )

    (s008_coord_los, s008_lidar_los, s008_beam_los), (s008_coord_nlos, s008_lidar_nlos, s008_beam_nlos) = s008_result
    (s009_coord_los, s009_lidar_los, s009_beam_los), (s009_coord_nlos, s009_lidar_nlos, s009_beam_nlos) = s009_result

    print("---------------- Per-dataset LOS/NLOS summary ----------------")
    print(
        "S008 -> LOS: {los}, NLOS: {nlos}".format(
            los=s008_coord_los.shape[0], nlos=s008_coord_nlos.shape[0]
        )
    )
    print(
        "       lidar(LOS): {l_los}, lidar(NLOS): {l_nlos}".format(
            l_los=s008_lidar_los.shape, l_nlos=s008_lidar_nlos.shape
        )
    )
    print(
        "       coord(LOS): {c_los}, coord(NLOS): {c_nlos}".format(
            c_los=s008_coord_los.shape, c_nlos=s008_coord_nlos.shape
        )
    )
    print(
        "       beam(LOS): {b_los}, beam(NLOS): {b_nlos}".format(
            b_los=s008_beam_los.shape, b_nlos=s008_beam_nlos.shape
        )
    )
    print(
        "S009 -> LOS: {los}, NLOS: {nlos}".format(
            los=s009_coord_los.shape[0], nlos=s009_coord_nlos.shape[0]
        )
    )
    print(
        "       lidar(LOS): {l_los}, lidar(NLOS): {l_nlos}".format(
            l_los=s009_lidar_los.shape, l_nlos=s009_lidar_nlos.shape
        )
    )
    print(
        "       coord(LOS): {c_los}, coord(NLOS): {c_nlos}".format(
            c_los=s009_coord_los.shape, c_nlos=s009_coord_nlos.shape
        )
    )
    print(
        "       beam(LOS): {b_los}, beam(NLOS): {b_nlos}".format(
            b_los=s009_beam_los.shape, b_nlos=s009_beam_nlos.shape
        )
    )
    print("--------------------------------------------------------------")

    print("Concatenating LOS subsets from S008 and S009...")
    # Joins LOS coordinates from S008 and S009 into one single array along the row dimension,
    # then converts values to 32-bit floating-point numbers.
    coord_los = np.concatenate((s008_coord_los, s009_coord_los), axis=0).astype(np.float32)
    lidar_los = np.concatenate((s008_lidar_los, s009_lidar_los), axis=0).astype(np.float32)
    y_los_dense = np.concatenate((s008_beam_los, s009_beam_los), axis=0).astype(np.float32)

    # Idem, but for NLOS data.
    print("Concatenating NLOS subsets from S008 and S009...")
    coord_nlos = np.concatenate((s008_coord_nlos, s009_coord_nlos), axis=0).astype(np.float32)
    lidar_nlos = np.concatenate((s008_lidar_nlos, s009_lidar_nlos), axis=0).astype(np.float32)
    y_nlos_dense = np.concatenate((s008_beam_nlos, s009_beam_nlos), axis=0).astype(np.float32)

    if coord_los.size == 0 or coord_nlos.size == 0:
        raise ValueError("Both LOS and NLOS samples are required to create balanced splits.")

    # Process dense beam tensors into sparse labels for LOS and NLOS datasets.
    print("Converting beam tensors to sparse labels...")
    y_los = convert_to_sparse_labels(y_los_dense)
    y_nlos = convert_to_sparse_labels(y_nlos_dense)

    print(f"LOS lidar shape: {lidar_los.shape}, coords shape: {coord_los.shape}, labels shape: {y_los.shape}")
    print(f"NLOS lidar shape: {lidar_nlos.shape}, coords shape: {coord_nlos.shape}, labels shape: {y_nlos.shape}")

    if remove_null_labels:
        print("Compacting labels to remove gaps/unseen IDs...")
        all_labels = np.concatenate((y_los, y_nlos))
        present_labels = np.unique(all_labels)
        label_map = np.full(int(present_labels.max()) + 1, -1, dtype=int)
        label_map[present_labels] = np.arange(present_labels.size, dtype=int)
        y_los = label_map[y_los]
        y_nlos = label_map[y_nlos]
        print(
            "Label remap -> present: {present}, new range: [0, {max_new}], removed gaps: {removed}".format(
                present=present_labels.size,
                max_new=present_labels.size - 1,
                removed=(int(present_labels.max()) + 1) - present_labels.size,
            )
        )

    condition_los = np.full(y_los.shape, "LOS", dtype=object)
    condition_nlos = np.full(y_nlos.shape, "NLOS", dtype=object)

    rng = np.random.default_rng(data_seed)
    print(f"Shuffling datasets with seed {data_seed}...")
    if return_conditions:
        lidar_los, coord_los, y_los, condition_los = _shuffle_together(
            rng, lidar_los, coord_los, y_los, condition_los
        )
        lidar_nlos, coord_nlos, y_nlos, condition_nlos = _shuffle_together(
            rng, lidar_nlos, coord_nlos, y_nlos, condition_nlos
        )
    else:
        lidar_los, coord_los, y_los = _shuffle_together(rng, lidar_los, coord_los, y_los)
        lidar_nlos, coord_nlos, y_nlos = _shuffle_together(rng, lidar_nlos, coord_nlos, y_nlos)

    # Count how many LOS and NLOS samples exist to define the splits.
    los_samples = lidar_los.shape[0]
    nlos_samples = lidar_nlos.shape[0]
    total_samples = los_samples + nlos_samples

    condition_train = condition_val = condition_test = None

    if stratify_by_class:
        print("================= Stratified split by class =================")
        lidar_all = np.concatenate((lidar_los, lidar_nlos), axis=0).astype(np.float32)
        coord_all = np.concatenate((coord_los, coord_nlos), axis=0).astype(np.float32)
        y_all = np.concatenate((y_los, y_nlos), axis=0)
        if return_conditions:
            condition_all = np.concatenate((condition_los, condition_nlos), axis=0)

        print(
            f"Total samples: {y_all.shape[0]} | "
            f"Label range: [0, {int(y_all.max())}] | Unique labels: {np.unique(y_all).size}"
        )

        # Manual per-class split to avoid scikit-learn failures on tiny classes
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        labels = np.unique(y_all)
        for label in labels:
            class_idx = np.where(y_all == label)[0]
            rng.shuffle(class_idx)
            total = class_idx.size

            if total < 2:
                # Keep singletons entirely in the train split
                train_indices.extend(class_idx.tolist())
                continue

            train_n = int(np.floor(0.7 * total))
            val_n = int(np.floor(0.2 * total))
            test_n = total - train_n - val_n

            # Ensure at least one train sample and prioritize validation over test for tiny classes
            if train_n < 1:
                train_n = 1
            remaining = total - train_n

            # If validation got rounded to zero but there is room, give it one before test
            if val_n < 1 and remaining > 0:
                val_n = 1
            if val_n > remaining:
                val_n = remaining

            test_n = total - train_n - val_n
            if test_n < 0:
                test_n = 0

            train_indices.extend(class_idx[:train_n].tolist())
            val_indices.extend(class_idx[train_n : train_n + val_n].tolist())
            test_indices.extend(class_idx[train_n + val_n : train_n + val_n + test_n].tolist())

        x_lidar_train = lidar_all[train_indices]
        x_coord_train = coord_all[train_indices]
        y_train = y_all[train_indices]
        if return_conditions:
            condition_train = condition_all[train_indices]

        x_lidar_val = lidar_all[val_indices]
        x_coord_val = coord_all[val_indices]
        y_val = y_all[val_indices]
        if return_conditions:
            condition_val = condition_all[val_indices]

        x_lidar_test = lidar_all[test_indices]
        x_coord_test = coord_all[test_indices]
        y_test = y_all[test_indices]
        if return_conditions:
            condition_test = condition_all[test_indices]

        print(
            "Split sizes (class-stratified) -> train: {tr}, val: {va}, test: {te}".format(
                tr=x_lidar_train.shape[0], va=x_lidar_val.shape[0], te=x_lidar_test.shape[0]
            )
        )

        split_counts = None  # handled separately in the summary below
    elif stratify:
        print("================= Stratified split overview =================")
        print(
            f"Available samples -> total: {total_samples}, LOS: {los_samples}, NLOS: {nlos_samples}"
        )

        total_train = int(np.floor(0.7 * total_samples))
        total_val = int(np.floor(0.2 * total_samples))
        total_test = total_samples - total_train - total_val

        if total_train == 0 or total_val == 0 or total_test == 0:
            raise ValueError(
                "The requested 70/20/10 split results in an empty partition when stratifying. "
                "Check that the combined datasets contain sufficient samples."
            )

        print(
            "Total split sizes (stratified) -> train: {tr}, val: {va}, test: {te}".format(
                tr=total_train, va=total_val, te=total_test
            )
        )

        los_train_count = int(np.floor(total_train * los_samples / total_samples))
        nlos_train_count = total_train - los_train_count

        los_val_count = int(np.floor(total_val * los_samples / total_samples))
        nlos_val_count = total_val - los_val_count

        los_test_count = los_samples - los_train_count - los_val_count
        nlos_test_count = nlos_samples - nlos_train_count - nlos_val_count

        if min(
            los_train_count,
            nlos_train_count,
            los_val_count,
            nlos_val_count,
            los_test_count,
            nlos_test_count,
        ) < 0:
            raise ValueError("Computed negative split sizes during stratification; check input sample counts.")

        print(
            "Per-class splits (stratified) -> train: LOS {tr_los}, NLOS {tr_nlos}; "
            "val: LOS {va_los}, NLOS {va_nlos}; test: LOS {te_los}, NLOS {te_nlos}".format(
                tr_los=los_train_count,
                tr_nlos=nlos_train_count,
                va_los=los_val_count,
                va_nlos=nlos_val_count,
                te_los=los_test_count,
                te_nlos=nlos_test_count,
            )
        )

        los_train_slice = slice(0, los_train_count)
        los_val_slice = slice(los_train_count, los_train_count + los_val_count)
        los_test_slice = slice(los_train_count + los_val_count, los_train_count + los_val_count + los_test_count)

        nlos_train_slice = slice(0, nlos_train_count)
        nlos_val_slice = slice(nlos_train_count, nlos_train_count + nlos_val_count)
        nlos_test_slice = slice(
            nlos_train_count + nlos_val_count, nlos_train_count + nlos_val_count + nlos_test_count
        )

        def _split_class_data_counts(
            lidar_array: np.ndarray,
            coord_array: np.ndarray,
            label_array: np.ndarray,
            train_n: int,
            val_n: int,
            test_n: int,
        ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            """
            Split one class (LOS or NLOS) into train/val/test using explicit counts.
            """
            train_slice = slice(0, train_n)
            val_slice = slice(train_n, train_n + val_n)
            test_slice = slice(train_n + val_n, train_n + val_n + test_n)

            return (
                (lidar_array[train_slice], coord_array[train_slice], label_array[train_slice]),
                (lidar_array[val_slice], coord_array[val_slice], label_array[val_slice]),
                (lidar_array[test_slice], coord_array[test_slice], label_array[test_slice]),
            )

        print("Splitting LOS class by slices for train/val/test (stratified)...")
        (lidar_los_train, coord_los_train, y_los_train), (
            lidar_los_val,
            coord_los_val,
            y_los_val,
        ), (
            lidar_los_test,
            coord_los_test,
            y_los_test,
        ) = _split_class_data_counts(
            lidar_los, coord_los, y_los, los_train_count, los_val_count, los_test_count
        )
        print(
            "LOS split sizes -> train: {tr}, val: {va}, test: {te}".format(
                tr=lidar_los_train.shape[0], va=lidar_los_val.shape[0], te=lidar_los_test.shape[0]
            )
        )
        if return_conditions:
            condition_los_train = condition_los[los_train_slice]
            condition_los_val = condition_los[los_val_slice]
            condition_los_test = condition_los[los_test_slice]

        print("Splitting NLOS class by slices for train/val/test (stratified)...")
        (
            lidar_nlos_train,
            coord_nlos_train,
            y_nlos_train,
        ), (
            lidar_nlos_val,
            coord_nlos_val,
            y_nlos_val,
        ), (
            lidar_nlos_test,
            coord_nlos_test,
            y_nlos_test,
        ) = _split_class_data_counts(
            lidar_nlos, coord_nlos, y_nlos, nlos_train_count, nlos_val_count, nlos_test_count
        )
        print(
            "NLOS split sizes -> train: {tr}, val: {va}, test: {te}".format(
                tr=lidar_nlos_train.shape[0], va=lidar_nlos_val.shape[0], te=lidar_nlos_test.shape[0]
            )
        )
        if return_conditions:
            condition_nlos_train = condition_nlos[nlos_train_slice]
            condition_nlos_val = condition_nlos[nlos_val_slice]
            condition_nlos_test = condition_nlos[nlos_test_slice]

        leftover_los = 0
        leftover_nlos = 0
        split_counts = {
            "train": {"los": los_train_count, "nlos": nlos_train_count},
            "val": {"los": los_val_count, "nlos": nlos_val_count},
            "test": {"los": los_test_count, "nlos": nlos_test_count},
        }
    else:
        min_samples = min(
            los_samples, nlos_samples
        )  # Takes the smaller of the two counts. This is the maximum balanced number of samples to use for each class.
        print("==================== Balancing overview ======================")
        print(
            f"Available samples -> LOS: {los_samples}, NLOS: {nlos_samples}. "
            f"Using min count for balance: {min_samples}"
        )

        train_count = int(np.floor(0.7 * min_samples))
        val_count = int(np.floor(0.2 * min_samples))
        test_count = min_samples - train_count - val_count

        if train_count == 0 or val_count == 0 or test_count == 0:
            raise ValueError(
                "The requested 70/20/10 split results in an empty partition. Check that the datasets contain"
                " sufficient LOS and NLOS samples."
            )

        print(
            "Per-class splits (balanced) -> train: {tr}, val: {va}, test: {te}".format(
                tr=train_count, va=val_count, te=test_count
            )
        )
        print(
            "Slices -> train: [0:{tr}), val: [{tr}:{tr_va}), test: [{tr_va}:{tr_va_te})".format(
                tr=train_count, tr_va=train_count + val_count, tr_va_te=train_count + val_count + test_count
            )
        )

        balanced_train_slice = slice(0, train_count)
        balanced_val_slice = slice(train_count, train_count + val_count)
        balanced_test_slice = slice(train_count + val_count, train_count + val_count + test_count)

        def _split_class_data(
            lidar_array: np.ndarray,
            coord_array: np.ndarray,
            label_array: np.ndarray,
        ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            """
            Defines another helper function that takes LIDAR, coordinates, and labels 
            for one class (LOS or NLOS) and splits them into train, validation, and test pieces.
            """
            train_slice = balanced_train_slice
            val_slice = balanced_val_slice
            test_slice = balanced_test_slice

            return (
                (lidar_array[train_slice], coord_array[train_slice], label_array[train_slice]),
                (lidar_array[val_slice], coord_array[val_slice], label_array[val_slice]),
                (lidar_array[test_slice], coord_array[test_slice], label_array[test_slice]),
            )

        print("Splitting LOS class by slices for train/val/test...")
        (lidar_los_train, coord_los_train, y_los_train), (
            lidar_los_val,
            coord_los_val,
            y_los_val,
        ), (
            lidar_los_test,
            coord_los_test,
            y_los_test,
        ) = _split_class_data(lidar_los, coord_los, y_los)
        print(
            "LOS split sizes -> train: {tr}, val: {va}, test: {te}".format(
                tr=lidar_los_train.shape[0], va=lidar_los_val.shape[0], te=lidar_los_test.shape[0]
            )
        )
        if return_conditions:
            condition_los_train = condition_los[balanced_train_slice]
            condition_los_val = condition_los[balanced_val_slice]
            condition_los_test = condition_los[balanced_test_slice]

        print("Splitting NLOS class by slices for train/val/test...")
        (
            lidar_nlos_train,
            coord_nlos_train,
            y_nlos_train,
        ), (
            lidar_nlos_val,
            coord_nlos_val,
            y_nlos_val,
        ), (
            lidar_nlos_test,
            coord_nlos_test,
            y_nlos_test,
        ) = _split_class_data(lidar_nlos, coord_nlos, y_nlos)
        print(
            "NLOS split sizes -> train: {tr}, val: {va}, test: {te}".format(
                tr=lidar_nlos_train.shape[0], va=lidar_nlos_val.shape[0], te=lidar_nlos_test.shape[0]
            )
        )
        if return_conditions:
            condition_nlos_train = condition_nlos[balanced_train_slice]
            condition_nlos_val = condition_nlos[balanced_val_slice]
            condition_nlos_test = condition_nlos[balanced_test_slice]

        leftover_los = max(0, los_samples - min_samples)
        leftover_nlos = max(0, nlos_samples - min_samples)
        split_counts = {
            "train": {"los": train_count, "nlos": train_count},
            "val": {"los": val_count, "nlos": val_count},
            "test": {"los": test_count + leftover_los, "nlos": test_count + leftover_nlos},
        }

    if not stratify_by_class:
        print("Combining LOS/NLOS slices for train and validation sets...")
        x_lidar_train = np.concatenate((lidar_los_train, lidar_nlos_train), axis=0)
        x_coord_train = np.concatenate((coord_los_train, coord_nlos_train), axis=0)
        y_train = np.concatenate((y_los_train, y_nlos_train), axis=0)
        if return_conditions:
            condition_train = np.concatenate((condition_los_train, condition_nlos_train), axis=0)

        x_lidar_val = np.concatenate((lidar_los_val, lidar_nlos_val), axis=0)
        x_coord_val = np.concatenate((coord_los_val, coord_nlos_val), axis=0)
        y_val = np.concatenate((y_los_val, y_nlos_val), axis=0)
        if return_conditions:
            condition_val = np.concatenate((condition_los_val, condition_nlos_val), axis=0)

        print("Combining remaining samples for test set (may be imbalanced)...")
        lidar_test_parts = [lidar_los_test, lidar_nlos_test]
        coord_test_parts = [coord_los_test, coord_nlos_test]
        y_test_parts = [y_los_test, y_nlos_test]
        if return_conditions:
            condition_test_parts = [condition_los_test, condition_nlos_test]

        if not stratify:
            if los_samples > min_samples:
                print(
                    f"  • Adding leftover LOS samples to test: index range [{min_samples}:{los_samples}) -> count {leftover_los}"
                )
                lidar_test_parts.append(lidar_los[min_samples:])
                coord_test_parts.append(coord_los[min_samples:])
                y_test_parts.append(y_los[min_samples:])
                if return_conditions:
                    condition_test_parts.append(condition_los[min_samples:])
            if nlos_samples > min_samples:
                print(
                    f"  • Adding leftover NLOS samples to test: index range [{min_samples}:{nlos_samples}) -> count {leftover_nlos}"
                )
                lidar_test_parts.append(lidar_nlos[min_samples:])
                coord_test_parts.append(coord_nlos[min_samples:])
                y_test_parts.append(y_nlos[min_samples:])
                if return_conditions:
                    condition_test_parts.append(condition_nlos[min_samples:])

        x_lidar_test = np.concatenate(lidar_test_parts, axis=0)
        x_coord_test = np.concatenate(coord_test_parts, axis=0)
        y_test = np.concatenate(y_test_parts, axis=0)
        if return_conditions:
            condition_test = np.concatenate(condition_test_parts, axis=0)

    if stratify:
        print(
            "Pre-shuffle test composition (stratified) -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["test"]["los"], nlos=split_counts["test"]["nlos"], tot=x_lidar_test.shape[0]
            )
        )
    elif not stratify_by_class:
        print(
            "Pre-shuffle test composition -> balanced: {bal}, leftover LOS: {l_los}, leftover NLOS: {l_nlos}, total: {tot}".format(
                bal=test_count * 2,
                l_los=leftover_los,
                l_nlos=leftover_nlos,
                tot=x_lidar_test.shape[0],
            )
        )

    print("Shuffling combined splits individually to avoid class ordering...")
    if return_conditions:
        x_lidar_train, x_coord_train, y_train, condition_train = _shuffle_together(
            rng, x_lidar_train, x_coord_train, y_train, condition_train
        )
        x_lidar_val, x_coord_val, y_val, condition_val = _shuffle_together(
            rng, x_lidar_val, x_coord_val, y_val, condition_val
        )
        x_lidar_test, x_coord_test, y_test, condition_test = _shuffle_together(
            rng, x_lidar_test, x_coord_test, y_test, condition_test
        )
    else:
        x_lidar_train, x_coord_train, y_train = _shuffle_together(rng, x_lidar_train, x_coord_train, y_train)
        x_lidar_val, x_coord_val, y_val = _shuffle_together(rng, x_lidar_val, x_coord_val, y_val)
        x_lidar_test, x_coord_test, y_test = _shuffle_together(rng, x_lidar_test, x_coord_test, y_test)

    print(f"Train shapes -> lidar: {x_lidar_train.shape}, coord: {x_coord_train.shape}, labels: {y_train.shape}")
    print(f"Validation shapes -> lidar: {x_lidar_val.shape}, coord: {x_coord_val.shape}, labels: {y_val.shape}")
    print(f"Test shapes -> lidar: {x_lidar_test.shape}, coord: {x_coord_test.shape}, labels: {y_test.shape}")

    if report_label_coverage and y_train.size and y_val.size and y_test.size:
        def _coverage(counts: np.ndarray, split_name: str) -> None:
            present = counts[counts > 0]
            if present.size == 0:
                print(f"{split_name}: no labels present.")
                return
            print(
                f"{split_name}: {present.size}/{counts.size} classes present"
            )
            print(f"{split_name} per-class counts: {counts.tolist()}")

        num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
        counts_train = np.bincount(y_train.astype(int), minlength=num_classes)
        counts_val = np.bincount(y_val.astype(int), minlength=num_classes)
        counts_test = np.bincount(y_test.astype(int), minlength=num_classes)
        total_train = counts_train.sum()
        total_val = counts_val.sum()
        total_test = counts_test.sum()

        print("================ Label coverage per split ================")
        _coverage(counts_train, "Train")
        _coverage(counts_val, "Val")
        _coverage(counts_test, "Test")

        missing_in_train = np.where((counts_train == 0) & ((counts_val > 0) | (counts_test > 0)))[0]
        if missing_in_train.size:
            print(
                "Labels missing in train but present in val/test (consider stratifying by label): "
                f"{missing_in_train.tolist()}"
            )
        for cls_idx in range(num_classes):
            class_total = counts_train[cls_idx] + counts_val[cls_idx] + counts_test[cls_idx]
            if class_total == 0:
                train_pct = val_pct = test_pct = 0.0
            else:
                train_pct = counts_train[cls_idx] / class_total * 100
                val_pct = counts_val[cls_idx] / class_total * 100
                test_pct = counts_test[cls_idx] / class_total * 100
            print(
                "Class {cls} train/val/test -> {tr:.2f}%/{va:.2f}%/{te:.2f}% (total {trc}/{vac}/{tec})".format(
                    cls=cls_idx,
                    tr=train_pct,
                    va=val_pct,
                    te=test_pct,
                    trc=int(counts_train[cls_idx]),
                    vac=int(counts_val[cls_idx]),
                    tec=int(counts_test[cls_idx]),
                )
            )
        print("===========================================================")

    # Final detailed summary (datasets, balancing decisions, and arrays)
    print("======================== Final Summary =======================")
    print("Source dataset composition:")
    print(
        "  S008 -> LOS: {los}, NLOS: {nlos}".format(
            los=s008_coord_los.shape[0], nlos=s008_coord_nlos.shape[0]
        )
    )
    print(
        "  S009 -> LOS: {los}, NLOS: {nlos}".format(
            los=s009_coord_los.shape[0], nlos=s009_coord_nlos.shape[0]
        )
    )
    print(
        "Combined (pre-split) -> LOS: {los}, NLOS: {nlos}".format(
            los=los_samples, nlos=nlos_samples
        )
    )
    if stratify_by_class:
        print("Split details (class-stratified):")
        print(
            "  Train: {tr}, Val: {va}, Test: {te}".format(
                tr=x_lidar_train.shape[0], va=x_lidar_val.shape[0], te=x_lidar_test.shape[0]
            )
        )
    elif stratify:
        print("Stratified split details:")
        print(
            "  Train class counts -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["train"]["los"], nlos=split_counts["train"]["nlos"], tot=x_lidar_train.shape[0]
            )
        )
        print(
            "  Val class counts   -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["val"]["los"], nlos=split_counts["val"]["nlos"], tot=x_lidar_val.shape[0]
            )
        )
        print(
            "  Test class counts  -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["test"]["los"], nlos=split_counts["test"]["nlos"], tot=x_lidar_test.shape[0]
            )
        )
    else:
        print("Balancing and split details:")
        print(
            "  Per-class used -> train: {tr}, val: {va}, test: {te}, remainders -> LOS: {l_los}, NLOS: {l_nlos}".format(
                tr=train_count, va=val_count, te=test_count, l_los=leftover_los, l_nlos=leftover_nlos
            )
        )
        print(
            "  Train class counts -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["train"]["los"], nlos=split_counts["train"]["nlos"], tot=x_lidar_train.shape[0]
            )
        )
        print(
            "  Val class counts   -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["val"]["los"], nlos=split_counts["val"]["nlos"], tot=x_lidar_val.shape[0]
            )
        )
        print(
            "  Test class counts  -> LOS: {los}, NLOS: {nlos}, total: {tot}".format(
                los=split_counts["test"]["los"], nlos=split_counts["test"]["nlos"], tot=x_lidar_test.shape[0]
            )
        )
    print("Final NumPy arrays:")
    print(
        "  x_lidar_train: {s} dtype={dt}".format(s=x_lidar_train.shape, dt=x_lidar_train.dtype)
    )
    print(
        "  x_lidar_val  : {s} dtype={dt}".format(s=x_lidar_val.shape, dt=x_lidar_val.dtype)
    )
    print(
        "  x_lidar_test : {s} dtype={dt}".format(s=x_lidar_test.shape, dt=x_lidar_test.dtype)
    )
    print(
        "  x_coord_train: {s} dtype={dt}".format(s=x_coord_train.shape, dt=x_coord_train.dtype)
    )
    print(
        "  x_coord_val  : {s} dtype={dt}".format(s=x_coord_val.shape, dt=x_coord_val.dtype)
    )
    print(
        "  x_coord_test : {s} dtype={dt}".format(s=x_coord_test.shape, dt=x_coord_test.dtype)
    )
    print(
        "  y_train      : {s} dtype={dt}".format(s=y_train.shape, dt=y_train.dtype)
    )
    print(
        "  y_val        : {s} dtype={dt}".format(s=y_val.shape, dt=y_val.dtype)
    )
    print(
        "  y_test       : {s} dtype={dt}".format(s=y_test.shape, dt=y_test.dtype)
    )
    print("===========================================================")

    if return_conditions:
        return (
            x_lidar_train,
            x_lidar_val,
            x_lidar_test,
            x_coord_train,
            x_coord_val,
            x_coord_test,
            y_train,
            y_val,
            y_test,
            condition_train,
            condition_val,
            condition_test,
        )

    return (
        x_lidar_train,
        x_lidar_val,
        x_lidar_test,
        x_coord_train,
        x_coord_val,
        x_coord_test,
        y_train,
        y_val,
        y_test,
    )


def load_dataset_raw_sparse_labels_by_condition(
    *,
    s008_coord_csv: str,
    s008_lidar_folder: str,
    s008_beam_output_path: str,
    s009_coord_csv: str,
    s009_lidar_folder: str,
    s009_beam_output_path: str,
    data_seed: int,
    report_label_coverage: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load Raymobtime S008 and S009 raw datasets and return per-dataset LOS and NLOS subsets.

    This variant keeps both the dataset (S008/S009) and condition (LOS/NLOS) separate and
    returns the full arrays without performing any internal splitting.

    Returns:
        Tuple containing four elements in this order (each is ``(lidar, coord, labels)``):

        1. S008 LOS
        2. S009 LOS
        3. S008 NLOS
        4. S009 NLOS
    """

    def _label_coverage(labels: np.ndarray, split_name: str) -> None:
        """Print how many classes appear in a split."""
        if labels.size == 0:
            print(f"{split_name}: no labels present.")
            return
        num_classes = int(labels.max()) + 1
        counts = np.bincount(labels.astype(int), minlength=num_classes)
        present = counts[counts > 0]
        print(f"{split_name}: {present.size}/{counts.size} classes present")
        print(f"{split_name} counts: {counts.tolist()}")

    print("======= load_dataset_raw_sparse_labels_by_condition =======")
    print("Loading S008 dataset with LOS/NLOS split...")
    s008_result = load_raymobtime_dataset(
        coord_csv_path=s008_coord_csv,
        lidar_folder=s008_lidar_folder,
        beam_output_path=s008_beam_output_path,
        split_los_nlos=True,
    )
    print("Loading S009 dataset with LOS/NLOS split...")
    s009_result = load_raymobtime_dataset(
        coord_csv_path=s009_coord_csv,
        lidar_folder=s009_lidar_folder,
        beam_output_path=s009_beam_output_path,
        split_los_nlos=True,
    )

    (s008_coord_los, s008_lidar_los, s008_beam_los), (s008_coord_nlos, s008_lidar_nlos, s008_beam_nlos) = s008_result
    (s009_coord_los, s009_lidar_los, s009_beam_los), (s009_coord_nlos, s009_lidar_nlos, s009_beam_nlos) = s009_result

    print("---------------- Per-dataset LOS/NLOS summary ----------------")
    print(
        "S008 -> LOS: {los}, NLOS: {nlos}".format(
            los=s008_coord_los.shape[0], nlos=s008_coord_nlos.shape[0]
        )
    )
    print(
        "S009 -> LOS: {los}, NLOS: {nlos}".format(
            los=s009_coord_los.shape[0], nlos=s009_coord_nlos.shape[0]
        )
    )
    print("--------------------------------------------------------------")

    if s008_coord_los.size == 0 or s008_coord_nlos.size == 0:
        raise ValueError("S008 must provide both LOS and NLOS samples to create splits.")
    if s009_coord_los.size == 0 or s009_coord_nlos.size == 0:
        raise ValueError("S009 must provide both LOS and NLOS samples to create splits.")

    print("Converting beam tensors to sparse labels...")
    s008_y_los = convert_to_sparse_labels(np.asarray(s008_beam_los, dtype=np.float32))
    s008_y_nlos = convert_to_sparse_labels(np.asarray(s008_beam_nlos, dtype=np.float32))
    s009_y_los = convert_to_sparse_labels(np.asarray(s009_beam_los, dtype=np.float32))
    s009_y_nlos = convert_to_sparse_labels(np.asarray(s009_beam_nlos, dtype=np.float32))

    s008_lidar_los = np.asarray(s008_lidar_los, dtype=np.float32)
    s008_lidar_nlos = np.asarray(s008_lidar_nlos, dtype=np.float32)
    s009_lidar_los = np.asarray(s009_lidar_los, dtype=np.float32)
    s009_lidar_nlos = np.asarray(s009_lidar_nlos, dtype=np.float32)

    s008_coord_los = np.asarray(s008_coord_los, dtype=np.float32)
    s008_coord_nlos = np.asarray(s008_coord_nlos, dtype=np.float32)
    s009_coord_los = np.asarray(s009_coord_los, dtype=np.float32)
    s009_coord_nlos = np.asarray(s009_coord_nlos, dtype=np.float32)

    print("==================== Dataset summary ====================")
    print(f"S008 LOS -> lidar: {s008_lidar_los.shape}, coord: {s008_coord_los.shape}, labels: {s008_y_los.shape}")
    print(f"S009 LOS -> lidar: {s009_lidar_los.shape}, coord: {s009_coord_los.shape}, labels: {s009_y_los.shape}")
    print(f"S008 NLOS -> lidar: {s008_lidar_nlos.shape}, coord: {s008_coord_nlos.shape}, labels: {s008_y_nlos.shape}")
    print(f"S009 NLOS -> lidar: {s009_lidar_nlos.shape}, coord: {s009_coord_nlos.shape}, labels: {s009_y_nlos.shape}")

    if report_label_coverage:
        print("================ Label coverage per condition ================")
        _label_coverage(s008_y_los, "S008 LOS")
        _label_coverage(s009_y_los, "S009 LOS")
        _label_coverage(s008_y_nlos, "S008 NLOS")
        _label_coverage(s009_y_nlos, "S009 NLOS")
        print("=============================================================")

    return (
        (s008_lidar_los, s008_coord_los, s008_y_los),
        (s009_lidar_los, s009_coord_los, s009_y_los),
        (s008_lidar_nlos, s008_coord_nlos, s008_y_nlos),
        (s009_lidar_nlos, s009_coord_nlos, s009_y_nlos),
    )


def load_dataset_raw_sparse_labels_s008_only(
    *,
    s008_coord_csv: str,
    s008_lidar_folder: str,
    s008_beam_output_path: str,
    data_seed: int,
    report_label_coverage: bool = True,
    remove_null_labels: bool = False,
    return_conditions: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load the full Raymobtime S008 dataset (raw) and convert beam tensors to sparse labels.

    This helper mirrors :func:`load_dataset_raw_sparse_labels_s009_only` but only processes
    the S008 files, returning the combined LOS and NLOS samples shuffled with ``data_seed``.
    """

    def _shuffle_together(rng: np.random.Generator, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        indices = np.arange(arrays[0].shape[0])
        rng.shuffle(indices)
        return tuple(array[indices] for array in arrays)

    print("============= load_dataset_raw_sparse_labels_s008_only =============")
    s008_result = load_raymobtime_dataset(
        coord_csv_path=s008_coord_csv,
        lidar_folder=s008_lidar_folder,
        beam_output_path=s008_beam_output_path,
        split_los_nlos=True,
    )
    (coord_los, lidar_los, beam_los), (coord_nlos, lidar_nlos, beam_nlos) = s008_result

    if coord_los.size == 0 or coord_nlos.size == 0:
        raise ValueError("S008 must provide both LOS and NLOS samples to compute metrics.")

    print("---------------- S008 LOS/NLOS summary ----------------")
    print(f"S008 -> LOS: {coord_los.shape[0]}, NLOS: {coord_nlos.shape[0]}")
    print(f"       lidar(LOS): {lidar_los.shape}, lidar(NLOS): {lidar_nlos.shape}")
    print(f"       coord(LOS): {coord_los.shape}, coord(NLOS): {coord_nlos.shape}")
    print(f"       beam(LOS): {beam_los.shape}, beam(NLOS): {beam_nlos.shape}")
    print("-------------------------------------------------------")

    beam_los = beam_los.astype(np.float32)
    beam_nlos = beam_nlos.astype(np.float32)
    y_los = convert_to_sparse_labels(beam_los)
    y_nlos = convert_to_sparse_labels(beam_nlos)

    if remove_null_labels:
        print("Compacting S008 labels to remove unseen IDs...")
        all_labels = np.concatenate((y_los, y_nlos))
        present_labels = np.unique(all_labels)
        label_map = np.full(int(present_labels.max()) + 1, -1, dtype=int)
        label_map[present_labels] = np.arange(present_labels.size, dtype=int)
        y_los = label_map[y_los]
        y_nlos = label_map[y_nlos]
        print(
            "Label remap -> present: {present}, new range: [0, {max_new}], removed gaps: {removed}".format(
                present=present_labels.size,
                max_new=present_labels.size - 1,
                removed=(int(present_labels.max()) + 1) - present_labels.size,
            )
        )

    lidar_los = np.asarray(lidar_los, dtype=np.float32)
    lidar_nlos = np.asarray(lidar_nlos, dtype=np.float32)
    coord_los = np.asarray(coord_los, dtype=np.float32)
    coord_nlos = np.asarray(coord_nlos, dtype=np.float32)

    condition_los = np.full(y_los.shape, "LOS", dtype=object)
    condition_nlos = np.full(y_nlos.shape, "NLOS", dtype=object)

    rng = np.random.default_rng(data_seed)
    print(f"Shuffling S008 subsets with seed {data_seed}...")
    if return_conditions:
        lidar_los, coord_los, y_los, condition_los = _shuffle_together(
            rng, lidar_los, coord_los, y_los, condition_los
        )
        lidar_nlos, coord_nlos, y_nlos, condition_nlos = _shuffle_together(
            rng, lidar_nlos, coord_nlos, y_nlos, condition_nlos
        )
    else:
        lidar_los, coord_los, y_los = _shuffle_together(rng, lidar_los, coord_los, y_los)
        lidar_nlos, coord_nlos, y_nlos = _shuffle_together(rng, lidar_nlos, coord_nlos, y_nlos)

    x_lidar = np.concatenate((lidar_los, lidar_nlos), axis=0).astype(np.float32)
    x_coord = np.concatenate((coord_los, coord_nlos), axis=0).astype(np.float32)
    y = np.concatenate((y_los, y_nlos), axis=0)
    if report_label_coverage and y.size:
        num_classes = int(y.max()) + 1
        counts = np.bincount(y.astype(int), minlength=num_classes)
        present = counts[counts > 0]
        print("================ S008 label coverage ===================")
        print(f"{present.size}/{counts.size} classes present")
        print(f"Per-class counts: {counts.tolist()}")
        print("========================================================")
    if return_conditions:
        condition = np.concatenate((condition_los, condition_nlos), axis=0)

    print("================ S008 combined summary =================")
    print(f"lidar shape: {x_lidar.shape}")
    print(f"coord shape: {x_coord.shape}")
    print(f"labels shape: {y.shape}")
    print("========================================================")

    if return_conditions:
        return x_lidar, x_coord, y, condition

    return x_lidar, x_coord, y


def load_dataset_raw_sparse_labels_s009_only(
    *,
    s009_coord_csv: str,
    s009_lidar_folder: str,
    s009_beam_output_path: str,
    data_seed: int,
    report_label_coverage: bool = True,
    remove_null_labels: bool = False,
    return_conditions: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Load the full Raymobtime S009 dataset (raw) and convert beam tensors to sparse labels.

    Unlike :func:`load_dataset_raw_sparse_labels`, this helper does not merge S009 with
    S008 or create train/validation splits. It simply loads every LOS and NLOS sample
    from S009, converts labels to sparse format, and shuffles the combined dataset
    with ``data_seed`` for deterministic ordering. When ``return_conditions`` is True,
    a companion array with the LOS/NLOS condition is returned alongside the features
    and labels.
    """

    def _shuffle_together(rng: np.random.Generator, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Shuffle multiple arrays in unison keeping row alignment."""
        indices = np.arange(arrays[0].shape[0])
        rng.shuffle(indices)
        return tuple(array[indices] for array in arrays)

    print("============= load_dataset_raw_sparse_labels_s009_only =============")
    s009_result = load_raymobtime_dataset(
        coord_csv_path=s009_coord_csv,
        lidar_folder=s009_lidar_folder,
        beam_output_path=s009_beam_output_path,
        split_los_nlos=True,
    )
    (coord_los, lidar_los, beam_los), (coord_nlos, lidar_nlos, beam_nlos) = s009_result

    if coord_los.size == 0 or coord_nlos.size == 0:
        raise ValueError("S009 must provide both LOS and NLOS samples to compute metrics.")

    print("---------------- S009 LOS/NLOS summary ----------------")
    print(f"S009 -> LOS: {coord_los.shape[0]}, NLOS: {coord_nlos.shape[0]}")
    print(f"       lidar(LOS): {lidar_los.shape}, lidar(NLOS): {lidar_nlos.shape}")
    print(f"       coord(LOS): {coord_los.shape}, coord(NLOS): {coord_nlos.shape}")
    print(f"       beam(LOS): {beam_los.shape}, beam(NLOS): {beam_nlos.shape}")
    print("-------------------------------------------------------")

    beam_los = beam_los.astype(np.float32)
    beam_nlos = beam_nlos.astype(np.float32)
    y_los = convert_to_sparse_labels(beam_los)
    y_nlos = convert_to_sparse_labels(beam_nlos)

    if remove_null_labels:
        print("Compacting S009 labels to remove unseen IDs...")
        all_labels = np.concatenate((y_los, y_nlos))
        present_labels = np.unique(all_labels)
        label_map = np.full(int(present_labels.max()) + 1, -1, dtype=int)
        label_map[present_labels] = np.arange(present_labels.size, dtype=int)
        y_los = label_map[y_los]
        y_nlos = label_map[y_nlos]
        print(
            "Label remap -> present: {present}, new range: [0, {max_new}], removed gaps: {removed}".format(
                present=present_labels.size,
                max_new=present_labels.size - 1,
                removed=(int(present_labels.max()) + 1) - present_labels.size,
            )
        )

    condition_los = np.full(y_los.shape, "LOS", dtype=object)
    condition_nlos = np.full(y_nlos.shape, "NLOS", dtype=object)

    rng = np.random.default_rng(data_seed)
    print(f"Shuffling S009 subsets with seed {data_seed}...")
    if return_conditions:
        lidar_los, coord_los, y_los, condition_los = _shuffle_together(
            rng, lidar_los, coord_los, y_los, condition_los
        )
        lidar_nlos, coord_nlos, y_nlos, condition_nlos = _shuffle_together(
            rng, lidar_nlos, coord_nlos, y_nlos, condition_nlos
        )
    else:
        lidar_los, coord_los, y_los = _shuffle_together(rng, lidar_los, coord_los, y_los)
        lidar_nlos, coord_nlos, y_nlos = _shuffle_together(rng, lidar_nlos, coord_nlos, y_nlos)

    x_lidar = np.concatenate((lidar_los, lidar_nlos), axis=0).astype(np.float32)
    x_coord = np.concatenate((coord_los, coord_nlos), axis=0).astype(np.float32)
    y = np.concatenate((y_los, y_nlos), axis=0)
    if return_conditions:
        condition = np.concatenate((condition_los, condition_nlos), axis=0)

    print("================ S009 combined summary =================")
    print(f"lidar shape: {x_lidar.shape}")
    print(f"coord shape: {x_coord.shape}")
    print(f"labels shape: {y.shape}")
    print("========================================================")

    if report_label_coverage and y.size:
        num_classes = int(y.max()) + 1
        counts = np.bincount(y.astype(int), minlength=num_classes)
        present = counts[counts > 0]
        print("================ S009 label coverage ===================")
        print(f"{present.size}/{counts.size} classes present")
        print(f"Per-class counts: {counts.tolist()}")
        print("========================================================")

    if return_conditions:
        return x_lidar, x_coord, y, condition

    return x_lidar, x_coord, y


# --------------------------------------------------------------------------------------
# Script entry point for manual testing
# --------------------------------------------------------------------------------------


# if __name__ == "__main__":
#     DEFAULT_COORD_CSV = (
#         "/media/matheus/SSD-2/matheus/datasets/RayWise/Raymobtime_s008/raw_data/" "CoordVehiclesRxPerScene_s008.csv"
#     )
#     DEFAULT_LIDAR_FOLDER = (
#         "/media/matheus/SSD-2/matheus/datasets/RayWise/Raymobtime_s008/processed_raw_data/" "lidar_data_s008"
#     )
#     DEFAULT_BEAM_OUTPUT = (
#         "/media/matheus/SSD-2/matheus/datasets/RayWise/Raymobtime_s008/baseline_data/beam_output/"
#         "beams_output_s008.npz"
#     )

#     print("Running standalone reconstruction using default dataset paths...")

#     try:
#         result = load_raymobtime_dataset(
#             coord_csv_path=DEFAULT_COORD_CSV,
#             lidar_folder=DEFAULT_LIDAR_FOLDER,
#             beam_output_path=DEFAULT_BEAM_OUTPUT,
#             split_los_nlos=True,
#         )
#         if isinstance(result, tuple) and len(result) == 2:
#             (coord_los, lidar_los, beam_los), (coord_nlos, lidar_nlos, beam_nlos) = result
#             print("LOS and NLOS datasets loaded successfully.")
#         else:
#             coord_input, lidar_input, beam_output = result
#             print("Unified dataset loaded successfully.")

#     except Exception as exc:  # pragma: no cover - manual debugging aid
#         print("An error occurred during reconstruction:")
#         print(exc)
