import os
import re
import glob
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd


def load_hdf5_file(
    file_path: str,
    dataset_key: str = "allEpisodeData",
) -> np.ndarray:
    """
    Load a single Raymobtime .hdf5 episode file and return its dataset as a NumPy array.

    Args:
        file_path: Full path to the .hdf5 file.
        dataset_key: Dataset key inside the HDF5 file to load. Defaults to "allEpisodeData".

    Returns:
        A NumPy array with the contents of the specified dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the dataset key is not present in the file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            available = ", ".join(list(f.keys()))
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in {file_path}. Available keys: {available}"
            )
        data = f[dataset_key][...]
    return data


def _extract_episode_number(filename: str) -> Union[int, None]:
    """Extract episode number from filenames like '..._e123.hdf5'."""
    m = re.search(r"[_\-]e(\d+)\.hdf5$", filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def load_hdf5_folder(
    folder_path: str,
    dataset_key: str = "allEpisodeData",
    pattern: str = "*.hdf5",
    sort_by_episode: bool = True,
) -> Dict[Union[int, str], np.ndarray]:
    """
    Load all .hdf5 episode files in a folder.

    Args:
        folder_path: Path to the folder containing .hdf5 files.
        dataset_key: Dataset key inside each HDF5 file to load. Defaults to "allEpisodeData".
        pattern: Glob pattern to match files. Defaults to "*.hdf5".
        sort_by_episode: If True, sort using the episode number extracted from the
            filename pattern '*_e<ep>.hdf5'. Files without a parsable episode use
            their basename as key and are placed after numeric keys.

    Returns:
        A dict mapping episode number (int) or filename (str) to the loaded NumPy array.

    Raises:
        FileNotFoundError: If the folder does not exist or no files match the pattern.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    file_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not file_paths:
        raise FileNotFoundError(f"No .hdf5 files found in {folder_path!r} with pattern {pattern!r}")

    # Optionally sort by parsed episode number, keeping unparsed at the end
    if sort_by_episode:

        def sort_key(p: str) -> Tuple[int, str]:
            ep = _extract_episode_number(os.path.basename(p))
            # Use large sentinel for files without episode number to push them to the end
            return (ep if ep is not None else 1_000_000_000, os.path.basename(p))

        file_paths = sorted(file_paths, key=sort_key)

    data_map: Dict[Union[int, str], np.ndarray] = {}
    for p in file_paths:
        key: Union[int, str]
        ep = _extract_episode_number(os.path.basename(p))
        key = ep if ep is not None else os.path.basename(p)
        data_map[key] = load_hdf5_file(p, dataset_key=dataset_key)

    return data_map


if __name__ == "__main__":
    # Example usage preserved from the original script, refactored for clarity.
    # Adjust paths and limits as needed.

    HDF5_FOLDER_PATH = (
        "/home/matheus/src/datasets/RayWise/Raymobtime_s008/raw_data/ray_tracing_data_s008_carrier60GHz/"
    )
    CSV_FILE_PATH = (
        "/home/matheus/src/datasets/RayWise/Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
    )

    # load an npz file:
    npz_file_path = "/home/matheus/src/datasets/RayWise/Raymobtime_s008/processed_raw_data/lidar_data_s008/obstacles_e_0.npz"
    npz_data = np.load(npz_file_path, allow_pickle=True)

    # print shape of arrays in the npz file
    for key in npz_data.files:
        print(f"Key: {key}, Shape: {npz_data[key].shape}")

    Episodios = 10  # 2085 (total of 2086)
    Cenas = 1  # Number of scenes per episode
    Pares = 10

    # Load coordinate CSV and prepare a 'Rays' column to receive arrays
    df_coord = pd.read_csv(CSV_FILE_PATH)
    df_coord["Rays"] = np.nan
    df_coord["Rays"] = df_coord["Rays"].astype(object)

    # Iterate over a few example episodes
    for ep in range(Episodios):
        hdf5_name = os.path.join(
            HDF5_FOLDER_PATH,
            f"rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e{ep}.hdf5",
        )
        hdf5_data = load_hdf5_file(hdf5_name, dataset_key="allEpisodeData")
        # hdf5_data[0][0][0][0]  # Cena 0, Par 0, Raio 0, Parâmetro 0

        for c in range(Cenas):
            for p in range(Pares):
                idx = ep * Cenas * Pares + c * Pares + p
                df_coord.at[idx, "Rays"] = hdf5_data[c][p].copy()

    print(df_coord.iloc[1])
    print(df_coord.iloc[1]["Rays"][0][0])
    # Acesso a primeira linha do dataframe, que corresponde ao Ep 0, Sc 0, e veículo 1
    # Parâmetros referente aos raios, raio 0, parâmetro 0 (Received power)
