from typing import Dict

import numpy as np


def load_npz_file(file_path: str, allow_pickle: bool = False) -> Dict[str, np.ndarray]:
    """Load a NumPy .npz archive and return its contents as a dict of arrays.

    This function wraps `numpy.load` in a context manager to ensure the
    underlying file handle is closed promptly, and converts the returned
    NpzFile into a plain dict.

    Args:
        file_path (str):
            Path to the .npz file to load.
        allow_pickle (bool):
            Whether to allow loading object arrays saved in pickle format.
            Only enable this if you trust the file source. Defaults to False.

    Returns:
        Dict[str, np.ndarray]:
            A mapping from each array name (as a key in the archive) to its
            corresponding NumPy array.

    Raises:
        FileNotFoundError:
            If `file_path` does not exist.
        ValueError:
            If the file is not a valid .npz archive or another loading error occurs.
    """
    try:
        # np.load returns an NpzFile, which is a dict-like object
        with np.load(file_path, allow_pickle=allow_pickle) as npz:
            # Build a plain dict so the file handle can be closed immediately
            return {name: npz[name] for name in npz.files}
    except FileNotFoundError as e:
        # Clearly indicate missing file
        raise FileNotFoundError(f"Could not find NPZ file: {file_path}") from e
    except Exception as e:
        # Catch corrupt archives, unsupported formats, etc.
        raise ValueError(f"Failed to load NPZ file '{file_path}': {e}") from e


def main() -> None:
    """
    BEAMS OUTPUT TEST -> 'output_classification': shape=(9638, 8, 32), dtype=complex128
    BEAMS OUTPUT TRAIN -> 'output_classification': shape=(9234, 8, 32), dtype=complex128
    BEAMS OUTPUT VALIDATION -> 'output_classification': shape=(1960, 8, 32), dtype=complex128
    
    COORD INPUT TEST -> 'coordinates': shape=(9638, 2), dtype=float64
    COORD INPUT TRAIN -> 'coordinates': shape=(9234, 2), dtype=float64
    COORD INPUT VALIDATION -> 'coordinates': shape=(1960, 2), dtype=float64
    
    LIDAR INPUT TEST -> 'input': shape=(9638, 20, 200, 10), dtype=int8
    LIDAR INPUT TRAIN -> 'input': shape=(9234, 20, 200, 10), dtype=int8
    LIDAR INPUT VALIDATION -> 'input': shape=(1960, 20, 200, 10), dtype=int8
    
    """
    npz_path = "./batool/lidar_input/lidar_test.npz"
    try:
        data_dict = load_npz_file(npz_path)
    except Exception as e:
        # Fail fast with a clear message
        print(f"Error: {e}")
        return

    # Inspect the contents
    for array_name, array in data_dict.items():
        print(f"• '{array_name}': shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
