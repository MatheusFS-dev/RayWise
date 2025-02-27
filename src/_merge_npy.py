# ---------------------------------------------------------------------------- #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ---------------------------------------------------------------------------- #

import os
import numpy as np
from typing import List

def merge_npz_files(npz_files: List[str], output_path: str) -> None:
    """
    Merges multiple .npz files into a single .npz file.

    Args:
        npz_files (List[str]): List of .npz file paths to merge.
        output_path (str): Path to save the merged .npz file.
    """
    merged_data = {}

    for file_path in npz_files:
        if not file_path.endswith(".npz"):
            continue

        try:
            data = np.load(file_path)

            # Merge keys from different files while keeping their filename as a prefix
            filename = os.path.basename(file_path).replace(".npz", "")
            for key in data.files:
                merged_key = f"{filename}/{key}"  # Unique key per file
                merged_data[merged_key] = data[key]

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Save merged data as a compressed .npz file
    if merged_data:
        np.savez_compressed(output_path, **merged_data)
        print(f"Successfully saved merged file: {output_path}")
    else:
        print(f"No valid data to merge, skipping save for: {output_path}")

if __name__ == "__main__":
    root_path = "./data/s008"

    # Get all subdirectories inside the root path
    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_path, subfolder)

        # Collect all .npz files in the subfolder
        npz_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(".npz")]

        if npz_files:
            output_filename = os.path.join(subfolder_path, f"{subfolder}_merged.npz")
            merge_npz_files(npz_files, output_filename)
