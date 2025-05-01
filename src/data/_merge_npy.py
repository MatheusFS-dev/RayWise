import os
import numpy as np
from typing import List

def extract_npz_to_npy(npz_file: str, output_dir: str) -> List[str]:
    """
    Extracts the contents of an .npz file and saves them as individual .npy files.

    Args:
        npz_file (str): Path to the .npz file.
        output_dir (str): Directory to save the extracted .npy files.

    Returns:
        List[str]: Paths of the extracted .npy files.
    """
    extracted_files = []
    
    try:
        data = np.load(npz_file)
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        for key in data.files:
            npy_path = os.path.join(output_dir, f"{key}.npy")
            np.save(npy_path, data[key])  # Save each array as .npy
            extracted_files.append(npy_path)

    except Exception as e:
        print(f"Error extracting {npz_file}: {e}")

    return extracted_files

def merge_npy_files(npy_files: List[str], output_path: str) -> None:
    """
    Merges multiple .npy files into a single .npz file.

    Args:
        npy_files (List[str]): List of .npy file paths to merge.
        output_path (str): Path to save the merged .npz file.
    """
    merged_data = {}

    for npy_file in npy_files:
        try:
            key_name = os.path.basename(npy_file).replace(".npy", "")
            merged_data[key_name] = np.load(npy_file)

        except Exception as e:
            print(f"Error loading {npy_file}: {e}")

    if merged_data:
        np.savez_compressed(output_path, **merged_data)
        print(f"Successfully saved merged file: {output_path}")
    else:
        print(f"No valid data to merge, skipping save for: {output_path}")

if __name__ == "__main__":
    root_path = "./data/s008"

    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_path, subfolder)

        # Collect all .npz files in the subfolder
        npz_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(".npz")]

        extracted_npy_files = []
        
        for npz_file in npz_files:
            print(f"Extracting {npz_file}...")
            npy_output_dir = os.path.join(subfolder_path, "extracted_npy")
            extracted_npy_files.extend(extract_npz_to_npy(npz_file, npy_output_dir))

        if extracted_npy_files:
            output_filename = os.path.join(subfolder_path, f"{subfolder}_merged.npz")
            merge_npy_files(extracted_npy_files, output_filename)
