# ---------------------------------------------------------------------------- #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ---------------------------------------------------------------------------- #

import numpy as np

def inspect_npy_file(file_path: str, num_elements: int = 5) -> None:
    """
    Reads a .npy file and prints relevant information about the dataset.

    Parameters:
    - file_path (str): Path to the .npy file.
    - num_elements (int): Number of elements to preview from the dataset.

    Returns:
    - None
    """
    try:
        # Load the .npy file
        data = np.load(file_path, allow_pickle=True)

        # Print dataset properties
        print(f"File: {file_path}")
        print(f"Type: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"Data Type: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")

        # Determine the length if applicable
        try:
            length = len(data)
            print(f"Length: {length}")
        except TypeError:
            print("Length: Not applicable")

        # Print first few elements
        print("\nFirst few elements:")
        print(data[:num_elements] if hasattr(data, '__getitem__') else "Cannot display elements")

        # Additional insights
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            print("\nStatistical Overview:")
            print(f"Min: {np.min(data)}")
            print(f"Max: {np.max(data)}")
            print(f"Mean: {np.mean(data)}")
            print(f"Std Dev: {np.std(data)}")

    except Exception as e:
        print(f"Error reading the .npy file: {e}")
        

if __name__ == "__main__":
    # Dict of file paths
    file_paths = {
        "beam_output": "./data/s009/beam_output/output_classification.npy",
        "coord_input": "./data/s009/coord_input/coordinates.npy",
        "image_input": "./data/s009/image_input/inputs.npy",
        "image_v2_input": "./data/s009/image_v2_input/inputs.npy",
        "lidar_input": "./data/s009/lidar_input/input.npy",
    }
    
    # For each file path print name and inspect
    for file_name, file_path in file_paths.items():
        print(f"\n{file_name}")
        inspect_npy_file(file_path, num_elements=1)
        print("----------------------------------------------------------")

