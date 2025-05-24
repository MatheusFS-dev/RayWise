import os
import numpy as np


def convert_to_sparse_labels(y: np.ndarray) -> np.ndarray:
    """
    Converts beam score targets to sparse integer labels suitable for SparseCategoricalCrossentropy loss.

    Args:
        y (np.ndarray): Original beam scores with shape (N, 8, 32).

    Returns:
        np.ndarray: Array of integer labels with shape (N,), where each label corresponds
                    to the index (flattened over 8x32) of the maximum score.
    """
    # Reshape the input so that each sample becomes a 1D array (e.g., 256 elements)
    y_flat = y.reshape(y.shape[0], -1)
    # For each sample, return the index of the maximum value
    labels = np.argmax(y_flat, axis=1)
    return labels


def load_dataset_sparse_labels():
    """
    Loads the dataset from the specified directory and converts the labels to sparse format.
    """

    # Define the base directory for data files
    DATA_DIR = "./data/s008"

    # —————————————————————————————— Load Train Data ————————————————————————————— #
    beam_output_train_path = os.path.join(DATA_DIR, "beam_output", "beam_output_train.npz")
    coord_input_train_path = os.path.join(DATA_DIR, "coord_input", "coord_train.npz")
    lidar_input_train_path = os.path.join(DATA_DIR, "lidar_input", "lidar_train.npz")

    # Load the data from the .npy files
    s008_y_train = np.load(beam_output_train_path)["output_classification"]
    s008_coord_input = np.load(coord_input_train_path)["coordinates"]
    s008_lidar_input = np.load(lidar_input_train_path)["input"]

    # Cast target beam outputs to float - REMOVING USELESS IMAG PART
    s008_y_train = s008_y_train.astype(np.float32)
    s008_coord_input = s008_coord_input.astype(np.float32)

    print(f"Shape before conversion: {s008_y_train.shape}")
    s008_y_train = convert_to_sparse_labels(s008_y_train)
    print(f"Shape after conversion: {s008_y_train.shape}")

    # Print the shapes of the loaded data
    print(f"y_train shape: {s008_y_train.shape}")
    print(f"coord_input shape: {s008_coord_input.shape}")
    print(f"lidar_input shape: {s008_lidar_input.shape}")

    # ——————————————————————————— Load Validation Data ——————————————————————————— #
    beam_output_val_path = os.path.join(DATA_DIR, "beam_output", "beam_output_val.npz")
    coord_input_val_path = os.path.join(DATA_DIR, "coord_input", "coord_val.npz")
    lidar_input_val_path = os.path.join(DATA_DIR, "lidar_input", "lidar_val.npz")

    # Load the data from the .npy files
    s008_y_val = np.load(beam_output_val_path)["output_classification"]
    s008_coord_input_val = np.load(coord_input_val_path)["coordinates"]
    s008_lidar_input_val = np.load(lidar_input_val_path)["input"]

    # Cast target beam outputs to float - REMOVING USELESS IMAG PART
    s008_y_val = s008_y_val.astype(np.float32)
    s008_coord_input_val = s008_coord_input_val.astype(np.float32)

    print(f"Shape before conversion: {s008_y_val.shape}")
    s008_y_val = convert_to_sparse_labels(s008_y_val)
    print(f"Shape after conversion: {s008_y_val.shape}")

    # Print the shapes of the loaded data
    print(f"y_val shape: {s008_y_val.shape}")
    print(f"coord_input_val shape: {s008_coord_input_val.shape}")
    print(f"lidar_input_val shape: {s008_lidar_input_val.shape}")

    # —————————————————————— Merge train and validation data ————————————————————— #
    s008_y_train = np.concatenate((s008_y_train, s008_y_val), axis=0)
    s008_coord_input = np.concatenate((s008_coord_input, s008_coord_input_val), axis=0)
    s008_lidar_input = np.concatenate((s008_lidar_input, s008_lidar_input_val), axis=0)

    # Print the shapes of the merged data
    print(f"y_train shape: {s008_y_train.shape}")
    print(f"coord_input shape: {s008_coord_input.shape}")
    print(f"lidar_input shape: {s008_lidar_input.shape}")

    # Define the base directory for data files
    DATA_DIR = "./data/s009"

    # Construct full paths to the .npy files
    beam_output_path = os.path.join(DATA_DIR, "beam_output", "beam_output.npz")
    coord_input_path = os.path.join(DATA_DIR, "coord_input", "coord_input.npz")
    lidar_input_path = os.path.join(DATA_DIR, "lidar_input", "lidar_input.npz")

    # Load the data from the .npy files
    s009_y = np.load(beam_output_path)["output_classification"]
    s009_coord_input = np.load(coord_input_path)["coordinates"]
    s009_lidar_input = np.load(lidar_input_path)["input"]

    # Cast target beam outputs to float - REMOVING USELESS IMAG PART
    s009_y = s009_y.astype(np.float32)
    s009_coord_input = s009_coord_input.astype(np.float32)

    print(f"Shape before conversion: {s009_y.shape}")
    s009_y = convert_to_sparse_labels(s009_y)
    print(f"Shape after conversion: {s009_y.shape}")

    # Print the shapes of the loaded data
    print(f"y shape: {s009_y.shape}")
    print(f"coord_input shape: {s009_coord_input.shape}")
    print(f"lidar_input shape: {s009_lidar_input.shape}")

    return (
        s008_coord_input,
        s008_lidar_input,
        s008_y_train,
        s009_coord_input,
        s009_lidar_input,
        s009_y,
    )
