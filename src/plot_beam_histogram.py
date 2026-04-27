"""Plot a histogram of Raymobtime beam indices.

This script loads Raymobtime beam output tensors from ``.npz`` files,
converts each sample to a sparse beam index in ``[0, 255]`` (8x32 flattened),
and plots a histogram showing how frequently each beam appears.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


# Global paths: edit these two variables if your dataset/output location changes.
DATA_ROOT = "./data"
OUTPUT_ROOT = "./runs/beams"


def convert_beam_scores_to_sparse_labels(beam_scores: np.ndarray) -> np.ndarray:
    """Convert beam score tensors to sparse beam indices.

    Args:
        beam_scores (np.ndarray): Beam score tensor with shape ``(N, 8, 32)``.
            Each sample stores the score of each beam pair. The conversion
            flattens the last two dimensions to 256 positions and applies
            ``argmax`` to select the dominant beam index for each sample.

    Returns:
        np.ndarray: Sparse integer labels with shape ``(N,)`` and values in
        ``[0, 255]``.

    Raises:
        ValueError: If ``beam_scores`` does not have three dimensions with
            shape ``(N, 8, 32)``.

    Examples:
        >>> scores = np.zeros((2, 8, 32), dtype=np.float32)
        >>> scores[0, 0, 0] = 1.0
        >>> scores[1, 7, 31] = 5.0
        >>> convert_beam_scores_to_sparse_labels(scores)
        array([  0, 255])
    """

    # Validate shape early so errors are explicit and easy to diagnose.
    if beam_scores.ndim != 3 or beam_scores.shape[1:] != (8, 32):
        raise ValueError(
            "Expected beam_scores with shape (N, 8, 32), but received "
            f"{beam_scores.shape}."
        )

    # Flatten the beam grid (8 x 32) into 256 positions and select the max.
    flattened_scores = beam_scores.reshape(beam_scores.shape[0], -1)
    sparse_labels = np.argmax(flattened_scores, axis=1)
    return sparse_labels


def load_npz_beam_scores(npz_path: str, key: str = "output_classification") -> np.ndarray:
    """Load beam score tensor from a Raymobtime ``.npz`` archive.

    Args:
        npz_path (str): Path to the ``.npz`` file.
        key (str): Array key to load from the archive. If this key is present,
            that array is loaded. If it is absent and the file contains a
            single array, the function falls back to that array. If it is
            absent and the file has multiple arrays, the function raises an
            error so the user can disambiguate.

    Returns:
        np.ndarray: Beam score tensor, typically shaped as ``(N, 8, 32)``.

    Raises:
        FileNotFoundError: If ``npz_path`` does not exist.
        KeyError: If ``key`` is missing and fallback is ambiguous.
        ValueError: If the archive has no arrays.

    Examples:
        >>> # Example call (path depends on your local dataset)
        >>> # scores = load_npz_beam_scores("./data/s008/beam_output/beam_output_train.npz")
        >>> # scores.shape
        >>> # (num_samples, 8, 32)
    """

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Beam output file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if len(data.files) == 0:
            raise ValueError(f"No arrays found in archive: {npz_path}")

        # Prefer the canonical Raymobtime key when available.
        if key in data:
            return data[key]

        # If only one key exists, fallback is deterministic and safe.
        if len(data.files) == 1:
            fallback_key = data.files[0]
            print(
                f"Key '{key}' not found in {npz_path}. "
                f"Using only available key: '{fallback_key}'."
            )
            return data[fallback_key]

        raise KeyError(
            f"Key '{key}' not found in {npz_path}. Available keys: {list(data.files)}"
        )


def collect_sparse_labels_from_files(npz_paths: List[str]) -> np.ndarray:
    """Load multiple beam files and concatenate sparse labels.

    Args:
        npz_paths (List[str]): List of ``.npz`` files that contain beam score
            tensors. Every file is loaded, converted to sparse labels, and
            concatenated in the same order.

    Returns:
        np.ndarray: Concatenated sparse beam labels with shape ``(total_N,)``.

    Raises:
        ValueError: If no labels are produced (for example, empty input list or
            empty arrays in every file).

    Examples:
        >>> # labels = collect_sparse_labels_from_files([
        >>> #     "./data/s008/beam_output/beam_output_train.npz",
        >>> #     "./data/s008/beam_output/beam_output_val.npz",
        >>> # ])
        >>> # labels.ndim
        >>> # 1
    """

    sparse_labels_per_file = []
    for npz_path in npz_paths:
        print(f"Loading beam scores from: {npz_path}")
        scores = load_npz_beam_scores(npz_path)
        labels = convert_beam_scores_to_sparse_labels(scores)
        print(f"  Loaded {labels.shape[0]} samples.")
        sparse_labels_per_file.append(labels)

    if not sparse_labels_per_file:
        raise ValueError("No beam files provided. Please populate npz_paths.")

    concatenated_labels = np.concatenate(sparse_labels_per_file, axis=0)
    if concatenated_labels.size == 0:
        raise ValueError("Loaded files but no samples were found.")
    return concatenated_labels


def plot_beam_histogram(labels: np.ndarray, title: str = "Raymobtime Beam Histogram") -> None:
    """Plot histogram of sparse beam labels for the 256-beam codebook.

    Args:
        labels (np.ndarray): Sparse beam labels with values in ``[0, 255]``.
        title (str): Plot title shown in the generated figure.

    Returns:
        None: This function renders a Matplotlib figure.

    Raises:
        ValueError: If labels are not a one-dimensional array.

    Examples:
        >>> # plot_beam_histogram(np.array([0, 0, 1, 255]))
    """

    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels array, but received shape {labels.shape}.")

    # Use fixed bin edges so each beam index maps to exactly one histogram bar.
    bin_edges = np.arange(257) - 0.5
    plt.figure(figsize=(14, 5))
    plt.hist(labels, bins=bin_edges, edgecolor="black", linewidth=0.2)
    plt.title(title)
    plt.xlabel("Beam Index (0-255)")
    plt.ylabel("Frequency")
    plt.xlim(-0.5, 255.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_beam_histogram_counts(labels: np.ndarray, num_beams: int = 256) -> np.ndarray:
    """Compute per-beam counts from sparse labels.

    Args:
        labels (np.ndarray): Sparse beam labels with shape ``(N,)``.
        num_beams (int): Number of beams in the codebook. If set to ``256``,
            the output count vector has one slot for every Raymobtime beam
            index from 0 to 255. If set to another positive integer, the
            function still counts labels but validates them against
            ``[0, num_beams - 1]``.

    Returns:
        np.ndarray: Histogram counts with shape ``(num_beams,)``.

    Raises:
        ValueError: If ``labels`` is not one-dimensional, ``num_beams`` is not
            positive, or labels fall outside the valid index range.

    Examples:
        >>> compute_beam_histogram_counts(np.array([0, 0, 2]), num_beams=4)
        array([2, 0, 1, 0])
    """

    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels array, but received shape {labels.shape}.")
    if num_beams <= 0:
        raise ValueError(f"num_beams must be positive, but received {num_beams}.")
    if labels.size > 0 and (labels.min() < 0 or labels.max() >= num_beams):
        raise ValueError(
            f"Labels out of range. Expected values in [0, {num_beams - 1}], "
            f"but found min={labels.min()} and max={labels.max()}."
        )

    # Use bincount with minlength so every beam index appears in the output.
    return np.bincount(labels, minlength=num_beams)


def save_histogram_csv(labels: np.ndarray, csv_path: str, num_beams: int = 256) -> None:
    """Save beam histogram values to CSV.

    Args:
        labels (np.ndarray): Sparse beam labels with shape ``(N,)``.
        csv_path (str): Output CSV path. Parent directories are created if
            needed.
        num_beams (int): Number of beam bins to export. If ``256``, the CSV
            includes all Raymobtime beams from 0 to 255. If another value is
            supplied, the CSV range adapts to that codebook size.

    Returns:
        None: Writes a CSV file with columns
        ``beam_index,count,relative_frequency``.

    Raises:
        ValueError: Propagated from histogram computation when labels are
            invalid.

    Examples:
        >>> # save_histogram_csv(np.array([0, 1, 1]), "./runs/beams/example.csv")
    """

    # Ensure destination folder exists before writing the CSV file.
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    counts = compute_beam_histogram_counts(labels, num_beams=num_beams)
    total = int(counts.sum())
    relative_frequency = counts / total if total > 0 else np.zeros_like(counts, dtype=np.float64)

    # Stack all columns so each row is one beam index with count and ratio.
    table = np.column_stack(
        [
            np.arange(num_beams, dtype=np.int64),
            counts.astype(np.int64),
            relative_frequency.astype(np.float64),
        ]
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="beam_index,count,relative_frequency",
        comments="",
        fmt=["%d", "%d", "%.10f"],
    )


def save_histogram_figure(
    labels: np.ndarray,
    image_path: str,
    title: str,
    num_beams: int = 256,
) -> None:
    """Save a beam histogram figure to disk.

    Args:
        labels (np.ndarray): Sparse beam labels with shape ``(N,)``.
        image_path (str): Output image path (for example ``.png``).
        title (str): Title shown in the saved figure.
        num_beams (int): Number of bins in the histogram. If ``256``, each bar
            maps to one Raymobtime beam index. If another value is provided,
            the axis and bins adapt to the specified codebook size.

    Returns:
        None: Writes the histogram image to ``image_path``.

    Raises:
        ValueError: If labels are invalid or ``num_beams`` is not positive.

    Examples:
        >>> # save_histogram_figure(np.array([0, 1]), "./runs/beams/example.png", "Example")
    """

    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels array, but received shape {labels.shape}.")
    if num_beams <= 0:
        raise ValueError(f"num_beams must be positive, but received {num_beams}.")

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    # Use fixed-width bins centered on integer beam indices.
    bin_edges = np.arange(num_beams + 1) - 0.5
    plt.figure(figsize=(14, 5))
    plt.hist(labels, bins=bin_edges, edgecolor="black", linewidth=0.2)
    plt.title(title)
    plt.xlabel(f"Beam Index (0-{num_beams - 1})")
    plt.ylabel("Frequency")
    plt.xlim(-0.5, num_beams - 0.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    # Use global path variables so dataset/output locations are configured once.
    output_dir = OUTPUT_ROOT

    # Keep s008 separated as requested (train + val merged only inside s008).
    s008_files = [
        os.path.join(DATA_ROOT, "s008", "beam_output", "beam_output_train.npz"),
        os.path.join(DATA_ROOT, "s008", "beam_output", "beam_output_val.npz"),
    ]

    # Keep s009 separated as requested.
    s009_files = [
        os.path.join(DATA_ROOT, "s009", "beam_output", "beam_output.npz"),
    ]

    s008_labels = collect_sparse_labels_from_files(s008_files)
    s009_labels = collect_sparse_labels_from_files(s009_files)

    print(f"s008 total samples loaded: {s008_labels.shape[0]}")
    print(f"s008 unique beams observed: {np.unique(s008_labels).size}")
    print(f"s009 total samples loaded: {s009_labels.shape[0]}")
    print(f"s009 unique beams observed: {np.unique(s009_labels).size}")

    s008_csv_path = os.path.join(output_dir, "s008_beam_histogram.csv")
    s009_csv_path = os.path.join(output_dir, "s009_beam_histogram.csv")
    s008_image_path = os.path.join(output_dir, "s008_beam_histogram.png")
    s009_image_path = os.path.join(output_dir, "s009_beam_histogram.png")

    save_histogram_csv(s008_labels, s008_csv_path, num_beams=256)
    save_histogram_csv(s009_labels, s009_csv_path, num_beams=256)
    save_histogram_figure(
        s008_labels,
        s008_image_path,
        title="Raymobtime Beam Histogram - s008",
        num_beams=256,
    )
    save_histogram_figure(
        s009_labels,
        s009_image_path,
        title="Raymobtime Beam Histogram - s009",
        num_beams=256,
    )

    print(f"Saved outputs to: {output_dir}")
    print(f"  - {s008_csv_path}")
    print(f"  - {s009_csv_path}")
    print(f"  - {s008_image_path}")
    print(f"  - {s009_image_path}")
