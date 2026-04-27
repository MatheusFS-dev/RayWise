"""Oracle extraction and throughput-result serialization helpers."""

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils_beam_metrics import (
    compute_sparse_labels_from_oracle,
    compute_topk_accuracy,
    compute_topk_throughput_ratio,
    ensure_non_decreasing,
    ensure_parent_directory,
    flatten_beam_grid,
    validate_score_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S009_BEAM_ARCHIVE_PATH = REPO_ROOT / "src" / "data" / "s009" / "beam_output" / "beam_output.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "throughput" / "outputs"
DEFAULT_ORACLE_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "s009_oracle_beam_scores.npz"
DEFAULT_K_VALUES = (1, 3, 5, 10, 20, 30, 40, 50)


def load_beam_archive(
    beam_archive_path: Path,
    *,
    archive_key: str = "output_classification",
    imag_tolerance: float = 1e-12,
) -> np.ndarray:
    """Load a dense beam tensor from a RayWise ``.npz`` archive.

    Args:
        beam_archive_path (Path): Filesystem path to the ``.npz`` archive.
        archive_key (str): Key inside the archive that stores the beam tensor.
            The default matches the repository's baseline archives. If a
            different key is provided, the function reads only that entry and
            raises if it is absent instead of searching alternative keys.
        imag_tolerance (float): Maximum allowed absolute imaginary component.
            If the stored tensor is complex-valued but its imaginary part stays
            below this tolerance, only the real component is kept. If the
            imaginary component exceeds the tolerance, the function raises so the
            caller does not silently change the communication metric.

    Returns:
        np.ndarray: Real-valued beam tensor with shape ``(num_samples, 8, 32)``
        and dtype ``float32``.

    Raises:
        FileNotFoundError: If ``beam_archive_path`` does not exist.
        KeyError: If ``archive_key`` is missing from the archive.
        ValueError: If the archive contains a meaningful imaginary component.

    Examples:
        >>> path = DEFAULT_S009_BEAM_ARCHIVE_PATH
        >>> load_beam_archive(path).shape
        (9638, 8, 32)
    """

    if not beam_archive_path.is_file():
        raise FileNotFoundError(f"Beam archive not found: {beam_archive_path}.")

    with np.load(beam_archive_path, allow_pickle=False) as archive:
        if archive_key not in archive:
            raise KeyError(
                f"Archive key '{archive_key}' not found in {beam_archive_path}. "
                f"Available keys: {list(archive.files)}."
            )
        beam_tensor = np.asarray(archive[archive_key])

    # The bundled baseline archives are stored as complex arrays whose
    # imaginary part is numerically zero. Enforcing that assumption prevents the
    # throughput metric from silently switching to a different signal definition.
    if np.iscomplexobj(beam_tensor):
        max_imag = float(np.max(np.abs(beam_tensor.imag)))
        if max_imag > imag_tolerance:
            raise ValueError(
                f"Beam archive {beam_archive_path} has non-negligible imaginary values: "
                f"max imaginary magnitude {max_imag:.6e}."
            )
        beam_tensor = beam_tensor.real

    return np.asarray(beam_tensor, dtype=np.float32)


def build_s009_oracle_beam_scores_archive(
    *,
    beam_archive_path: Path = DEFAULT_S009_BEAM_ARCHIVE_PATH,
    output_path: Path = DEFAULT_ORACLE_OUTPUT_PATH,
) -> dict[str, np.ndarray]:
    """Build and save the flattened S009 oracle beam score archive.

    Args:
        beam_archive_path (Path): Path to the repo-local S009 beam archive. The
            current implementation assumes the baseline archive under
            ``src/data/s009`` because it is available inside the repository and
            already aligned with the 256-class label order.
        output_path (Path): Destination ``.npz`` file. The function always
            writes a fresh archive so downstream evaluations use an explicit and
            reproducible oracle snapshot.

    Returns:
        dict[str, np.ndarray]: In-memory payload containing ``psi_all``,
        ``sample_ids``, ``best_idx``, ``best_gain``, and ``beam_shape``.

    Raises:
        FileNotFoundError: If the source archive is missing.
        ValueError: If the source tensor is malformed or inconsistent.

    Examples:
        >>> payload = build_s009_oracle_beam_scores_archive(output_path=DEFAULT_ORACLE_OUTPUT_PATH)
        >>> payload["psi_all"].shape
        (9638, 256)
    """

    beam_tensor = load_beam_archive(beam_archive_path)
    psi_all = validate_score_matrix(
        flatten_beam_grid(beam_tensor),
        matrix_name="psi_all",
        require_non_negative=True,
    ).astype(np.float32)
    best_idx = compute_sparse_labels_from_oracle(psi_all)
    best_gain = np.max(psi_all, axis=1).astype(np.float32)
    sample_ids = np.arange(psi_all.shape[0], dtype=np.int64)
    beam_shape = np.asarray(beam_tensor.shape[1:], dtype=np.int64)

    # Re-derive the sparse labels from the flattened tensor and store them
    # alongside the oracle scores so later evaluations can validate alignment.
    if not np.array_equal(best_idx, np.argmax(psi_all, axis=1).astype(np.int64)):
        raise ValueError("Oracle label validation failed after flattening beam scores.")

    ensure_parent_directory(output_path)
    np.savez_compressed(
        output_path,
        psi_all=psi_all,
        sample_ids=sample_ids,
        best_idx=best_idx,
        best_gain=best_gain,
        beam_shape=beam_shape,
    )
    return {
        "psi_all": psi_all,
        "sample_ids": sample_ids,
        "best_idx": best_idx,
        "best_gain": best_gain,
        "beam_shape": beam_shape,
    }


def load_oracle_beam_scores_archive(
    oracle_archive_path: Path = DEFAULT_ORACLE_OUTPUT_PATH,
) -> dict[str, np.ndarray]:
    """Load a previously generated oracle beam score archive.

    Args:
        oracle_archive_path (Path): Path to the ``.npz`` file created by
            :func:`build_s009_oracle_beam_scores_archive`.

    Returns:
        dict[str, np.ndarray]: Dictionary containing the stored arrays.

    Raises:
        FileNotFoundError: If ``oracle_archive_path`` does not exist.
        KeyError: If required fields are missing.
        ValueError: If the stored arrays are inconsistent.
    """

    if not oracle_archive_path.is_file():
        raise FileNotFoundError(
            f"Oracle archive not found: {oracle_archive_path}. "
            "Run build_s009_oracle_beam_scores_archive first."
        )

    required_keys = {"psi_all", "sample_ids", "best_idx", "best_gain", "beam_shape"}
    with np.load(oracle_archive_path, allow_pickle=False) as archive:
        missing_keys = required_keys.difference(archive.files)
        if missing_keys:
            raise KeyError(
                f"Oracle archive {oracle_archive_path} is missing keys: {sorted(missing_keys)}."
            )
        payload = {key: np.asarray(archive[key]) for key in required_keys}

    psi_all = validate_score_matrix(
        payload["psi_all"],
        matrix_name="psi_all",
        require_non_negative=True,
    )
    if payload["sample_ids"].ndim != 1:
        raise ValueError("sample_ids must have shape (N,).")
    if payload["best_idx"].ndim != 1:
        raise ValueError("best_idx must have shape (N,).")
    if payload["best_gain"].ndim != 1:
        raise ValueError("best_gain must have shape (N,).")
    if payload["sample_ids"].shape[0] != psi_all.shape[0]:
        raise ValueError("sample_ids length does not match psi_all row count.")
    if payload["best_idx"].shape[0] != psi_all.shape[0]:
        raise ValueError("best_idx length does not match psi_all row count.")
    if payload["best_gain"].shape[0] != psi_all.shape[0]:
        raise ValueError("best_gain length does not match psi_all row count.")

    recalculated_best_idx = compute_sparse_labels_from_oracle(psi_all)
    if not np.array_equal(payload["best_idx"].astype(np.int64), recalculated_best_idx):
        raise ValueError("Stored best_idx does not match psi_all argmax labels.")
    recalculated_best_gain = np.max(psi_all, axis=1)
    if not np.allclose(payload["best_gain"], recalculated_best_gain):
        raise ValueError("Stored best_gain does not match psi_all maxima.")

    payload["psi_all"] = psi_all.astype(np.float32)
    payload["sample_ids"] = payload["sample_ids"].astype(np.int64)
    payload["best_idx"] = payload["best_idx"].astype(np.int64)
    payload["best_gain"] = payload["best_gain"].astype(np.float32)
    payload["beam_shape"] = payload["beam_shape"].astype(np.int64)
    return payload


def compute_model_throughput_result(
    *,
    model_name: str,
    oracle_scores: np.ndarray,
    predicted_scores: np.ndarray,
    k_values: Sequence[int],
    sample_ids: np.ndarray,
    predicted_sample_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute top-k accuracy and throughput ratio for one saved score matrix.

    Args:
        model_name (str): Stable identifier for the evaluated artifact. The
            value is written verbatim to the JSON and CSV outputs.
        oracle_scores (np.ndarray): Oracle score matrix with shape
            ``(num_samples, num_classes)`` and non-negative values.
        predicted_scores (np.ndarray): Model score matrix with the same shape as
            ``oracle_scores``. Values may be logits, probabilities, or expanded
            restricted scores, as long as ranking over the 256-beam codebook is
            preserved.
        k_values (Sequence[int]): Requested shortlist sizes.
        sample_ids (np.ndarray): Oracle sample IDs with shape ``(num_samples,)``.
            These IDs are compared directly against ``predicted_sample_ids`` so
            the function fails fast when score rows and oracle rows are not
            aligned.
        predicted_sample_ids (np.ndarray): Sample IDs attached to the prediction
            matrix. They must match ``sample_ids`` exactly. No reordering is
            attempted because silent realignment would hide data bugs.

    Returns:
        dict[str, Any]: Compact JSON-serializable payload containing
        ``model_name``, ``k_values``, ``throughput_ratio``, ``topk_accuracy``,
        and ``num_samples``.

    Raises:
        ValueError: If sample IDs, shapes, or monotonicity checks fail.

    Examples:
        >>> oracle = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=np.float32)
        >>> pred = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        >>> result = compute_model_throughput_result(
        ...     model_name="demo",
        ...     oracle_scores=oracle,
        ...     predicted_scores=pred,
        ...     k_values=[1, 2],
        ...     sample_ids=np.array([0, 1]),
        ...     predicted_sample_ids=np.array([0, 1]),
        ... )
        >>> result["model_name"]
        'demo'
    """

    validated_sample_ids = np.asarray(sample_ids, dtype=np.int64)
    validated_predicted_sample_ids = np.asarray(predicted_sample_ids, dtype=np.int64)
    if validated_sample_ids.ndim != 1 or validated_predicted_sample_ids.ndim != 1:
        raise ValueError("sample_ids and predicted_sample_ids must both have shape (N,).")
    if not np.array_equal(validated_sample_ids, validated_predicted_sample_ids):
        raise ValueError(
            f"Sample IDs for {model_name} do not match the oracle archive. "
            "Prediction rows must already be aligned."
        )

    validated_oracle_scores = validate_score_matrix(
        oracle_scores,
        matrix_name="oracle_scores",
        require_non_negative=True,
    )
    validated_predicted_scores = validate_score_matrix(
        predicted_scores,
        matrix_name="predicted_scores",
        expected_num_samples=validated_oracle_scores.shape[0],
        expected_num_classes=validated_oracle_scores.shape[1],
        require_non_negative=False,
    )

    labels = compute_sparse_labels_from_oracle(validated_oracle_scores)
    throughput_ratio_by_k = compute_topk_throughput_ratio(
        validated_oracle_scores,
        validated_predicted_scores,
        k_values,
    )
    topk_accuracy_by_k = compute_topk_accuracy(
        labels,
        validated_predicted_scores,
        k_values,
    )

    # These metrics should improve or stay equal as the shortlist grows.
    ensure_non_decreasing(throughput_ratio_by_k, metric_name=f"{model_name} throughput_ratio")
    ensure_non_decreasing(topk_accuracy_by_k, metric_name=f"{model_name} topk_accuracy")

    ordered_k_values = [int(k) for k in sorted(int(k) for k in k_values)]
    return {
        "model_name": model_name,
        "k_values": ordered_k_values,
        "throughput_ratio": [float(throughput_ratio_by_k[k]) for k in ordered_k_values],
        "topk_accuracy": [float(topk_accuracy_by_k[k]) for k in ordered_k_values],
        "num_samples": int(validated_oracle_scores.shape[0]),
    }


def save_score_archive(
    *,
    output_path: Path,
    scores: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    """Persist one model's score matrix for later inspection.

    Args:
        output_path (Path): Destination ``.npz`` file.
        scores (np.ndarray): Score matrix with shape ``(num_samples, num_classes)``.
            Values may be logits, probabilities, or restricted-score expansions.
        sample_ids (np.ndarray): Sample IDs aligned with the rows of ``scores``.

    Returns:
        None: The function writes the archive to ``output_path``.

    Raises:
        ValueError: If shapes are inconsistent.
    """

    validated_scores = validate_score_matrix(
        scores,
        matrix_name="scores",
        require_non_negative=False,
    ).astype(np.float32)
    validated_sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if validated_sample_ids.ndim != 1:
        raise ValueError("sample_ids must have shape (N,).")
    if validated_sample_ids.shape[0] != validated_scores.shape[0]:
        raise ValueError("sample_ids length does not match score rows.")

    ensure_parent_directory(output_path)
    np.savez_compressed(output_path, scores=validated_scores, sample_ids=validated_sample_ids)


def save_throughput_result(
    *,
    output_path: Path,
    result: dict[str, Any],
) -> None:
    """Write one model throughput result to JSON.

    Args:
        output_path (Path): Destination JSON path.
        result (dict[str, Any]): JSON-serializable throughput result produced by
            :func:`compute_model_throughput_result`.

    Returns:
        None: The function writes the JSON payload with stable formatting.

    Raises:
        TypeError: If ``result`` is not JSON-serializable.
        OSError: If the file cannot be written.
    """

    ensure_parent_directory(output_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def write_throughput_summary_csv(
    *,
    output_path: Path,
    results: Sequence[dict[str, Any]],
) -> None:
    """Write a compact CSV summary across evaluated models and shortlist sizes.

    Args:
        output_path (Path): Destination CSV path.
        results (Sequence[dict[str, Any]]): Sequence of throughput result
            dictionaries. Each result must contain the keys produced by
            :func:`compute_model_throughput_result`.

    Returns:
        None: The function writes one row per ``(model_name, k)`` pair.

    Raises:
        KeyError: If a required field is missing from one of the result payloads.
        ValueError: If the list lengths inside a result are inconsistent.
    """

    ensure_parent_directory(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model_name",
                "k",
                "topk_accuracy",
                "throughput_ratio",
                "num_samples",
            ],
        )
        writer.writeheader()
        for result in results:
            model_name = str(result["model_name"])
            k_values = list(result["k_values"])
            topk_accuracy = list(result["topk_accuracy"])
            throughput_ratio = list(result["throughput_ratio"])
            if not (len(k_values) == len(topk_accuracy) == len(throughput_ratio)):
                raise ValueError(
                    f"Result lengths for {model_name} do not match: "
                    f"k_values={len(k_values)}, topk_accuracy={len(topk_accuracy)}, "
                    f"throughput_ratio={len(throughput_ratio)}."
                )
            for k, accuracy, ratio in zip(k_values, topk_accuracy, throughput_ratio):
                writer.writerow(
                    {
                        "model_name": model_name,
                        "k": int(k),
                        "topk_accuracy": float(accuracy),
                        "throughput_ratio": float(ratio),
                        "num_samples": int(result["num_samples"]),
                    }
                )
