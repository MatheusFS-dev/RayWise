"""Beam-metric helpers for shortlist-based throughput evaluation."""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def flatten_beam_grid(beam_scores: np.ndarray) -> np.ndarray:
    """Flatten RayWise beam tensors from ``(N, 8, 32)`` to ``(N, 256)``.

    Args:
        beam_scores (np.ndarray): Dense oracle beam tensor with shape
            ``(num_samples, num_rx_beams, num_tx_beams)``. The current RayWise
            setup uses ``(N, 8, 32)`` and the flattening follows NumPy's
            default row-major order so it matches the repository's sparse label
            encoding. The function expects a batch dimension; passing a single
            sample shaped ``(8, 32)`` raises an error instead of silently
            changing the interpretation.

    Returns:
        np.ndarray: Flattened score matrix with shape ``(num_samples, 256)``.

    Raises:
        ValueError: If ``beam_scores`` is not a 3-D tensor.

    Examples:
        >>> grid = np.zeros((2, 8, 32), dtype=np.float32)
        >>> flatten_beam_grid(grid).shape
        (2, 256)
    """

    if beam_scores.ndim != 3:
        raise ValueError(
            "beam_scores must have shape (N, R, T), but received "
            f"{beam_scores.shape}."
        )

    # The repository uses row-major flattening when converting oracle grids to
    # sparse labels. Reusing the same reshape keeps beam indexing consistent.
    return beam_scores.reshape(beam_scores.shape[0], -1)


def validate_score_matrix(
    score_matrix: np.ndarray,
    *,
    matrix_name: str,
    expected_num_samples: int | None = None,
    expected_num_classes: int | None = None,
    require_non_negative: bool = False,
) -> np.ndarray:
    """Validate a 2-D score matrix used for ranking or oracle gains.

    Args:
        score_matrix (np.ndarray): Candidate matrix to validate. It must have
            shape ``(num_samples, num_classes)``. If ``require_non_negative`` is
            ``True``, every value must be greater than or equal to zero because
            the caller intends to apply ``log2(1 + x)``. If
            ``require_non_negative`` is ``False``, the matrix may contain any
            finite real values, which is appropriate for logits used only for
            ranking.
        matrix_name (str): Human-readable label used in error messages so shape
            or numeric failures identify the offending input clearly.
        expected_num_samples (int | None): Expected number of rows. If ``None``,
            the function only enforces that the matrix is 2-D. If an integer is
            provided, the function raises when the row count differs.
        expected_num_classes (int | None): Expected number of columns. If
            ``None``, the function only enforces that the matrix is 2-D. If an
            integer is provided, the function raises when the column count
            differs.
        require_non_negative (bool): If ``True``, reject negative values so the
            throughput transform remains mathematically valid. If ``False``, the
            matrix is only required to be finite, which is appropriate for
            prediction logits where only ordering matters.

    Returns:
        np.ndarray: The validated matrix converted to ``float64`` for numerically
        stable downstream aggregation.

    Raises:
        ValueError: If the matrix is not 2-D, has the wrong shape, contains
            NaN/Inf, or violates the non-negativity requirement.

    Examples:
        >>> scores = validate_score_matrix(
        ...     np.array([[0.1, 0.2], [0.3, 0.4]]),
        ...     matrix_name="oracle",
        ...     expected_num_samples=2,
        ...     expected_num_classes=2,
        ...     require_non_negative=True,
        ... )
        >>> scores.dtype == np.float64
        True
    """

    validated = np.asarray(score_matrix, dtype=np.float64)
    if validated.ndim != 2:
        raise ValueError(
            f"{matrix_name} must have shape (N, C), but received {validated.shape}."
        )
    if expected_num_samples is not None and validated.shape[0] != expected_num_samples:
        raise ValueError(
            f"{matrix_name} row count mismatch. Expected {expected_num_samples}, "
            f"received {validated.shape[0]}."
        )
    if expected_num_classes is not None and validated.shape[1] != expected_num_classes:
        raise ValueError(
            f"{matrix_name} column count mismatch. Expected {expected_num_classes}, "
            f"received {validated.shape[1]}."
        )
    if not np.isfinite(validated).all():
        raise ValueError(f"{matrix_name} contains NaN or Inf values.")
    if require_non_negative and np.any(validated < 0.0):
        raise ValueError(f"{matrix_name} contains negative values.")
    return validated


def compute_sparse_labels_from_oracle(oracle_scores: np.ndarray) -> np.ndarray:
    """Return the oracle best-beam index for each sample.

    Args:
        oracle_scores (np.ndarray): Oracle beam score matrix with shape
            ``(num_samples, num_classes)``. Scores must be finite and
            non-negative because they represent communication gains. The
            returned labels follow the same flattened beam order used by the
            classifier outputs.

    Returns:
        np.ndarray: Sparse best-beam labels with shape ``(num_samples,)`` and
        dtype ``int64``.

    Raises:
        ValueError: If ``oracle_scores`` is not a valid 2-D non-negative score
            matrix.
    """

    validated = validate_score_matrix(
        oracle_scores,
        matrix_name="oracle_scores",
        require_non_negative=True,
    )
    return np.argmax(validated, axis=1).astype(np.int64)


def compute_topk_shortlist_indices(
    predicted_scores: np.ndarray,
    k_values: Sequence[int],
) -> dict[int, np.ndarray]:
    """Build per-sample shortlist indices for multiple ``k`` values.

    Args:
        predicted_scores (np.ndarray): Model score matrix with shape
            ``(num_samples, num_classes)``. The values may be logits or
            probabilities because only their ordering is used.
        k_values (Sequence[int]): Shortlist sizes to evaluate. Every ``k`` must
            satisfy ``1 <= k <= num_classes``. Repeated values are rejected so
            callers do not silently overwrite results.

    Returns:
        dict[int, np.ndarray]: Mapping from each ``k`` to an integer array with
        shape ``(num_samples, k)`` storing the top-``k`` class indices in
        descending score order.

    Raises:
        ValueError: If the score matrix is invalid, a ``k`` is out of range, or
            ``k_values`` contains duplicates.

    Examples:
        >>> scores = np.array([[0.1, 0.9, 0.3]])
        >>> shortlists = compute_topk_shortlist_indices(scores, [1, 2])
        >>> shortlists[1].tolist()
        [[1]]
    """

    validated = validate_score_matrix(
        predicted_scores,
        matrix_name="predicted_scores",
        require_non_negative=False,
    )
    unique_k_values = [int(k) for k in k_values]
    if len(unique_k_values) != len(set(unique_k_values)):
        raise ValueError(f"k_values must be unique, but received {list(k_values)}.")

    num_classes = validated.shape[1]
    for k in unique_k_values:
        if k < 1 or k > num_classes:
            raise ValueError(
                f"Invalid k={k}. Expected 1 <= k <= {num_classes}."
            )

    # Sorting once is cheaper than recomputing a full argsort for every k.
    sorted_indices = np.argsort(validated, axis=1)[:, ::-1]
    return {k: sorted_indices[:, :k].astype(np.int64, copy=False) for k in unique_k_values}


def compute_topk_accuracy(
    labels: np.ndarray,
    predicted_scores: np.ndarray,
    k_values: Sequence[int],
) -> dict[int, float]:
    """Compute top-k accuracy for a score matrix against sparse labels.

    Args:
        labels (np.ndarray): Sparse class labels with shape ``(num_samples,)``.
            Each label must be an integer in ``[0, num_classes - 1]``. Passing a
            one-hot array or a column vector raises explicitly so evaluation does
            not silently misinterpret the target format.
        predicted_scores (np.ndarray): Model score matrix with shape
            ``(num_samples, num_classes)``. Logits and probabilities are both
            valid because only ranking is used. If the matrix contains NaN or
            Inf, evaluation stops instead of masking the issue.
        k_values (Sequence[int]): Requested top-k values. Each ``k`` must be in
            ``[1, num_classes]``.

    Returns:
        dict[int, float]: Mapping from ``k`` to the fraction of samples whose
            label appears inside the predicted top-``k`` shortlist.

    Raises:
        ValueError: If labels or scores are malformed or if label IDs fall
            outside the score matrix range.

    Examples:
        >>> labels = np.array([1, 0])
        >>> scores = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> compute_topk_accuracy(labels, scores, [1])
        {1: 1.0}
    """

    validated_scores = validate_score_matrix(
        predicted_scores,
        matrix_name="predicted_scores",
        require_non_negative=False,
    )
    validated_labels = np.asarray(labels, dtype=np.int64)
    if validated_labels.ndim != 1:
        raise ValueError(
            f"labels must have shape (N,), but received {validated_labels.shape}."
        )
    if validated_labels.shape[0] != validated_scores.shape[0]:
        raise ValueError(
            "labels and predicted_scores must have the same number of samples. "
            f"Received {validated_labels.shape[0]} and {validated_scores.shape[0]}."
        )
    if validated_labels.size:
        label_min = int(validated_labels.min())
        label_max = int(validated_labels.max())
        num_classes = validated_scores.shape[1]
        if label_min < 0 or label_max >= num_classes:
            raise ValueError(
                "labels contain class IDs outside the prediction range. "
                f"Observed min={label_min}, max={label_max}, num_classes={num_classes}."
            )

    shortlists = compute_topk_shortlist_indices(validated_scores, k_values)
    total = max(int(validated_labels.shape[0]), 1)
    accuracies: dict[int, float] = {}
    for k, shortlist in shortlists.items():
        hits = (shortlist == validated_labels[:, None]).any(axis=1)
        accuracies[k] = float(hits.sum() / total)
    return accuracies


def compute_topk_throughput_ratio(
    oracle_scores: np.ndarray,
    predicted_scores: np.ndarray,
    k_values: Sequence[int],
) -> dict[int, float]:
    """Compute shortlist throughput ratio ``R_T(k)`` for multiple shortlist sizes.

    Args:
        oracle_scores (np.ndarray): Oracle beam gain matrix ``Psi`` with shape
            ``(num_samples, num_classes)``. Values must be finite and
            non-negative because the throughput transform uses ``log2(1 + Psi)``.
        predicted_scores (np.ndarray): Model score matrix with shape
            ``(num_samples, num_classes)``. If the values are logits, only their
            ordering is used. If the values are probabilities, the same ranking
            logic applies. The matrix may contain negative numbers because no
            logarithm is applied to it directly.
        k_values (Sequence[int]): Requested shortlist sizes. Each ``k`` must be
            in ``[1, num_classes]`` and the returned metric is guaranteed to be
            based on the same shortlist definition used by top-k accuracy.

    Returns:
        dict[int, float]: Mapping from each ``k`` to the throughput ratio
            ``R_T(k)`` computed as the sum of ``log2(1 + best oracle gain inside
            the predicted shortlist)`` divided by the sum of ``log2(1 + oracle
            best gain over the full codebook)``.

    Raises:
        ValueError: If the matrices are malformed, shapes differ, or the oracle
            denominator is zero.

    Examples:
        >>> oracle = np.array([[1.0, 4.0], [3.0, 2.0]])
        >>> pred = np.array([[0.9, 0.1], [0.8, 0.2]])
        >>> round(compute_topk_throughput_ratio(oracle, pred, [1])[1], 6)
        0.783812
    """

    validated_oracle = validate_score_matrix(
        oracle_scores,
        matrix_name="oracle_scores",
        require_non_negative=True,
    )
    validated_predicted = validate_score_matrix(
        predicted_scores,
        matrix_name="predicted_scores",
        expected_num_samples=validated_oracle.shape[0],
        expected_num_classes=validated_oracle.shape[1],
        require_non_negative=False,
    )

    oracle_best_scores = np.max(validated_oracle, axis=1)
    denominator = float(np.sum(np.log2(1.0 + oracle_best_scores)))
    if denominator <= 0.0:
        raise ValueError("The oracle throughput denominator must be strictly positive.")

    shortlists = compute_topk_shortlist_indices(validated_predicted, k_values)
    throughput_ratio: dict[int, float] = {}
    for k, shortlist in shortlists.items():
        shortlist_oracle_scores = np.take_along_axis(validated_oracle, shortlist, axis=1)
        selected_oracle_scores = np.max(shortlist_oracle_scores, axis=1)
        numerator = float(np.sum(np.log2(1.0 + selected_oracle_scores)))
        throughput_ratio[k] = numerator / denominator
    return throughput_ratio


def expand_restricted_class_scores(
    restricted_scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    full_num_classes: int,
    fill_value: float = -np.inf,
) -> np.ndarray:
    """Expand a restricted class score matrix back to the full codebook size.

    Args:
        restricted_scores (np.ndarray): Score matrix with shape
            ``(num_samples, restricted_num_classes)``. The columns correspond to
            the beam IDs listed in ``class_ids``. The values may be logits or
            probabilities. If a downstream consumer expects probabilities,
            ``fill_value`` should remain at ``-inf`` so missing classes never
            enter any shortlist even when modeled classes have small values.
        class_ids (np.ndarray): Integer beam IDs represented by
            ``restricted_scores``. The length must equal the number of restricted
            columns and every ID must be unique and within
            ``[0, full_num_classes - 1]``.
        full_num_classes (int): Size of the full beam codebook. In the current
            setup this is ``256``.
        fill_value (float): Score assigned to classes omitted from the restricted
            predictor. If left at ``-inf``, omitted classes can never be selected
            by a top-k shortlist. If set to another value, callers explicitly
            choose how unmodeled classes should rank relative to modeled ones.

    Returns:
        np.ndarray: Expanded matrix with shape ``(num_samples, full_num_classes)``.

    Raises:
        ValueError: If the score matrix or class IDs are inconsistent.

    Examples:
        >>> scores = np.array([[0.7, 0.3]])
        >>> expand_restricted_class_scores(scores, np.array([1, 3]), full_num_classes=5)
        array([[-inf,  0.7, -inf,  0.3, -inf]])
    """

    validated_scores = validate_score_matrix(
        restricted_scores,
        matrix_name="restricted_scores",
        require_non_negative=False,
    )
    validated_class_ids = np.asarray(class_ids, dtype=np.int64)
    if validated_class_ids.ndim != 1:
        raise ValueError(
            "class_ids must have shape (restricted_num_classes,), but received "
            f"{validated_class_ids.shape}."
        )
    if validated_scores.shape[1] != validated_class_ids.shape[0]:
        raise ValueError(
            "restricted_scores column count must match class_ids length. "
            f"Received {validated_scores.shape[1]} and {validated_class_ids.shape[0]}."
        )
    if full_num_classes <= 0:
        raise ValueError(f"full_num_classes must be positive, but received {full_num_classes}.")
    if np.unique(validated_class_ids).shape[0] != validated_class_ids.shape[0]:
        raise ValueError("class_ids must be unique.")
    if validated_class_ids.size:
        class_min = int(validated_class_ids.min())
        class_max = int(validated_class_ids.max())
        if class_min < 0 or class_max >= full_num_classes:
            raise ValueError(
                "class_ids fall outside the full codebook range. "
                f"Observed min={class_min}, max={class_max}, full_num_classes={full_num_classes}."
            )

    expanded = np.full(
        (validated_scores.shape[0], full_num_classes),
        fill_value=fill_value,
        dtype=np.float64,
    )
    expanded[:, validated_class_ids] = validated_scores
    return expanded


def ensure_non_decreasing(
    metric_by_k: Mapping[int, float],
    *,
    metric_name: str,
    tolerance: float = 1e-12,
) -> None:
    """Validate that a top-k metric is monotonic non-decreasing.

    Args:
        metric_by_k (Mapping[int, float]): Mapping from shortlist size to metric
            value. The check is performed in ascending ``k`` order.
        metric_name (str): Human-readable metric name included in error messages.
        tolerance (float): Allowed negative drift between consecutive ``k``
            values. Keeping this very small allows harmless floating-point noise
            while still catching shortlist-construction bugs.

    Returns:
        None: The function only validates the metric and raises on failure.

    Raises:
        ValueError: If the metric decreases by more than ``tolerance`` between
            consecutive shortlist sizes.
    """

    previous_value: float | None = None
    previous_k: int | None = None
    for current_k in sorted(metric_by_k):
        current_value = float(metric_by_k[current_k])
        if previous_value is not None and current_value + tolerance < previous_value:
            raise ValueError(
                f"{metric_name} must be non-decreasing with k, but it dropped from "
                f"{previous_value:.12f} at k={previous_k} to {current_value:.12f} "
                f"at k={current_k}."
            )
        previous_k = current_k
        previous_value = current_value


def ensure_parent_directory(output_path: Path) -> None:
    """Create the parent directory for an output file if needed.

    Args:
        output_path (Path): Target file path. If the parent directory already
            exists, the function leaves it untouched. If it does not exist, the
            full parent chain is created.

    Returns:
        None: The filesystem side effect is limited to directory creation.

    Raises:
        OSError: If the parent directory cannot be created.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
