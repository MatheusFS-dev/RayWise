"""Generate paired-bootstrap accuracy comparisons for the proposed RayWise models.

This module scores the saved joint 1D CNN and the saved MOTPE-distilled student
on the repo-local S009 test split, then estimates paired-bootstrap confidence
intervals for their top-k accuracy differences. The scope is intentionally
narrow because the current revision only needs statistical qualification for the
comparison between models that can already be executed locally.
"""

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
THROUGHPUT_ROOT = SRC_ROOT / "evaluation" / "throughput"
for import_root in (SRC_ROOT, THROUGHPUT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.evaluation.throughput.evaluate_models_throughput import (
    DISTILLED_TOP1_MODEL_PATH,
    JOINT_TOP1_MODEL_PATH,
    MODEL_BATCH_SIZE,
    STANDARD_MODEL_DATA_SEED,
    fit_standard_scaler_from_train_split,
    load_baseline_raywise_arrays,
    predict_scores_with_saved_model,
    transform_with_standard_scaler,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "paired_bootstrap_proposed_vs_distilled"
DEFAULT_K_VALUES = (1, 10)
DEFAULT_NUM_BOOTSTRAP_SAMPLES = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_SEED = 20260528
DEFAULT_BOOTSTRAP_BATCH_SIZE = 256


def compute_topk_correct_mask(labels: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Return one boolean correctness flag per sample for a given top-k cutoff.

    Args:
        labels (np.ndarray): Sparse ground-truth class indices with shape
            ``(num_samples,)``. Each value must identify one of the score
            columns exactly.
        scores (np.ndarray): Model score matrix with shape
            ``(num_samples, num_classes)``. Higher scores indicate better
            predicted classes. The helper only uses ranking, so the values may
            be logits or probabilities.
        k (int): Shortlist size used to define correctness. If ``k`` is ``1``,
            the mask marks only exact argmax hits. If ``k`` is larger, a sample
            is marked correct whenever the ground-truth class appears anywhere
            in the top-k shortlist. Larger ``k`` values therefore produce
            greater or equal accuracy, but they weaken the interpretation for
            Reviewer 3 comment 4 because that comment is specifically about
            top-1.

    Returns:
        np.ndarray: Boolean array with shape ``(num_samples,)`` where ``True``
        marks samples whose label is present in the model's top-k shortlist.

    Raises:
        ValueError: If ``labels`` and ``scores`` do not share the same number of
            samples, if either array has the wrong rank, or if ``k`` falls
            outside ``[1, num_classes]``.

    Examples:
        >>> labels = np.asarray([0, 2], dtype=np.int64)
        >>> scores = np.asarray([[0.9, 0.1, 0.0], [0.6, 0.2, 0.7]], dtype=np.float32)
        >>> compute_topk_correct_mask(labels, scores, 1).tolist()
        [True, True]
    """

    validated_labels = np.asarray(labels, dtype=np.int64)
    validated_scores = np.asarray(scores, dtype=np.float32)
    if validated_labels.ndim != 1:
        raise ValueError(
            f"labels must have shape (N,), but received {validated_labels.shape}."
        )
    if validated_scores.ndim != 2:
        raise ValueError(
            f"scores must have shape (N, C), but received {validated_scores.shape}."
        )
    if validated_scores.shape[0] != validated_labels.shape[0]:
        raise ValueError(
            "labels and scores must share the same number of samples. "
            f"Received labels={validated_labels.shape[0]} and scores={validated_scores.shape[0]}."
        )
    if not 1 <= k <= validated_scores.shape[1]:
        raise ValueError(
            f"k must be in [1, {validated_scores.shape[1]}], but received {k}."
        )

    # Only the rank ordering matters here. ``argpartition`` avoids the full
    # sort cost while still guaranteeing that the returned block contains the
    # top-k candidates for each sample.
    topk_indices = np.argpartition(validated_scores, -k, axis=1)[:, -k:]
    return (topk_indices == validated_labels[:, None]).any(axis=1)


def compute_paired_bootstrap_interval(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    num_bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
    bootstrap_batch_size: int = DEFAULT_BOOTSTRAP_BATCH_SIZE,
) -> dict[str, float]:
    """Estimate a paired-bootstrap confidence interval for an accuracy gap.

    Args:
        reference_correct (np.ndarray): Boolean correctness mask for the
            reference model with shape ``(num_samples,)``. In this module the
            reference branch is the saved joint 1D CNN, so positive differences
            mean the candidate outperforms the standalone model.
        candidate_correct (np.ndarray): Boolean correctness mask for the
            candidate model with the same shape as ``reference_correct``. Here
            the candidate branch is the saved MOTPE-distilled student. If the
            candidate is better on a sample, that sample contributes positively
            to the paired difference.
        num_bootstrap_samples (int): Number of paired-bootstrap resamples. More
            samples stabilize the percentile interval but increase runtime.
            Smaller values are faster but produce noisier interval endpoints.
        confidence_level (float): Central coverage level in ``(0, 1)``. For
            example, ``0.95`` returns the 2.5th and 97.5th percentiles, whereas
            ``0.90`` returns a narrower 5th to 95th percentile interval.
        random_seed (int): Seed for the NumPy bootstrap generator. Different
            seeds change the exact interval endpoints slightly but should not
            change the qualitative interpretation when the number of resamples is
            large.
        bootstrap_batch_size (int): Number of resamples processed per batch.
            Larger batches reduce Python overhead but use more memory because
            each batch materializes a ``(batch_size, num_samples)`` index
            matrix. Smaller batches reduce memory pressure at the cost of more
            loop iterations.

    Returns:
        dict[str, float]: Summary dictionary containing the reference accuracy,
        candidate accuracy, point estimate of ``candidate - reference``, and the
        lower and upper confidence interval bounds.

    Raises:
        ValueError: If the correctness vectors have different shapes, if either
            vector is not one-dimensional, or if any scalar configuration value
            is outside its valid range.

    Examples:
        >>> ref = np.asarray([True, False, True, False], dtype=bool)
        >>> cand = np.asarray([True, True, True, False], dtype=bool)
        >>> out = compute_paired_bootstrap_interval(ref, cand, 256, 0.95, 7)
        >>> out["accuracy_difference"] > 0.0
        True
    """

    validated_reference = np.asarray(reference_correct, dtype=bool)
    validated_candidate = np.asarray(candidate_correct, dtype=bool)
    if validated_reference.ndim != 1 or validated_candidate.ndim != 1:
        raise ValueError(
            "reference_correct and candidate_correct must both have shape (N,)."
        )
    if validated_reference.shape != validated_candidate.shape:
        raise ValueError(
            "reference_correct and candidate_correct must have the same shape. "
            f"Received {validated_reference.shape} and {validated_candidate.shape}."
        )
    if num_bootstrap_samples <= 0:
        raise ValueError(
            f"num_bootstrap_samples must be positive, but received {num_bootstrap_samples}."
        )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), but received {confidence_level}."
        )
    if bootstrap_batch_size <= 0:
        raise ValueError(
            f"bootstrap_batch_size must be positive, but received {bootstrap_batch_size}."
        )

    num_samples = validated_reference.shape[0]
    rng = np.random.default_rng(random_seed)

    reference_float = validated_reference.astype(np.float32, copy=False)
    candidate_float = validated_candidate.astype(np.float32, copy=False)

    # The paired setup resamples shared sample indices for both models, which
    # preserves sample-level correlation and directly targets the accuracy gap
    # on the fixed S009 test set.
    bootstrap_differences = np.empty(num_bootstrap_samples, dtype=np.float32)
    written = 0
    while written < num_bootstrap_samples:
        current_batch_size = min(bootstrap_batch_size, num_bootstrap_samples - written)
        sampled_indices = rng.integers(
            low=0,
            high=num_samples,
            size=(current_batch_size, num_samples),
            dtype=np.int64,
        )
        reference_batch = reference_float[sampled_indices].mean(axis=1, dtype=np.float32)
        candidate_batch = candidate_float[sampled_indices].mean(axis=1, dtype=np.float32)
        bootstrap_differences[written : written + current_batch_size] = candidate_batch - reference_batch
        written += current_batch_size

    alpha = 1.0 - confidence_level
    confidence_interval_low, confidence_interval_high = np.quantile(
        bootstrap_differences,
        [alpha / 2.0, 1.0 - (alpha / 2.0)],
    ).astype(np.float64)

    reference_accuracy = float(reference_float.mean(dtype=np.float64))
    candidate_accuracy = float(candidate_float.mean(dtype=np.float64))
    return {
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_difference": candidate_accuracy - reference_accuracy,
        "confidence_interval_low": float(confidence_interval_low),
        "confidence_interval_high": float(confidence_interval_high),
    }


def score_required_models(
    *,
    batch_size: int = MODEL_BATCH_SIZE,
) -> dict[str, Any]:
    """Score the locally runnable proposed models on S009 with shared inputs.

    Args:
        batch_size (int): Prediction batch size used for both models. Larger
            values reduce interpreter overhead but increase GPU or host-memory
            pressure. Smaller values are slower but safer when the active
            environment has tighter memory limits.

    Returns:
        dict[str, Any]: Payload containing S009 labels plus one score matrix per
        required model under the keys ``joint_scores`` and ``distilled_scores``.

    Raises:
        FileNotFoundError: If either saved checkpoint is missing.
        ImportError: If TensorFlow is unavailable in the active interpreter.
        ValueError: If baseline arrays or model outputs have unexpected shapes.
    """

    baseline_arrays = load_baseline_raywise_arrays()

    # The saved proposed models were trained with the standard S008 train-split
    # scaler. Rebuilding that scaler here keeps the comparison aligned with the
    # reported paper metrics instead of fitting statistics on S009.
    scaler_mean, scaler_scale = fit_standard_scaler_from_train_split(
        coordinates=baseline_arrays["s008_coord_all"],
        data_seed=STANDARD_MODEL_DATA_SEED,
    )
    s009_coordinates = transform_with_standard_scaler(
        baseline_arrays["s009_coord"],
        mean=scaler_mean,
        scale=scaler_scale,
    )

    joint_scores = predict_scores_with_saved_model(
        model_path=JOINT_TOP1_MODEL_PATH,
        lidar=baseline_arrays["s009_lidar"],
        coordinates=s009_coordinates,
        batch_size=batch_size,
    )
    distilled_scores = predict_scores_with_saved_model(
        model_path=DISTILLED_TOP1_MODEL_PATH,
        lidar=baseline_arrays["s009_lidar"],
        coordinates=s009_coordinates,
        batch_size=batch_size,
    )

    return {
        "labels": baseline_arrays["s009_labels"],
        "joint_scores": joint_scores,
        "distilled_scores": distilled_scores,
    }


def build_comparison_summary(
    *,
    labels: np.ndarray,
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    k_values: Sequence[int],
    num_bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
    bootstrap_batch_size: int,
) -> dict[str, Any]:
    """Build paired-bootstrap summaries for the requested top-k operating points.

    Args:
        labels (np.ndarray): Sparse S009 labels with shape ``(num_samples,)``.
        reference_scores (np.ndarray): Score matrix for the reference model.
            This branch defines the baseline accuracy used in the difference
            calculation.
        candidate_scores (np.ndarray): Score matrix for the candidate model. The
            reported difference is always ``candidate - reference``.
        k_values (Sequence[int]): Top-k cutoffs to summarize. Smaller values
            emphasize exact ranking quality, while larger values emphasize
            shortlist coverage.
        num_bootstrap_samples (int): Number of bootstrap resamples passed
            through to :func:`compute_paired_bootstrap_interval`.
        confidence_level (float): Confidence level passed through to
            :func:`compute_paired_bootstrap_interval`.
        random_seed (int): Base random seed for the bootstrap. Each ``k`` uses
            ``random_seed + k`` so that repeated calls remain reproducible while
            still producing independent resample streams across different top-k
            summaries.
        bootstrap_batch_size (int): Batch size passed through to
            :func:`compute_paired_bootstrap_interval`.

    Returns:
        dict[str, Any]: Nested summary keyed by ``top_1``, ``top_10``, and any
        additional requested cutoffs.

    Raises:
        ValueError: If ``k_values`` is empty.
    """

    if not k_values:
        raise ValueError("k_values must contain at least one cutoff.")

    summary: dict[str, Any] = {}
    for k in k_values:
        reference_correct = compute_topk_correct_mask(labels=labels, scores=reference_scores, k=int(k))
        candidate_correct = compute_topk_correct_mask(labels=labels, scores=candidate_scores, k=int(k))
        summary[f"top_{int(k)}"] = compute_paired_bootstrap_interval(
            reference_correct=reference_correct,
            candidate_correct=candidate_correct,
            num_bootstrap_samples=num_bootstrap_samples,
            confidence_level=confidence_level,
            random_seed=random_seed + int(k),
            bootstrap_batch_size=bootstrap_batch_size,
        )
    return summary


def format_percentage(value: float) -> str:
    """Format a unit-interval metric as a percentage with two decimals.

    Args:
        value (float): Value in unit scale, typically an accuracy or confidence
            interval endpoint. The function does not clip the value because out-
            of-range numbers should remain visible during debugging.

    Returns:
        str: Percentage string with two decimal places.

    Raises:
        None: Formatting does not validate the numeric range.
    """

    return f"{100.0 * value:.2f}%"


def write_summary_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    num_bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
) -> None:
    """Write the paired-bootstrap summary to JSON and Markdown artifacts.

    Args:
        output_dir (Path): Destination directory for the generated files.
        summary (dict[str, Any]): Nested summary returned by
            :func:`build_comparison_summary`.
        num_bootstrap_samples (int): Number of resamples used. This is recorded
            in the report so the paper text can describe the procedure exactly.
        confidence_level (float): Confidence level recorded beside the interval.
        random_seed (int): Bootstrap seed recorded for reproducibility.

    Returns:
        None: Creates or overwrites ``summary.json`` and ``report.md``.

    Raises:
        OSError: If the output directory or files cannot be written.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "reference_model": str(JOINT_TOP1_MODEL_PATH),
        "candidate_model": str(DISTILLED_TOP1_MODEL_PATH),
        "num_bootstrap_samples": int(num_bootstrap_samples),
        "confidence_level": float(confidence_level),
        "random_seed": int(random_seed),
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Paired Bootstrap Report: Proposed vs Proposed-Distilled",
        "",
        "## Setup",
        "",
        f"- Reference model: `{JOINT_TOP1_MODEL_PATH}`",
        f"- Candidate model: `{DISTILLED_TOP1_MODEL_PATH}`",
        f"- Bootstrap resamples: {num_bootstrap_samples}",
        f"- Confidence level: {confidence_level:.2f}",
        f"- Random seed: {random_seed}",
        "",
        "## Results",
        "",
        "| Metric | Reference | Candidate | Difference (candidate - reference) | Confidence interval |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric_name, metric_summary in summary.items():
        report_lines.append(
            "| "
            f"{metric_name.replace('_', '-').title()} | "
            f"{format_percentage(metric_summary['reference_accuracy'])} | "
            f"{format_percentage(metric_summary['candidate_accuracy'])} | "
            f"{format_percentage(metric_summary['accuracy_difference'])} | "
            f"[{format_percentage(metric_summary['confidence_interval_low'])}, "
            f"{format_percentage(metric_summary['confidence_interval_high'])}] |"
        )

    # The interpretation line is kept narrow so the resulting text can be
    # quoted or adapted directly in the revision response without overclaiming.
    top1_summary = summary["top_1"]
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The key manuscript comparison is the top-1 row above.",
            (
                "For top-1, the MOTPE-distilled student changes accuracy by "
                f"{format_percentage(top1_summary['accuracy_difference'])} relative to the joint 1D CNN, "
                f"with a {confidence_level:.0%} paired-bootstrap interval of "
                f"[{format_percentage(top1_summary['confidence_interval_low'])}, "
                f"{format_percentage(top1_summary['confidence_interval_high'])}]."
            ),
        ]
    )

    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def generate_required_bootstrap_results(
    *,
    output_dir: Path = OUTPUT_DIR,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    batch_size: int = MODEL_BATCH_SIZE,
    num_bootstrap_samples: int = DEFAULT_NUM_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_batch_size: int = DEFAULT_BOOTSTRAP_BATCH_SIZE,
) -> dict[str, Any]:
    """Score the required models and write the paired-bootstrap report.

    Args:
        output_dir (Path): Directory where the JSON and Markdown artifacts are
            written. Changing this path only affects the report location; it
            does not change the evaluated models or dataset split.
        k_values (Sequence[int]): Top-k cutoffs to summarize. Keeping the
            default ``(1, 10)`` matches the paper's core operating points.
            Adding more values broadens the report but increases runtime
            slightly because each cutoff triggers its own paired bootstrap.
        batch_size (int): Prediction batch size for both saved models. Larger
            values can be faster on the GPU but use more memory.
        num_bootstrap_samples (int): Number of paired-bootstrap resamples.
            Larger values yield smoother interval estimates but increase runtime.
        confidence_level (float): Central interval coverage in ``(0, 1)``. A
            larger value widens the interval, whereas a smaller value narrows it.
        random_seed (int): Bootstrap seed used for reproducibility. Changing the
            seed changes the exact Monte Carlo estimate but should not change the
            qualitative interpretation when enough resamples are used.
        bootstrap_batch_size (int): Number of bootstrap resamples processed per
            batch. Larger batches trade higher memory usage for lower loop
            overhead.

    Returns:
        dict[str, Any]: The nested comparison summary that was also written to
        disk. The caller can use this return value directly in notebooks or
        ad-hoc manuscript update code without rereading the output files.

    Raises:
        FileNotFoundError: If the saved checkpoints or required dataset archives
            are missing.
        ImportError: If TensorFlow is unavailable.
        ValueError: If any scoring or bootstrap argument is invalid.
        OSError: If the report artifacts cannot be written.

    Examples:
        >>> summary = generate_required_bootstrap_results(num_bootstrap_samples=512)
        >>> "top_1" in summary
        True
    """

    scored_models = score_required_models(batch_size=batch_size)
    summary = build_comparison_summary(
        labels=scored_models["labels"],
        reference_scores=scored_models["joint_scores"],
        candidate_scores=scored_models["distilled_scores"],
        k_values=k_values,
        num_bootstrap_samples=num_bootstrap_samples,
        confidence_level=confidence_level,
        random_seed=random_seed,
        bootstrap_batch_size=bootstrap_batch_size,
    )
    write_summary_artifacts(
        output_dir=output_dir,
        summary=summary,
        num_bootstrap_samples=num_bootstrap_samples,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    return summary
