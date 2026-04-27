"""Measure stats for the best ensemble and knowledge-distillation artifacts."""

import os
import re
import sys
import json
import pickle
import time

print("\n\033[91mRun this script with sudo if using CPU to measure model stats!!\033[0m")
print(f"\033[94m{sys.executable} is running this script\033[0m")
print(
    "\033[93mWARNING: If using Anaconda Python environment, ensure you are using the correct Python interpreter.\033[0m\n"
)

from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional, Sequence

import pandas as pd
import tensorflow as tf
import numpy as np
from autogluon.tabular import TabularPredictor

layers = tf.keras.layers

from araras.ml.model.stats import get_model_stats, render_model_stats_report


DEFAULT_OUTPUT_DIR = Path("./runs/stats")
BEST_ENSEMBLE_RESULTS_PATH = Path(
    "/home/matheus/src/RayWise/src/runs/ensemble_cnn1d_stacking_top10/results.json"
)
BEST_KD_STUDY_DIR = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_ensemble_cnn1d_stacking_kd_top10"
)


@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(layers.Layer):
    """Convert raw LiDAR occupancy tensors to a 4-channel semantic tensor.

    Returns:
        tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)`` whose
        channels encode base station, vehicle, obstacle, and free space.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Map the repository's LiDAR encoding into one-hot semantic channels.

        Args:
            inputs (tf.Tensor): Dense LiDAR tensor with shape
                ``(batch, 20, 200, 10)`` and values in ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Semantic one-hot tensor with shape ``(batch, 20, 200, 4)``.

        Raises:
            tf.errors.InvalidArgumentError: If TensorFlow encounters an invalid
                shape or dtype during graph execution.
        """
        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)

    def get_config(self) -> dict[str, Any]:
        """Return a serializable configuration for Keras model loading.

        Returns:
            dict[str, Any]: Standard Keras layer configuration.
        """
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(layers.Layer):
    """Tile 2D coordinates across the flattened LiDAR sequence length.

    Args:
        grid_length (int): Number of flattened LiDAR positions. Larger values
            repeat the coordinate pair more times and therefore consume more
            memory. Smaller values reduce memory use but would break models that
            expect the original V7 token length.
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        """Initialize the coordinate tiling layer.

        Args:
            grid_length (int): Sequence length to broadcast the 2D coordinates
                across. This should match the flattened LiDAR token count used by
                the saved model.
            **kwargs (Any): Additional Keras layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Broadcast coordinates across the sequence axis expected by Conv1D.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled coordinate tensor with shape
            ``(batch, grid_length, 2)``.

        Raises:
            tf.errors.InvalidArgumentError: If TensorFlow encounters an invalid
                shape during graph execution.
        """
        expanded = tf.expand_dims(inputs, axis=1)
        multiples = tf.stack([1, self.grid_length, 1])
        return tf.tile(expanded, multiples)

    def get_config(self) -> dict[str, Any]:
        """Return a serializable configuration for Keras model loading.

        Returns:
            dict[str, Any]: Standard Keras layer configuration including the
            stored ``grid_length``.
        """
        config = super().get_config()
        config["grid_length"] = self.grid_length
        return config


def _write_report(report: str, *, name: str, output_dir: Optional[Path]) -> None:
    """Persist a text report when an output directory is configured.

    Args:
        report (str): Fully formatted report body to write.
        name (str): Logical artifact name used to derive the output filename.
            This value is sanitized only by replacing path separators with
            underscores, so it should remain reasonably short and descriptive.
        output_dir (Optional[Path]): Destination directory for the report.
            If a path is provided, the directory is created when missing and the
            report is written as ``<name>.txt``. If ``None``, the function does
            not write anything to disk.

    Returns:
        None: The function performs file-system side effects only.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name.replace(os.sep, '_')}.txt"
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Report saved to {output_path}")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path (Path): Path to the JSON file. The file must exist and contain a
            top-level JSON object.

    Returns:
        dict[str, Any]: Parsed JSON object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contents are not a JSON object.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def _format_size_bytes(num_bytes: int) -> str:
    """Convert a byte count into a compact human-readable string.

    Args:
        num_bytes (int): Size in bytes. Values less than zero are not valid.

    Returns:
        str: Human-readable size string such as ``"12.34 MB"``.

    Raises:
        ValueError: If ``num_bytes`` is negative.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative.")

    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def _get_directory_size_bytes(path: Path) -> int:
    """Compute the total size of all files contained in a directory tree.

    Args:
        path (Path): Directory whose recursive file size should be summed. The
            directory must exist; symlink sizes are counted based on the target
            file metadata returned by ``Path.stat``.

    Returns:
        int: Total byte size across all regular files under ``path``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        NotADirectoryError: If ``path`` is not a directory.
        OSError: If file metadata cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")

    total_size = 0
    # Walk the tree explicitly so the report captures the full deployed bundle,
    # not just top-level files like predictor.pkl.
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def _resolve_best_ensemble_predictor(results_path: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve the exported AutoGluon predictor from an ensemble results file.

    Args:
        results_path (Path): Path to the stacking run ``results.json``. The file
            must contain the ``autogluon_path`` key at the top level.

    Returns:
        tuple[Path, dict[str, Any]]: Predictor directory path and parsed results
        JSON for use in downstream reporting.

    Raises:
        FileNotFoundError: If ``results_path`` or the referenced predictor
            directory does not exist.
        KeyError: If the required ``autogluon_path`` key is missing.
        NotADirectoryError: If the referenced predictor path is not a directory.
    """
    results = _load_json(results_path)
    predictor_path = Path(results["autogluon_path"])
    if not predictor_path.exists():
        # Results files often preserve the original training machine's absolute
        # path. If the bundle was moved, retry with the same directory name next
        # to the checked-in results file before failing.
        relocated_predictor_path = results_path.parent / predictor_path.name
        if relocated_predictor_path.exists():
            predictor_path = relocated_predictor_path
        else:
            raise FileNotFoundError(f"Ensemble predictor directory does not exist: {predictor_path}")
    if not predictor_path.is_dir():
        raise NotADirectoryError(f"Ensemble predictor path is not a directory: {predictor_path}")
    return predictor_path, results


def _resolve_best_kd_model(study_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Resolve the best distilled student model from a KD study directory.

    Args:
        study_dir (Path): Root directory of a KD Optuna study. The directory
            must include ``best_trial.json`` and a ``trials/trial_XXXX`` folder
            containing ``student_distilled.keras`` and ``results.json`` for the
            recorded best trial.

    Returns:
        tuple[Path, dict[str, Any], dict[str, Any]]: Student model path, parsed
        study summary, and parsed best-trial ``results.json``.

    Raises:
        FileNotFoundError: If any required summary or artifact file is missing.
        KeyError: If the summary does not include ``best_trial_number``.
    """
    summary_path = study_dir / "best_trial.json"
    summary = _load_json(summary_path)
    best_trial_number = int(summary["best_trial_number"])
    trial_dir = study_dir / "trials" / f"trial_{best_trial_number:04d}"
    model_path = trial_dir / "student_distilled.keras"
    results_path = trial_dir / "results.json"

    if not model_path.exists():
        raise FileNotFoundError(f"KD student model does not exist: {model_path}")

    return model_path, summary, _load_json(results_path)


def _prepare_inputs(model: tf.keras.Model, lidar: np.ndarray, coords: np.ndarray) -> np.ndarray | list[np.ndarray]:
    """Map synthetic LiDAR and coordinate arrays to the model input signature.

    Args:
        model (tf.keras.Model): Loaded base model whose input tensors determine
            input ordering and assignment.
        lidar (np.ndarray): LiDAR tensor shaped for the base model's LiDAR
            input.
        coords (np.ndarray): Coordinate tensor shaped for the base model's
            coordinate input.

    Returns:
        np.ndarray | list[np.ndarray]: Single input array for single-input
        models, or an ordered list matching ``model.inputs`` for multi-input
        models.

    Raises:
        ValueError: If the provided arrays cannot be matched to the model's
            input signature.
    """
    lidar = np.asarray(lidar, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    lidar_shape = tuple(lidar.shape[1:])
    coords_shape = tuple(coords.shape[1:])

    def compatible(expected: tuple[int, ...], actual: tuple[int, ...]) -> bool:
        if len(expected) != len(actual):
            return False
        for exp, act in zip(expected, actual):
            if exp is not None and int(exp) != int(act):
                return False
        return True

    prepared_by_name: dict[str, np.ndarray] = {}
    for tensor in model.inputs:
        expected = tuple(tensor.shape[1:])
        input_name = tensor.name.split(":")[0]
        if "lidar" in input_name:
            candidate = lidar
        elif "coord" in input_name:
            candidate = coords
        elif compatible(expected, lidar_shape):
            candidate = lidar
        elif compatible(expected, coords_shape):
            candidate = coords
        else:
            raise ValueError(
                f"Cannot match input {tensor.name} expected {tensor.shape}. "
                f"lidar={lidar.shape}, coords={coords.shape}."
            )
        prepared_by_name[input_name] = candidate

    ordered_inputs = [prepared_by_name[t.name.split(":")[0]] for t in model.inputs]
    if len(ordered_inputs) == 1:
        return ordered_inputs[0]
    return ordered_inputs


def _build_meta_feature_names(classes: np.ndarray, expected_models: int) -> list[str]:
    """Create the exact stacking feature names expected by the AutoGluon model.

    Args:
        classes (np.ndarray): Ordered class ids retained by the stacking run.
        expected_models (int): Number of base models contributing probability
            blocks.

    Returns:
        list[str]: Feature names in the ``m<idx>_class_<class_id>`` format.

    Raises:
        ValueError: If ``expected_models`` is less than one.
    """
    if expected_models < 1:
        raise ValueError(f"expected_models must be >= 1, got {expected_models}.")
    return [
        f"m{model_index + 1}_class_{int(class_id)}"
        for model_index in range(expected_models)
        for class_id in classes
    ]


def _infer_stacking_layout_from_predictor(predictor: TabularPredictor) -> tuple[np.ndarray, int, list[str]]:
    """Infer model count and class order from persisted predictor features.

    Args:
        predictor (TabularPredictor): Loaded AutoGluon predictor whose feature
            names follow the stacking convention ``m<idx>_class_<class_id>``.

    Returns:
        tuple[np.ndarray, int, list[str]]: Ordered class ids, number of base
        models, and the predictor feature list.

    Raises:
        ValueError: If the predictor feature names do not follow the expected
            stacking naming convention or use inconsistent class order.
    """
    feature_names = predictor.features()
    pattern = re.compile(r"^m(\d+)_class_(\d+)$")
    classes_by_model: dict[int, list[int]] = {}
    for feature_name in feature_names:
        match = pattern.match(feature_name)
        if not match:
            raise ValueError(f"Unexpected predictor feature name for stacking benchmark: {feature_name}")
        model_index = int(match.group(1))
        class_id = int(match.group(2))
        classes_by_model.setdefault(model_index, []).append(class_id)

    if not classes_by_model:
        raise ValueError("Predictor does not expose any stacking feature names.")

    expected_models = max(classes_by_model)
    reference_classes = classes_by_model[1]
    for model_index in range(1, expected_models + 1):
        if classes_by_model.get(model_index) != reference_classes:
            raise ValueError("Predictor feature layout uses inconsistent class ordering across models.")

    return np.asarray(reference_classes, dtype=np.int64), expected_models, feature_names


def _build_stacking_meta_frame(predictions: Sequence[np.ndarray], classes: np.ndarray) -> pd.DataFrame:
    """Convert base-model probability outputs into the stacker's input frame.

    Args:
        predictions (Sequence[np.ndarray]): Per-model probability matrices with
            shape ``(N, C_full)``.
        classes (np.ndarray): Ordered subset of class ids retained by the
            stacking training run.

    Returns:
        pd.DataFrame: Meta-feature frame with one probability column per
        ``(model, class)`` pair in the exact layout used during training.

    Raises:
        ValueError: If no predictions are provided or the prediction matrices do
            not agree on sample count.
    """
    if not predictions:
        raise ValueError("At least one prediction matrix is required to build stacking features.")

    sample_count = int(np.asarray(predictions[0]).shape[0])
    selected_blocks: list[np.ndarray] = []
    for prediction in predictions:
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape[0] != sample_count:
            raise ValueError("All prediction matrices must share the same sample count.")
        selected_blocks.append(prediction[:, classes])

    stacked_features = np.concatenate(selected_blocks, axis=1)
    feature_names = _build_meta_feature_names(classes, expected_models=len(predictions))
    return pd.DataFrame(stacked_features, columns=feature_names)


def _measure_stacking_pipeline_once(
    models: Sequence[tf.keras.Model],
    predictor: TabularPredictor,
    lidar_input: np.ndarray,
    coord_input: np.ndarray,
    classes: np.ndarray,
    predictor_feature_names: Sequence[str],
    coord_scaler: Any | None,
) -> dict[str, float]:
    """Execute one full stacking inference pass and return per-part wall-clock times.

    Each base model is timed individually. The AutoGluon meta-learner call is
    timed separately so the caller can break down pipeline latency by component.

    Args:
        models (Sequence[tf.keras.Model]): Loaded base Keras models.
        predictor (TabularPredictor): Persisted AutoGluon meta-learner.
        lidar_input (np.ndarray): Single-batch LiDAR input tensor.
        coord_input (np.ndarray): Single-batch raw coordinate input tensor.
        classes (np.ndarray): Ordered class ids used by the meta-learner.
        predictor_feature_names (Sequence[str]): Exact feature order expected by
            the AutoGluon predictor.
        coord_scaler (Any | None): Optional scikit-learn compatible scaler with
            ``transform``. If provided, coordinates are transformed before base
            model inference. If ``None``, raw coordinates are used as-is.

    Returns:
        dict[str, float]: Per-part wall-clock seconds. Keys are
        ``"model_<i>"`` (0-based) for each base model and ``"meta"`` for the
        AutoGluon predictor call.

    Raises:
        ValueError: If the stacked prediction outputs are inconsistent.
    """
    raw_coords = np.asarray(coord_input, dtype=np.float32)
    scaled_coords = coord_scaler.transform(raw_coords) if coord_scaler is not None else raw_coords

    part_latencies: dict[str, float] = {}
    prediction_blocks: list[np.ndarray] = []
    # Time each base model independently so the caller can report per-model
    # latency in addition to the total pipeline time.
    for i, model in enumerate(models):
        prepared_inputs = _prepare_inputs(model, lidar_input, scaled_coords)
        t0 = time.perf_counter()
        prediction_blocks.append(model.predict(prepared_inputs, batch_size=1, verbose=0))
        part_latencies[f"model_{i}"] = time.perf_counter() - t0

    meta_frame = _build_stacking_meta_frame(prediction_blocks, classes)
    meta_frame = meta_frame.loc[:, list(predictor_feature_names)]
    t0 = time.perf_counter()
    predictor.predict_proba(meta_frame, as_multiclass=True)
    part_latencies["meta"] = time.perf_counter() - t0

    return part_latencies


def measure_stacking_ensemble_pipeline_stats(
    *,
    predictor_path: Path,
    results: dict[str, Any],
    results_path: Path,
    name: str,
    benchmark_runs: int = 10,
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
    reference_top1: Optional[float] = None,
    reference_top10: Optional[float] = None,
) -> str:
    """Measure end-to-end stats for the stacking ensemble pipeline.

    Args:
        predictor_path (Path): AutoGluon predictor directory for the meta-learner.
        results (dict[str, Any]): Parsed stacking ``results.json`` containing
            base model paths and artifact locations.
        results_path (Path): Path to the stacking ``results.json`` file. This
            is used to resolve sibling preprocessing artifacts when the JSON does
            not explicitly list them.
        name (str): Report name used in console output and file naming.
        benchmark_runs (int): Number of repeated end-to-end inference passes.
            Higher values reduce timing noise but increase total runtime.
        output_dir (Optional[Path]): Destination directory for the rendered
            report. If ``None``, the report is printed only.
        reference_top1 (Optional[float]): Optional Top-1 accuracy reference in
            ``[0.0, 1.0]``.
        reference_top10 (Optional[float]): Optional Top-10 accuracy reference in
            ``[0.0, 1.0]``.

    Returns:
        str: Rendered stacking report text.

    Raises:
        FileNotFoundError: If required ensemble artifacts are missing.
        ValueError: If ``benchmark_runs`` is not positive.
    """
    if benchmark_runs < 1:
        raise ValueError(f"benchmark_runs must be >= 1, got {benchmark_runs}.")

    predictor = TabularPredictor.load(str(predictor_path))
    classes, expected_models, predictor_feature_names = _infer_stacking_layout_from_predictor(predictor)

    artifacts = results.get("artifacts", {})
    scaler_path = Path(artifacts["coord_scaler"]) if artifacts.get("coord_scaler") else results_path.parent / "coord_scaler.pkl"
    coord_scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as handle:
            coord_scaler = pickle.load(handle)

    base_model_paths = [Path(path_str) for path_str in results["config"]["model_paths"]]
    models = [load_model(model_path) for model_path in base_model_paths]
    if len(models) != expected_models:
        raise ValueError(
            f"Predictor expects {expected_models} base model(s), but results.json lists {len(models)} model path(s)."
        )

    # Use a single synthetic sample with the real model input shapes so the
    # benchmark captures the deployed pipeline without requiring dataset access.
    first_model = models[0]
    lidar_shape = (1, 20, 200, 10)
    coord_shape = (1, 2)
    for tensor in first_model.inputs:
        input_name = tensor.name.split(":")[0]
        shape = tuple(int(dim) if dim is not None else 1 for dim in tensor.shape[1:])
        if "lidar" in input_name:
            lidar_shape = (1, *shape)
        elif "coord" in input_name:
            coord_shape = (1, *shape)

    lidar_input = np.zeros(lidar_shape, dtype=np.float32)
    coord_input = np.zeros(coord_shape, dtype=np.float32)

    # Profile each base model once to aggregate exact neural-network params,
    # FLOPs, and serialized size. The AutoGluon stack itself is heterogeneous,
    # so those architecture metrics are only well-defined for the Keras stage.
    base_stats = [get_model_stats(model, batch_size=1, device="cpu", test_runs=1, verbose=0) for model in models]
    total_parameters = int(sum(int(stats["parameters"]) for stats in base_stats))
    total_model_size_bytes = int(sum(int(stats["model_size"]) for stats in base_stats))
    total_flops = int(sum(int(stats["flops"]) for stats in base_stats))
    total_macs = int(sum(int(stats["macs"]) for stats in base_stats))
    predictor_bundle_size_bytes = _get_directory_size_bytes(predictor_path)

    _measure_stacking_pipeline_once(
        models,
        predictor,
        lidar_input,
        coord_input,
        classes,
        predictor_feature_names,
        coord_scaler,
    )

    latencies_seconds: list[float] = []
    part_latency_accumulator: dict[str, list[float]] = {}
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        part_latencies = _measure_stacking_pipeline_once(
            models,
            predictor,
            lidar_input,
            coord_input,
            classes,
            predictor_feature_names,
            coord_scaler,
        )
        latencies_seconds.append(time.perf_counter() - start)
        for part, t in part_latencies.items():
            part_latency_accumulator.setdefault(part, []).append(t)

    average_latency = mean(latencies_seconds)
    part_latency_means: dict[str, float] = {
        part: mean(times) for part, times in part_latency_accumulator.items()
    }
    std_latency = pstdev(latencies_seconds) if len(latencies_seconds) > 1 else 0.0
    peak_latency = max(latencies_seconds)
    throughput_samples_per_second = 1.0 / average_latency if average_latency > 0 else float("inf")

    report_lines = [
        f"Stacking ensemble stats for {name}",
        f"predictor_path: {predictor_path}",
        f"base_model_count: {len(models)}",
        f"base_model_paths: {', '.join(str(path) for path in base_model_paths)}",
        f"base_model_parameters_total: {total_parameters}",
        f"base_model_size_total: {_format_size_bytes(total_model_size_bytes)} ({total_model_size_bytes} bytes)",
        f"base_model_flops_total: {total_flops}",
        f"base_model_macs_total: {total_macs}",
        f"meta_predictor_bundle_size: {_format_size_bytes(predictor_bundle_size_bytes)} ({predictor_bundle_size_bytes} bytes)",
        f"total_deployed_size: {_format_size_bytes(total_model_size_bytes + predictor_bundle_size_bytes)} ({total_model_size_bytes + predictor_bundle_size_bytes} bytes)",
        f"benchmark_runs: {benchmark_runs}",
        f"end_to_end_mean_seconds: {average_latency:.6f}",
        f"end_to_end_std_seconds: {std_latency:.6f}",
        f"end_to_end_peak_seconds: {peak_latency:.6f}",
        f"throughput_samples_per_second: {throughput_samples_per_second:.2f}",
        *[
            f"base_model_{i + 1}_mean_seconds: {part_latency_means.get(f'model_{i}', 0.0):.6f}"
            for i in range(len(models))
        ],
        f"meta_predictor_mean_seconds: {part_latency_means.get('meta', 0.0):.6f}",
        "meta_predictor_params: N/A (heterogeneous AutoGluon ensemble)",
        "meta_predictor_flops: N/A (heterogeneous AutoGluon ensemble)",
    ]

    if reference_top1 is not None:
        report_lines.append(f"reference_top1: {reference_top1 * 100:.2f}%")
    if reference_top10 is not None:
        report_lines.append(f"reference_top10: {reference_top10 * 100:.2f}%")

    report = "\n".join(report_lines)
    print("\n" + report + "\n")
    _write_report(report, name=name, output_dir=output_dir)
    return report


def measure_and_print_model_stats(
    model: tf.keras.Model,
    *,
    name: str = "model",
    batch_size: int = 1,
    gpu_index: int = 0,
    test_runs: int = 5,
    verbose: int = 1,
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
    extra_attrs: Optional[dict[str, Any]] = None,
) -> str:
    """Collect TensorFlow model stats on CPU and GPU and emit a text report.

    Args:
        model (tf.keras.Model): Loaded Keras model to profile. The model must be
            fully built so parameter counts and inference tracing are available.
        name (str): Report name used in the rendered output and output filename.
            If the same name is reused across runs and ``output_dir`` is set, the
            report file is overwritten.
        batch_size (int): Batch size used by the profiler. Larger values can
            increase throughput while also raising memory usage; smaller values
            better approximate single-sample latency.
        gpu_index (int): Zero-based GPU index used for the GPU pass. If that GPU
            is unavailable, the function keeps the CPU results and skips the GPU
            branch.
        test_runs (int): Number of repeated profiling runs requested from
            ``get_model_stats``. Higher values reduce timing noise but increase
            total measurement time.
        verbose (int): Verbosity forwarded to ``get_model_stats``. If ``0``, the
            underlying profiler runs quietly. If ``1`` or higher, it emits more
            progress information, which is useful for debugging at the cost of
            extra console output.
        output_dir (Optional[Path]): Destination directory for the rendered
            report. If a path is provided, the report is written to disk. If
            ``None``, the report is printed only.
        extra_attrs (Optional[dict[str, Any]]): Additional attributes injected
            into the rendered report. If provided, these keys supplement the
            default ``model_name`` and ``batch_size`` metadata.

    Returns:
        str: Rendered profiling report text.

    Raises:
        RuntimeError: If neither CPU nor GPU stats could be collected.
    """
    cpu_stats = get_model_stats(
        model,
        batch_size=batch_size,
        device="cpu",
        test_runs=test_runs,
        verbose=verbose,
    )

    gpu_stats = None
    try:
        gpu_stats = get_model_stats(
            model,
            batch_size=batch_size,
            device=f"gpu/{gpu_index}",
            test_runs=test_runs,
            verbose=verbose,
        )
    except RuntimeError as exc:
        print(f"[measure_model_stats] Skipping GPU profiling: {exc}")
    except Exception as exc:  # pragma: no cover - defensive guard for demo usage
        print(f"[measure_model_stats] GPU profiling failed: {exc}")

    structural_stats = gpu_stats or cpu_stats
    if not structural_stats:
        raise RuntimeError("No stats collected from CPU or GPU runs")

    report_attrs = {"model_name": name, "batch_size": batch_size}
    if extra_attrs:
        report_attrs.update(extra_attrs)

    report = render_model_stats_report(
        structural_stats,
        cpu_stats=cpu_stats,
        gpu_stats=gpu_stats,
        extra_attrs=report_attrs,
    )

    print("\n" + report + "\n")

    _write_report(report, name=name, output_dir=output_dir)

    return report


def measure_autogluon_predictor_stats(
    predictor_path: Path,
    *,
    name: str,
    benchmark_rows: int = 1,
    benchmark_runs: int = 10,
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
    reference_top1: Optional[float] = None,
    reference_top10: Optional[float] = None,
) -> str:
    """Measure deployment-oriented stats for an AutoGluon tabular predictor.

    Args:
        predictor_path (Path): Directory containing a persisted
            ``TabularPredictor`` bundle. The directory must be loadable via
            ``TabularPredictor.load``.
        name (str): Logical report name used in the printed header and output
            filename.
        benchmark_rows (int): Number of synthetic rows used per benchmark call.
            If ``1``, the report emphasizes single-sample latency. If greater
            than ``1``, the measured latency reflects small-batch inference and
            the computed throughput may be more representative of bulk scoring.
        benchmark_runs (int): Number of repeated ``predict_proba`` calls. Higher
            values reduce variance but take longer to execute.
        output_dir (Optional[Path]): Destination directory for the rendered text
            report. If ``None``, the report is printed only.
        reference_top1 (Optional[float]): Optional reference Top-1 accuracy in
            ``[0.0, 1.0]``. If provided, it is included in the report for easier
            comparison against architectural cost.
        reference_top10 (Optional[float]): Optional reference Top-10 accuracy in
            ``[0.0, 1.0]``. If provided, it is included beside ``reference_top1``
            to capture ranking quality relevant to beam selection.

    Returns:
        str: Rendered predictor stats report text.

    Raises:
        FileNotFoundError: If ``predictor_path`` does not exist.
        NotADirectoryError: If ``predictor_path`` is not a directory.
        ValueError: If ``benchmark_rows`` or ``benchmark_runs`` is not positive.
    """
    if benchmark_rows < 1:
        raise ValueError(f"benchmark_rows must be >= 1, got {benchmark_rows}.")
    if benchmark_runs < 1:
        raise ValueError(f"benchmark_runs must be >= 1, got {benchmark_runs}.")
    if not predictor_path.exists():
        raise FileNotFoundError(f"Predictor directory does not exist: {predictor_path}")
    if not predictor_path.is_dir():
        raise NotADirectoryError(f"Predictor path is not a directory: {predictor_path}")

    load_start = time.perf_counter()
    predictor = TabularPredictor.load(str(predictor_path))
    load_seconds = time.perf_counter() - load_start

    feature_names = predictor.features()
    synthetic_frame = pd.DataFrame(
        [{feature_name: 0.0 for feature_name in feature_names} for _ in range(benchmark_rows)]
    )

    # Run one warm-up prediction so the repeated timing loop better reflects a
    # steady-state deployment path after lazy objects have been initialized.
    predictor.predict_proba(synthetic_frame, as_multiclass=True)

    latencies_seconds: list[float] = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        predictor.predict_proba(synthetic_frame, as_multiclass=True)
        latencies_seconds.append(time.perf_counter() - start)

    model_names = predictor.model_names()
    predictor_size_bytes = _get_directory_size_bytes(predictor_path)
    avg_latency_seconds = mean(latencies_seconds)
    std_latency_seconds = pstdev(latencies_seconds) if len(latencies_seconds) > 1 else 0.0
    rows_per_second = benchmark_rows / avg_latency_seconds if avg_latency_seconds > 0 else float("inf")

    report_lines = [
        f"AutoGluon predictor stats for {name}",
        f"artifact_path: {predictor_path}",
        f"problem_type: {predictor.problem_type}",
        f"label_column: {predictor.label}",
        f"feature_count: {len(feature_names)}",
        f"class_count: {len(getattr(predictor, 'class_labels', []))}",
        f"model_count: {len(model_names)}",
        f"models: {', '.join(model_names)}",
        f"bundle_size: {_format_size_bytes(predictor_size_bytes)} ({predictor_size_bytes} bytes)",
        f"load_time_seconds: {load_seconds:.6f}",
        f"benchmark_rows: {benchmark_rows}",
        f"benchmark_runs: {benchmark_runs}",
        f"predict_proba_mean_seconds: {avg_latency_seconds:.6f}",
        f"predict_proba_std_seconds: {std_latency_seconds:.6f}",
        f"throughput_rows_per_second: {rows_per_second:.2f}",
    ]

    if reference_top1 is not None:
        report_lines.append(f"reference_top1: {reference_top1 * 100:.2f}%")
    if reference_top10 is not None:
        report_lines.append(f"reference_top10: {reference_top10 * 100:.2f}%")

    report = "\n".join(report_lines)
    print("\n" + report + "\n")
    _write_report(report, name=name, output_dir=output_dir)
    return report


def patch_lambda_globals(model: tf.keras.Model) -> None:
    """Inject TensorFlow globals into Lambda layers after model loading.

    Args:
        model (tf.keras.Model): Loaded Keras model that may include Lambda
            layers referencing global names such as ``tf``, ``np``, or
            ``K``. The function mutates those Lambda function globals in place.

    Returns:
        None: The input model is patched in place.

    Raises:
        AttributeError: If the model contains malformed layer objects that do
            not expose the Keras attributes accessed during traversal.
    """
    def walk(m):
        for layer in m.layers:
            yield layer
            if isinstance(layer, tf.keras.Model):
                yield from walk(layer)

    for layer in walk(model):
        if isinstance(layer, tf.keras.layers.Lambda):
            fn = getattr(layer, "function", None) or getattr(layer, "_function", None)
            if callable(fn) and hasattr(fn, "__globals__"):
                fn.__globals__.setdefault("tf", tf)
                fn.__globals__.setdefault("np", np)
                fn.__globals__.setdefault("K", tf.keras.backend)


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Keras model that may contain Lambda layers.

    Args:
        model_path (Path): Path to the ``.keras`` model file. The file must
            exist and be compatible with the current TensorFlow installation.

    Returns:
        tf.keras.Model: Loaded Keras model with Lambda globals patched for the
        symbols used across this repository.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        OSError: If TensorFlow cannot deserialize the model file.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # The saved architecture includes Lambda layers, so safe mode must be
    # disabled before deserialization can restore those callables.
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects={
            "tf": tf,
            "LidarToOneHot": LidarToOneHot,
            "CoordTileFlat": CoordTileFlat,
            "raywise>LidarToOneHot": LidarToOneHot,
            "raywise>CoordTileFlat": CoordTileFlat,
        },
    )
    patch_lambda_globals(model)
    return model


def measure_best_ensemble_case(*, output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR) -> str:
    """Measure the best stacking-ensemble artifact available in this workspace.

    Args:
        output_dir (Optional[Path]): Destination directory for the generated
            report. If ``None``, the report is printed only.

    Returns:
        str: Rendered predictor stats report.

    Raises:
        FileNotFoundError: If the ensemble results file or predictor bundle is
            missing.
        KeyError: If the ensemble results file lacks required keys.
    """
    predictor_path, results = _resolve_best_ensemble_predictor(BEST_ENSEMBLE_RESULTS_PATH)
    top1 = float(results["meta_test_topk"]["top_1"])
    top10 = float(results["meta_test_topk"]["top_10"])
    return measure_stacking_ensemble_pipeline_stats(
        predictor_path=predictor_path,
        results=results,
        results_path=BEST_ENSEMBLE_RESULTS_PATH,
        name="ensemble_best_case_top10_stack",
        benchmark_runs=10,
        output_dir=output_dir,
        reference_top1=top1,
        reference_top10=top10,
    )


def measure_best_kd_case(*, output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR) -> str:
    """Measure the best knowledge-distillation student model on disk.

    Args:
        output_dir (Optional[Path]): Destination directory for the generated
            report. If ``None``, the report is printed only.

    Returns:
        str: Rendered Keras profiling report.

    Raises:
        FileNotFoundError: If the KD study summary or best student model is
            missing.
        KeyError: If required summary or trial metrics are absent.
    """
    model_path, summary, results = _resolve_best_kd_model(BEST_KD_STUDY_DIR)
    model = load_model(model_path)
    best_trial_number = int(summary["best_trial_number"])
    return measure_and_print_model_stats(
        model,
        name=f"kd_best_case_trial_{best_trial_number:04d}",
        batch_size=1,
        gpu_index=0,
        test_runs=10,
        verbose=1,
        output_dir=output_dir,
        extra_attrs={
            "study_dir": str(BEST_KD_STUDY_DIR),
            "student_model_path": str(model_path),
            "best_trial_number": best_trial_number,
            "reference_top1": results["student_test_topk"]["top_1"],
            "reference_top10": results["student_test_topk"]["top_10"],
        },
    )


if __name__ == "__main__":
    measure_best_ensemble_case(output_dir=DEFAULT_OUTPUT_DIR)
    measure_best_kd_case(output_dir=DEFAULT_OUTPUT_DIR)
