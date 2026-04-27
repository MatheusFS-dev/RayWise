"""Evaluate a cascaded LOS/NLOS routing pipeline with top-k beam metrics.

This script performs a two-stage inference flow on raw S009 data:
1. A binary classifier predicts whether each sample is LOS or NLOS.
2. The sample is routed to the corresponding specialized beam model.

The implementation mirrors the top-k computation style used in
``src/evaluate_raw.py`` while preserving each model's own coordinate scaler.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _load_dataset_raw import (
    load_dataset_raw_sparse_labels_by_condition,
    load_dataset_raw_sparse_labels_s009_only,
)


DATA_SEED = 0
BATCH_SIZE = 64
TOP_K = (1, 3, 5, 10, 20, 30, 50)

# If None, skip single-sample timing benchmark. If int, run benchmark on a
# single fixed sample repeated N times, measuring each model with batch_size=1.
SINGLE_SAMPLE_TIMING_SAMPLES: int | None = 100

# Fixed sample index used by the single-sample repeated benchmark.
SINGLE_SAMPLE_TIMING_INDEX = 0

# If True, model loading expects custom layers to be available and uses
# safe-mode deserialization. If False, it allows Lambda deserialization and
# then patches Lambda globals for backwards compatibility.
USE_CUSTOM_LAYERS = False

# Dataset paths used by the raw S009 loader.
DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")
S008_COORD_CSV = DATA_ROOT / "Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
S008_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s008/processed_raw_data/lidar_data_s008"
S008_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s008/baseline_data/beam_output/beams_output_s008.npz"
S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

# Direct model paths for each stage.
CLASSIFIER_MODEL_PATH = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_classifier/nas_classifier_NLOS_LOS_v0.0/optuna_study/model/top_1_trial_535.keras"
)
LOS_MODEL_PATH = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_LOS/optuna_study/model/top_1_trial_1160.keras"
)
NLOS_MODEL_PATH = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_NLOS/optuna_study/model/top_1_trial_188.keras"
)

# Condition encoding used by the classifier training code.
LABEL_LOS = 0
LABEL_NLOS = 1


@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(tf.keras.layers.Layer):
    """Convert raw LiDAR semantic codes into 4 binary semantic channels.

    Returns:
        tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)`` where the
        channels encode base station, vehicle, obstacle, and free space.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic channel conversion.

        Args:
            inputs (tf.Tensor): LiDAR tensor with shape ``(batch, 20, 200, 10)``
                and semantic values in ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Converted tensor with shape ``(batch, 20, 200, 4)``.
        """
        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)


@tf.keras.utils.register_keras_serializable(package="RayWise")
class LidarChannelOneHot(tf.keras.layers.Layer):
    """Convert integer LiDAR channel values to one-hot vectors per channel.

    This layer is used by the classifier architecture where each of the 10
    LiDAR channels is expanded from a scalar semantic code to a small one-hot
    codebook.

    Args:
        offset (int): Value added before one-hot to map ``{-2, -1, 0, 1}`` into
            ``{0, 1, 2, 3}``.
        depth (int): Number of one-hot bins.
    """

    def __init__(self, offset: int = 2, depth: int = 4, **kwargs: Any) -> None:
        """Initialize the layer.

        Args:
            offset (int): Offset used to map semantic codes to non-negative
                indices.
            depth (int): One-hot depth.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.offset = int(offset)
        self.depth = int(depth)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply one-hot expansion for each LiDAR channel.

        Args:
            inputs (tf.Tensor): Integer-like tensor of semantic codes.

        Returns:
            tf.Tensor: One-hot tensor with one extra trailing dimension.
        """
        indices = tf.cast(inputs + self.offset, tf.int32)
        return tf.one_hot(indices, depth=self.depth, dtype=tf.float32)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Return static output shape for Keras shape inference.

        Args:
            input_shape (tf.TensorShape): Input shape.

        Returns:
            tf.TensorShape: Shape with ``depth`` appended at the end.
        """
        return tf.TensorShape(tuple(input_shape) + (self.depth,))

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer config.

        Returns:
            dict[str, Any]: Keras config dictionary.
        """
        config = super().get_config()
        config.update({"offset": self.offset, "depth": self.depth})
        return config


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(tf.keras.layers.Layer):
    """Tile 2D coordinates across a flattened LiDAR sequence.

    Args:
        grid_length (int): Flattened LiDAR sequence length.
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        """Store grid length for tiling.

        Args:
            grid_length (int): Number of tokens after LiDAR flattening.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Tile each coordinate pair across the flattened sequence.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tensor with shape ``(batch, grid_length, 2)``.
        """
        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer config.

        Returns:
            dict[str, Any]: Keras config dictionary.
        """
        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


def _fit_scalers_like_training() -> tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Rebuild classifier/LOS/NLOS scalers exactly like original training code.

    The original scripts fit each ``StandardScaler`` on the S008 training split:
    - Classifier: split S008 combined LOS+NLOS with stratification by condition.
    - LOS specialist: split only S008 LOS samples.
    - NLOS specialist: split only S008 NLOS samples.

    All splits use ``test_size=0.2``, ``random_state=DATA_SEED``, and
    ``shuffle=True``.

    Returns:
        tuple[StandardScaler, StandardScaler, StandardScaler]:
            ``(classifier_scaler, los_scaler, nlos_scaler)``.

    Raises:
        ValueError: If any required split is empty.
    """
    (
        (s008_lidar_los, s008_coord_los, _),
        _,
        (s008_lidar_nlos, s008_coord_nlos, _),
        _,
    ) = load_dataset_raw_sparse_labels_by_condition(
        s008_coord_csv=str(S008_COORD_CSV),
        s008_lidar_folder=str(S008_LIDAR_FOLDER),
        s008_beam_output_path=str(S008_BEAM_OUTPUT),
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
    )

    # Reproduce classifier preprocessing: fit on S008 train split with LOS/NLOS stratification.
    s008_lidar_all = np.concatenate([s008_lidar_los, s008_lidar_nlos], axis=0)
    s008_coord_all = np.concatenate([s008_coord_los, s008_coord_nlos], axis=0)
    s008_condition = np.concatenate(
        [
            np.full(shape=(s008_coord_los.shape[0],), fill_value=LABEL_LOS, dtype=np.int32),
            np.full(shape=(s008_coord_nlos.shape[0],), fill_value=LABEL_NLOS, dtype=np.int32),
        ],
        axis=0,
    )
    _, _, classifier_x_coord_train, _, _, _ = train_test_split(
        s008_lidar_all,
        s008_coord_all,
        s008_condition,
        test_size=0.2,
        random_state=DATA_SEED,
        shuffle=True,
        stratify=s008_condition,
    )
    classifier_scaler = StandardScaler()
    classifier_scaler.fit(classifier_x_coord_train)

    # Reproduce LOS specialist preprocessing: fit on S008 LOS train split.
    _, _, los_x_coord_train, _, _, _ = train_test_split(
        s008_lidar_los,
        s008_coord_los,
        np.zeros(s008_coord_los.shape[0], dtype=np.int32),
        test_size=0.2,
        random_state=DATA_SEED,
        shuffle=True,
    )
    los_scaler = StandardScaler()
    los_scaler.fit(los_x_coord_train)

    # Reproduce NLOS specialist preprocessing: fit on S008 NLOS train split.
    _, _, nlos_x_coord_train, _, _, _ = train_test_split(
        s008_lidar_nlos,
        s008_coord_nlos,
        np.ones(s008_coord_nlos.shape[0], dtype=np.int32),
        test_size=0.2,
        random_state=DATA_SEED,
        shuffle=True,
    )
    nlos_scaler = StandardScaler()
    nlos_scaler.fit(nlos_x_coord_train)

    return classifier_scaler, los_scaler, nlos_scaler


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a Keras model with custom objects used in this repository.

    Args:
        model_path (Path): Path to a saved ``.keras`` model.

    Returns:
        tf.keras.Model: Loaded model ready for inference.
    """
    custom_objects = {
        "tf": tf,
        "LidarToOneHot": LidarToOneHot,
        "CoordTileFlat": CoordTileFlat,
        "LidarChannelOneHot": LidarChannelOneHot,
        "raywise>LidarToOneHot": LidarToOneHot,
        "raywise>CoordTileFlat": CoordTileFlat,
        "RayWise>LidarChannelOneHot": LidarChannelOneHot,
    }

    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=USE_CUSTOM_LAYERS,
        compile=False,
        custom_objects=custom_objects,
    )
    return model


def patch_lambda_globals(model: tf.keras.Model) -> None:
    """Patch Lambda globals to improve deserialization compatibility.

    Args:
        model (tf.keras.Model): Loaded model that may contain Lambda layers.

    Returns:
        None: The model is patched in-place.
    """

    def walk(submodel: tf.keras.Model):
        for layer in submodel.layers:
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


def _prepare_inputs(model: tf.keras.Model, lidar: np.ndarray, coords: np.ndarray) -> Any:
    """Match LiDAR and coordinate tensors to the model input signature.

    Args:
        model (tf.keras.Model): Model whose input layout may vary by architecture.
        lidar (np.ndarray): LiDAR tensor.
        coords (np.ndarray): Coordinate tensor.

    Returns:
        Any: Single tensor or list of tensors in model input order.

    Raises:
        ValueError: If an input tensor cannot be matched by name or shape.
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

    by_name: dict[str, np.ndarray] = {}
    for tensor in model.inputs:
        expected = tuple(tensor.shape[1:])
        input_name = tensor.name.split(":")[0].lower()

        # Prefer semantic name matching to avoid ambiguity in two-input models.
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
                f"lidar={lidar.shape}, coords={coords.shape}"
            )

        by_name[tensor.name.split(":")[0]] = candidate

    ordered = [by_name[t.name.split(":")[0]] for t in model.inputs]
    return ordered if len(ordered) > 1 else ordered[0]


def _topk_stats(predictions: np.ndarray, labels: np.ndarray, ks: tuple[int, ...]) -> dict[int, dict[str, float]]:
    """Compute top-k accuracy counts similarly to ``src/evaluate_raw.py``.

    Args:
        predictions (np.ndarray): Model output matrix with shape ``(N, C)``.
        labels (np.ndarray): Sparse class ids with shape ``(N,)``.
        ks (tuple[int, ...]): Top-k values to evaluate.

    Returns:
        dict[int, dict[str, float]]: Per-k stats containing ``correct`` and
        ``accuracy``.

    Raises:
        ValueError: If predictions are not two-dimensional.
    """
    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions with shape (N, C), got {predictions.shape}")

    total = int(labels.shape[0])
    if total == 0:
        return {k: {"correct": 0, "accuracy": 0.0} for k in ks}

    num_classes = int(predictions.shape[1])
    sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
    stats: dict[int, dict[str, float]] = {}

    for k in ks:
        k_eff = min(int(k), num_classes)
        topk = sorted_indices[:, :k_eff]
        hits = (topk == labels[:, None]).any(axis=1)
        correct = int(hits.sum())
        stats[int(k)] = {"correct": correct, "accuracy": correct / total}

    return stats


def _decode_classifier_labels(predictions: np.ndarray) -> np.ndarray:
    """Decode classifier outputs into binary LOS/NLOS labels.

    Args:
        predictions (np.ndarray): Classifier outputs. If shape is ``(N, 1)``
            the model is treated as sigmoid and thresholded at 0.5. If shape is
            ``(N, 2)`` it is treated as softmax and decoded via argmax.

    Returns:
        np.ndarray: Integer labels where 0=LOS and 1=NLOS.

    Raises:
        ValueError: If output rank/shape is unsupported.
    """
    if predictions.ndim == 1:
        return (predictions >= 0.5).astype(np.int32)
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected classifier output shape: {predictions.shape}")
    if predictions.shape[1] == 1:
        return (predictions[:, 0] >= 0.5).astype(np.int32)
    if predictions.shape[1] == 2:
        return np.argmax(predictions, axis=1).astype(np.int32)
    raise ValueError(
        "Classifier output must be sigmoid (N,1) or softmax (N,2). "
        f"Received {predictions.shape}."
    )


def _condition_to_binary(condition_labels: np.ndarray) -> np.ndarray:
    """Map string LOS/NLOS condition labels to binary integers.

    Args:
        condition_labels (np.ndarray): Array of condition names.

    Returns:
        np.ndarray: Integer vector where 0=LOS and 1=NLOS.

    Raises:
        ValueError: If unknown condition values are present.
    """
    text = np.char.upper(np.asarray(condition_labels).astype(str))
    mapped = np.full(shape=text.shape, fill_value=-1, dtype=np.int32)
    mapped[text == "LOS"] = 0
    mapped[text == "NLOS"] = 1
    if np.any(mapped < 0):
        unknown = np.unique(text[mapped < 0])
        raise ValueError(f"Unexpected condition labels: {unknown.tolist()}")
    return mapped


def _print_topk_block(label: str, topk: dict[int, dict[str, float]], total: int, indent: str = "  ") -> None:
    """Print one formatted top-k block.

    Args:
        label (str): Prefix label (for example, ``"LOS "``).
        topk (dict[int, dict[str, float]]): Top-k statistics.
        total (int): Denominator used in the count display.
        indent (str): Output indentation.

    Returns:
        None: Prints to stdout.
    """
    for k in sorted(topk.keys()):
        stats = topk[k]
        print(f"{indent}{label}Top-{k}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{total})")


def _timing_stats(times_s: np.ndarray) -> dict[str, float]:
    """Summarize a vector of timing values in seconds.

    Args:
        times_s (np.ndarray): One-dimensional array of elapsed times in
            seconds.

    Returns:
        dict[str, float]: Statistics with keys ``count``, ``mean``, ``std``,
        ``min``, ``p50``, ``p90``, and ``max``. If ``times_s`` is empty,
        ``count`` is 0 and all numeric stats are 0.0.

    Raises:
        ValueError: If ``times_s`` is not one-dimensional.
    """
    if times_s.ndim != 1:
        raise ValueError(f"Expected 1D timing array, got shape {times_s.shape}")

    if times_s.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    return {
        "count": float(times_s.size),
        "mean": float(np.mean(times_s)),
        "std": float(np.std(times_s)),
        "min": float(np.min(times_s)),
        "p50": float(np.percentile(times_s, 50)),
        "p90": float(np.percentile(times_s, 90)),
        "max": float(np.max(times_s)),
    }


def _print_timing_stats_block(title: str, stats: dict[str, float], indent: str = "  ") -> None:
    """Print timing statistics formatted in milliseconds.

    Args:
        title (str): Label for the printed block.
        stats (dict[str, float]): Output from ``_timing_stats``.
        indent (str): Prefix indentation used for all printed lines.

    Returns:
        None: Prints to stdout.
    """
    count = int(stats["count"])
    print(f"{indent}{title} ({count} samples)")
    print(
        f"{indent}  mean={stats['mean']*1e3:.3f} ms | std={stats['std']*1e3:.3f} ms | "
        f"min={stats['min']*1e3:.3f} ms | p50={stats['p50']*1e3:.3f} ms | "
        f"p90={stats['p90']*1e3:.3f} ms | max={stats['max']*1e3:.3f} ms"
    )


def _benchmark_single_sample_inference(
    classifier_model: tf.keras.Model,
    los_model: tf.keras.Model,
    nlos_model: tf.keras.Model,
    x_lidar: np.ndarray,
    coords_classifier: np.ndarray,
    coords_los: np.ndarray,
    coords_nlos: np.ndarray,
    repetitions: int,
    sample_index: int,
) -> dict[str, np.ndarray]:
    """Benchmark repeated single-sample inference for each model.

    Args:
        classifier_model (tf.keras.Model): Router classifier model.
        los_model (tf.keras.Model): LOS specialist model.
        nlos_model (tf.keras.Model): NLOS specialist model.
        x_lidar (np.ndarray): LiDAR input tensor for S009.
        coords_classifier (np.ndarray): Coordinates transformed with the
            classifier scaler.
        coords_los (np.ndarray): Coordinates transformed with the LOS scaler.
        coords_nlos (np.ndarray): Coordinates transformed with the NLOS scaler.
        repetitions (int): Number of repeated runs on the same sample.
        sample_index (int): Index of the sample to benchmark.

    Returns:
        dict[str, np.ndarray]: Timing arrays in seconds for keys
        ``classifier``, ``los``, ``nlos``, ``total_models``, and
        ``total_routed``.

    Raises:
        ValueError: If ``repetitions`` is not positive or index is invalid.
    """
    if repetitions <= 0:
        raise ValueError(f"SINGLE_SAMPLE_TIMING_SAMPLES must be > 0, got {repetitions}")
    if sample_index < 0 or sample_index >= int(x_lidar.shape[0]):
        raise ValueError(
            f"SINGLE_SAMPLE_TIMING_INDEX out of bounds: {sample_index} for {int(x_lidar.shape[0])} samples"
        )

    classifier_times: list[float] = []
    los_times: list[float] = []
    nlos_times: list[float] = []
    total_models_times: list[float] = []
    total_routed_times: list[float] = []

    # Keep the exact same single sample for all repetitions to isolate runtime
    # noise from input-data variability.
    lidar_i = x_lidar[sample_index : sample_index + 1]
    coord_classifier_i = coords_classifier[sample_index : sample_index + 1]
    coord_los_i = coords_los[sample_index : sample_index + 1]
    coord_nlos_i = coords_nlos[sample_index : sample_index + 1]

    classifier_inputs = _prepare_inputs(classifier_model, lidar_i, coord_classifier_i)
    los_inputs = _prepare_inputs(los_model, lidar_i, coord_los_i)
    nlos_inputs = _prepare_inputs(nlos_model, lidar_i, coord_nlos_i)

    for _ in range(int(repetitions)):

        classifier_inputs = _prepare_inputs(classifier_model, lidar_i, coord_classifier_i)
        t0_classifier = time.perf_counter()
        classifier_outputs = classifier_model.predict(classifier_inputs, batch_size=1, verbose=0)
        classifier_elapsed_s = time.perf_counter() - t0_classifier
        predicted_label = int(_decode_classifier_labels(np.asarray(classifier_outputs))[0])

        los_elapsed_s = 0.0
        nlos_elapsed_s = 0.0
        t0_los = time.perf_counter()
        _ = los_model.predict(los_inputs, batch_size=1, verbose=0)
        los_elapsed_s = time.perf_counter() - t0_los

        t0_nlos = time.perf_counter()
        _ = nlos_model.predict(nlos_inputs, batch_size=1, verbose=0)
        nlos_elapsed_s = time.perf_counter() - t0_nlos

        classifier_times.append(classifier_elapsed_s)
        los_times.append(los_elapsed_s)
        nlos_times.append(nlos_elapsed_s)
        total_models_times.append(classifier_elapsed_s + los_elapsed_s + nlos_elapsed_s)

        # Routed total keeps the original cascade semantics for the same sample.
        if predicted_label == LABEL_LOS:
            total_routed_times.append(classifier_elapsed_s + los_elapsed_s)
        else:
            total_routed_times.append(classifier_elapsed_s + nlos_elapsed_s)

    return {
        "classifier": np.asarray(classifier_times, dtype=np.float64),
        "los": np.asarray(los_times, dtype=np.float64),
        "nlos": np.asarray(nlos_times, dtype=np.float64),
        "total_models": np.asarray(total_models_times, dtype=np.float64),
        "total_routed": np.asarray(total_routed_times, dtype=np.float64),
    }


def main() -> None:
    """Run routed evaluation over raw S009 and print routing/top-k metrics.

    Returns:
        None: Prints evaluation summary to stdout.

    Raises:
        FileNotFoundError: If required model/scaler artifacts are missing.
        ValueError: If model outputs are inconsistent.
    """
    classifier_model = load_model(CLASSIFIER_MODEL_PATH)
    los_model = load_model(LOS_MODEL_PATH)
    nlos_model = load_model(NLOS_MODEL_PATH)

    classifier_scaler, los_scaler, nlos_scaler = _fit_scalers_like_training()

    if not USE_CUSTOM_LAYERS:
        patch_lambda_globals(classifier_model)
        patch_lambda_globals(los_model)
        patch_lambda_globals(nlos_model)

    x_lidar_s009, x_coord_s009, y_s009, condition_s009 = load_dataset_raw_sparse_labels_s009_only(
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
        remove_null_labels=False,
        return_conditions=True,
    )

    y_s009 = np.asarray(y_s009, dtype=np.int64)
    true_condition = _condition_to_binary(condition_s009)

    # Apply each scaler independently to preserve the exact preprocessing used
    # when each model was trained.
    coords_classifier = classifier_scaler.transform(x_coord_s009).astype(np.float32)
    coords_los = los_scaler.transform(x_coord_s009).astype(np.float32)
    coords_nlos = nlos_scaler.transform(x_coord_s009).astype(np.float32)

    n_samples = int(x_lidar_s009.shape[0])

    classifier_inputs = _prepare_inputs(classifier_model, x_lidar_s009, coords_classifier)
    t0_clf = time.perf_counter()
    classifier_outputs = classifier_model.predict(classifier_inputs, batch_size=BATCH_SIZE, verbose=0)
    classifier_inference_s = time.perf_counter() - t0_clf
    predicted_condition = _decode_classifier_labels(classifier_outputs)

    los_pred_mask = predicted_condition == 0
    nlos_pred_mask = predicted_condition == 1

    routed_predictions = np.zeros(shape=(n_samples, 256), dtype=np.float32)

    los_inference_s = 0.0
    if np.any(los_pred_mask):
        los_inputs = _prepare_inputs(los_model, x_lidar_s009[los_pred_mask], coords_los[los_pred_mask])
        t0_los = time.perf_counter()
        routed_predictions[los_pred_mask] = los_model.predict(los_inputs, batch_size=BATCH_SIZE, verbose=0)
        los_inference_s = time.perf_counter() - t0_los

    nlos_inference_s = 0.0
    if np.any(nlos_pred_mask):
        nlos_inputs = _prepare_inputs(nlos_model, x_lidar_s009[nlos_pred_mask], coords_nlos[nlos_pred_mask])
        t0_nlos = time.perf_counter()
        routed_predictions[nlos_pred_mask] = nlos_model.predict(nlos_inputs, batch_size=BATCH_SIZE, verbose=0)
        nlos_inference_s = time.perf_counter() - t0_nlos

    # Total wall-clock time a sample experiences: classifier + its specialist.
    total_inference_s = classifier_inference_s + los_inference_s + nlos_inference_s

    routed_topk = _topk_stats(routed_predictions, y_s009, TOP_K)

    los_true_mask = true_condition == 0
    nlos_true_mask = true_condition == 1
    los_true_topk = _topk_stats(routed_predictions[los_true_mask], y_s009[los_true_mask], TOP_K)
    nlos_true_topk = _topk_stats(routed_predictions[nlos_true_mask], y_s009[nlos_true_mask], TOP_K)

    routing_accuracy = float(np.mean(predicted_condition == true_condition))
    los_recall = float(np.mean(predicted_condition[los_true_mask] == 0)) if np.any(los_true_mask) else 0.0
    nlos_recall = float(np.mean(predicted_condition[nlos_true_mask] == 1)) if np.any(nlos_true_mask) else 0.0
    routing_balanced_accuracy = 0.5 * (los_recall + nlos_recall)

    print("Routing stage (classifier):")
    print(f"  Total samples: {len(true_condition)}")
    print(f"  Routing accuracy: {routing_accuracy*100:.2f}%")
    print(f"  Routing balanced accuracy: {routing_balanced_accuracy*100:.2f}%")
    print(f"  Predicted LOS: {int(los_pred_mask.sum())}")
    print(f"  Predicted NLOS: {int(nlos_pred_mask.sum())}")
    print()

    print("Routed beam top-k (overall):")
    _print_topk_block("", routed_topk, int(y_s009.shape[0]))
    print(f"  True LOS samples: {int(los_true_mask.sum())}")
    _print_topk_block("LOS ", los_true_topk, int(los_true_mask.sum()))
    print(f"  True NLOS samples: {int(nlos_true_mask.sum())}")
    _print_topk_block("NLOS ", nlos_true_topk, int(nlos_true_mask.sum()))
    print()

    n_los_pred = int(los_pred_mask.sum())
    n_nlos_pred = int(nlos_pred_mask.sum())
    print("Inference time:")
    print(f"  Classifier:      {classifier_inference_s*1e3:8.1f} ms total  |  {classifier_inference_s/n_samples*1e6:7.2f} µs/sample  ({n_samples} samples)")
    if n_los_pred > 0:
        print(f"  LOS specialist:  {los_inference_s*1e3:8.1f} ms total  |  {los_inference_s/n_los_pred*1e6:7.2f} µs/sample  ({n_los_pred} samples)")
    if n_nlos_pred > 0:
        print(f"  NLOS specialist: {nlos_inference_s*1e3:8.1f} ms total  |  {nlos_inference_s/n_nlos_pred*1e6:7.2f} µs/sample  ({n_nlos_pred} samples)")
    print(f"  Total pipeline:  {total_inference_s*1e3:8.1f} ms total  |  {total_inference_s/n_samples*1e6:7.2f} µs/sample  ({n_samples} samples)")

    if SINGLE_SAMPLE_TIMING_SAMPLES is not None:
        single_sample_times = _benchmark_single_sample_inference(
            classifier_model=classifier_model,
            los_model=los_model,
            nlos_model=nlos_model,
            x_lidar=x_lidar_s009,
            coords_classifier=coords_classifier,
            coords_los=coords_los,
            coords_nlos=coords_nlos,
            repetitions=int(SINGLE_SAMPLE_TIMING_SAMPLES),
            sample_index=int(SINGLE_SAMPLE_TIMING_INDEX),
        )

        print()
        print(
            f"Single-sample benchmark (sample index {int(SINGLE_SAMPLE_TIMING_INDEX)}, "
            f"{int(SINGLE_SAMPLE_TIMING_SAMPLES)} repetitions):"
        )
        _print_timing_stats_block("Classifier", _timing_stats(single_sample_times["classifier"]))
        _print_timing_stats_block("LOS model", _timing_stats(single_sample_times["los"]))
        _print_timing_stats_block("NLOS model", _timing_stats(single_sample_times["nlos"]))
        _print_timing_stats_block("Total (classifier + LOS + NLOS)", _timing_stats(single_sample_times["total_models"]))
        _print_timing_stats_block("Total routed pipeline", _timing_stats(single_sample_times["total_routed"]))


if __name__ == "__main__":
    main()
