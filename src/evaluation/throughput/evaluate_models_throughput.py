"""Evaluate saved RayWise models with the shortlist throughput metric."""

import gc
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from compute_throughput_ratio import (
    DEFAULT_K_VALUES,
    DEFAULT_ORACLE_OUTPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    build_s009_oracle_beam_scores_archive,
    compute_model_throughput_result,
    load_oracle_beam_scores_archive,
    save_score_archive,
    save_throughput_result,
    write_throughput_summary_csv,
)
from utils_beam_metrics import (
    expand_restricted_class_scores,
    flatten_beam_grid,
)


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent
BASELINE_S008_ROOT = SRC_ROOT / "data" / "s008"
BASELINE_S009_ROOT = SRC_ROOT / "data" / "s009"
JOINT_TOP1_MODEL_PATH = (
    REPO_ROOT
    / "weights"
    / "nas_best_architectures_cnn1d"
    / "optuna_study"
    / "model"
    / "top_1_trial_4026.keras"
)
DISTILLED_TOP1_MODEL_PATH = (
    REPO_ROOT
    / "weights"
    / "nas_multiobjective_best_knowledge_distillation"
    / "optuna_study"
    / "model"
    / "top_1_trial_721.keras"
)
ENSEMBLE_RESULTS_JSON_PATH = REPO_ROOT / "weights" / "ensemble_cnn1d_stacking" / "results.json"
ENSEMBLE_PREDICTOR_DIR = (
    REPO_ROOT / "weights" / "ensemble_cnn1d_stacking" / "autogluon_meta_20260114_151507"
)
RELATED_WORK_01_MODEL_PATH = (
    REPO_ROOT / "results" / "related_works" / "related_work_01" / "related_work_01.keras"
)
RELATED_WORK_02_MODEL_PATH = (
    REPO_ROOT / "results" / "related_works" / "related_work_02" / "related_work_02.keras"
)
RELATED_WORK_03_MODEL_PATH = (
    REPO_ROOT / "results" / "related_works" / "related_work_03" / "related_work_03.keras"
)
RELATED_WORK_04_MODEL_PATH = (
    REPO_ROOT / "results" / "related_works" / "related_work_04" / "related_work_04.keras"
)
RELATED_WORK_05_MODEL_PATH = (
    REPO_ROOT / "results" / "related_works" / "related_work_05" / "related_work_05.keras"
)
FULL_NUM_CLASSES = 256
MODEL_BATCH_SIZE = 64
STANDARD_MODEL_DATA_SEED = 8128
DEFAULT_EVALUATION_SEEDS = (8128, 8129, 8130)
RESTRICTED_CLASS_FILL_VALUE = -1.0e30


def load_baseline_raywise_arrays(
    *,
    s008_root: Path = BASELINE_S008_ROOT,
    s009_root: Path = BASELINE_S009_ROOT,
) -> dict[str, np.ndarray]:
    """Load the repo-local baseline S008/S009 arrays needed for evaluation.

    Args:
        s008_root (Path): Directory that stores the baseline S008 train/validation
            archives. The loader expects the repository layout under
            ``src/data/s008`` and fails explicitly if any required file is
            missing.
        s009_root (Path): Directory that stores the baseline S009 test archives
            under ``src/data/s009``.

    Returns:
        dict[str, np.ndarray]: Dictionary containing merged S008 coordinates,
        S009 LiDAR inputs, S009 coordinates, S009 sparse labels, and the
        positional S009 sample IDs used throughout the throughput pipeline.

    Raises:
        FileNotFoundError: If one of the required baseline archives is missing.
        ValueError: If any loaded array has an unexpected number of samples.
    """

    s008_coord_train_path = s008_root / "coord_input" / "coord_train.npz"
    s008_coord_val_path = s008_root / "coord_input" / "coord_val.npz"
    s009_coord_path = s009_root / "coord_input" / "coord_input.npz"
    s009_lidar_path = s009_root / "lidar_input" / "lidar_input.npz"
    s008_beam_train_path = s008_root / "beam_output" / "beam_output_train.npz"
    s008_beam_val_path = s008_root / "beam_output" / "beam_output_val.npz"
    s009_beam_path = s009_root / "beam_output" / "beam_output.npz"

    required_paths = [
        s008_coord_train_path,
        s008_coord_val_path,
        s009_coord_path,
        s009_lidar_path,
        s008_beam_train_path,
        s008_beam_val_path,
        s009_beam_path,
    ]
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(f"Required baseline archive not found: {required_path}.")

    with np.load(s008_coord_train_path, allow_pickle=False) as archive:
        s008_coord_train = np.asarray(archive["coordinates"], dtype=np.float32)
    with np.load(s008_coord_val_path, allow_pickle=False) as archive:
        s008_coord_val = np.asarray(archive["coordinates"], dtype=np.float32)
    with np.load(s009_coord_path, allow_pickle=False) as archive:
        s009_coord = np.asarray(archive["coordinates"], dtype=np.float32)
    with np.load(s009_lidar_path, allow_pickle=False) as archive:
        s009_lidar = np.asarray(archive["input"], dtype=np.float32)
    with np.load(s008_beam_train_path, allow_pickle=False) as archive:
        s008_beam_train = np.asarray(archive["output_classification"])
    with np.load(s008_beam_val_path, allow_pickle=False) as archive:
        s008_beam_val = np.asarray(archive["output_classification"])
    with np.load(s009_beam_path, allow_pickle=False) as archive:
        s009_beam = np.asarray(archive["output_classification"])

    s008_coord_all = np.concatenate((s008_coord_train, s008_coord_val), axis=0).astype(np.float32)

    # Re-derive sparse labels directly from the flattened S009 oracle tensor so
    # the throughput pipeline and the classifier metrics use identical indexing.
    s009_labels = np.argmax(flatten_beam_grid(np.asarray(s009_beam.real, dtype=np.float32)), axis=1).astype(np.int64)
    sample_ids = np.arange(s009_labels.shape[0], dtype=np.int64)

    if s009_lidar.shape[0] != s009_coord.shape[0] or s009_lidar.shape[0] != s009_labels.shape[0]:
        raise ValueError(
            "Baseline S009 arrays are misaligned. "
            f"lidar={s009_lidar.shape[0]}, coord={s009_coord.shape[0]}, labels={s009_labels.shape[0]}."
        )

    return {
        "s008_coord_all": s008_coord_all,
        "s009_lidar": s009_lidar,
        "s009_coord": s009_coord,
        "s009_labels": s009_labels,
        "sample_ids": sample_ids,
    }


def split_train_validation_indices(
    *,
    num_samples: int,
    data_seed: int,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the repository's shuffled 80/20 train-validation split.

    Args:
        num_samples (int): Total number of S008 samples.
        data_seed (int): Random seed used by the original training scripts.
        test_ratio (float): Fraction assigned to validation. The current
            repository setup uses ``0.2``. The function mirrors the
            ``train_test_split(..., test_size=0.2, random_state=data_seed,
            shuffle=True)`` behavior by reserving ``ceil(test_ratio * N)``
            shuffled indices for validation and the remainder for training.

    Returns:
        tuple[np.ndarray, np.ndarray]: Training indices followed by validation
        indices, both with dtype ``int64``.

    Raises:
        ValueError: If ``num_samples`` or ``test_ratio`` is invalid.
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, but received {num_samples}.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), but received {test_ratio}.")

    rng = np.random.RandomState(data_seed)
    permutation = rng.permutation(num_samples)
    num_val = int(math.ceil(num_samples * test_ratio))

    # Scikit-learn takes the first block of shuffled indices as validation and
    # keeps the remainder as training when only test_size is specified.
    val_indices = permutation[:num_val].astype(np.int64)
    train_indices = permutation[num_val:].astype(np.int64)
    return train_indices, val_indices


def fit_standard_scaler_from_train_split(
    *,
    coordinates: np.ndarray,
    data_seed: int,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a StandardScaler-equivalent transform on the S008 training split.

    Args:
        coordinates (np.ndarray): Full merged S008 coordinate matrix with shape
            ``(num_samples, 2)``.
        data_seed (int): Random seed used to recreate the original training
            split. Using a different seed changes the learned mean and standard
            deviation and therefore changes the model inputs.
        test_ratio (float): Fraction of shuffled samples assigned to validation.
            The remaining samples define the scaler statistics.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(mean, scale)`` arrays, both with shape
        ``(2,)`` and dtype ``float32``.

    Raises:
        ValueError: If ``coordinates`` is not a 2-D matrix.
    """

    validated_coordinates = np.asarray(coordinates, dtype=np.float32)
    if validated_coordinates.ndim != 2:
        raise ValueError(
            "coordinates must have shape (N, D), but received "
            f"{validated_coordinates.shape}."
        )

    train_indices, _ = split_train_validation_indices(
        num_samples=validated_coordinates.shape[0],
        data_seed=data_seed,
        test_ratio=test_ratio,
    )
    train_coordinates = validated_coordinates[train_indices]

    # StandardScaler uses the population standard deviation. Any zero-variance
    # feature is assigned scale 1.0 so the transform stays well-defined.
    mean = train_coordinates.mean(axis=0, dtype=np.float64)
    scale = train_coordinates.std(axis=0, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def transform_with_standard_scaler(
    coordinates: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply a precomputed standardization transform to coordinates.

    Args:
        coordinates (np.ndarray): Coordinate matrix with shape ``(num_samples, 2)``.
        mean (np.ndarray): Feature-wise mean learned from the S008 training split.
        scale (np.ndarray): Feature-wise population standard deviation. Any zero
            entries should already have been replaced with ``1.0`` by
            :func:`fit_standard_scaler_from_train_split`.

    Returns:
        np.ndarray: Standardized coordinates with dtype ``float32``.

    Raises:
        ValueError: If shapes are inconsistent.
    """

    validated_coordinates = np.asarray(coordinates, dtype=np.float32)
    validated_mean = np.asarray(mean, dtype=np.float32)
    validated_scale = np.asarray(scale, dtype=np.float32)
    if validated_coordinates.ndim != 2:
        raise ValueError(
            "coordinates must have shape (N, D), but received "
            f"{validated_coordinates.shape}."
        )
    if validated_mean.shape != (validated_coordinates.shape[1],):
        raise ValueError(
            "mean shape does not match coordinates. "
            f"Expected {(validated_coordinates.shape[1],)}, received {validated_mean.shape}."
        )
    if validated_scale.shape != validated_mean.shape:
        raise ValueError(
            f"scale shape {validated_scale.shape} does not match mean shape {validated_mean.shape}."
        )
    return ((validated_coordinates - validated_mean) / validated_scale).astype(np.float32)


def _load_tensorflow_components() -> tuple[Any, dict[str, Any]]:
    """Import TensorFlow lazily and build the required custom layer registry.

    Args:
        None: The function only inspects the runtime environment.

    Returns:
        tuple[Any, dict[str, Any]]: Imported TensorFlow module and the
        ``custom_objects`` dictionary needed to deserialize the saved Keras
        checkpoints used by this evaluation pipeline.

    Raises:
        ImportError: If TensorFlow is not installed in the active interpreter.
    """

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to score the saved Keras checkpoints for throughput "
            "evaluation. Activate the project ML environment before calling this function."
        ) from exc

    @tf.keras.utils.register_keras_serializable(package="raywise")
    class LidarToOneHot(tf.keras.layers.Layer):
        """Convert raw LiDAR semantic values into a 4-channel one-hot map."""

        def call(self, inputs: Any) -> Any:
            """Convert one LiDAR batch into semantic occupancy channels.

            Args:
                inputs (Any): Raw LiDAR tensor with shape ``(batch, 20, 200, 10)``
                    and semantic values in ``{-2, -1, 0, 1}``.

            Returns:
                Any: TensorFlow tensor with shape ``(batch, 20, 200, 4)`` and
                float32 dtype.
            """

            base_station = tf.cast(
                tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True),
                tf.float32,
            )
            vehicle = tf.cast(
                tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True),
                tf.float32,
            )
            obstacle = tf.cast(
                tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True),
                tf.float32,
            )
            free_space = tf.cast(
                tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True),
                tf.float32,
            )
            return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)

        def get_config(self) -> dict[str, Any]:
            """Return the serializable Keras configuration.

            Args:
                None: The layer has no constructor arguments to persist.

            Returns:
                dict[str, Any]: Serializable configuration dictionary.
            """

            return super().get_config()

    @tf.keras.utils.register_keras_serializable(package="raywise")
    class CoordTileFlat(tf.keras.layers.Layer):
        """Tile 2-D coordinates across the flattened LiDAR sequence length."""

        def __init__(self, grid_length: int, **kwargs: Any) -> None:
            """Initialize the coordinate tiling layer.

            Args:
                grid_length (int): Number of flattened LiDAR positions. Larger
                    values increase the repeated coordinate sequence length but
                    do not change the per-position feature values.
                **kwargs (Any): Standard Keras layer keyword arguments.

            Returns:
                None: The constructor only stores layer configuration.
            """

            super().__init__(**kwargs)
            self.grid_length = int(grid_length)

        def call(self, inputs: Any) -> Any:
            """Repeat coordinates across the flattened LiDAR sequence.

            Args:
                inputs (Any): Coordinate tensor with shape ``(batch, 2)``.

            Returns:
                Any: TensorFlow tensor with shape ``(batch, grid_length, 2)``.
            """

            return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

        def get_config(self) -> dict[str, Any]:
            """Return the serializable Keras configuration.

            Args:
                None: The layer only serializes its ``grid_length`` attribute.

            Returns:
                dict[str, Any]: Serializable configuration dictionary.
            """

            config = super().get_config()
            config.update({"grid_length": self.grid_length})
            return config

    return tf, {"tf": tf, "LidarToOneHot": LidarToOneHot, "CoordTileFlat": CoordTileFlat}


def patch_lambda_globals(
    *,
    model: Any,
    tf_module: Any,
) -> None:
    """Patch legacy Lambda layers so deserialized callables can resolve globals.

    Args:
        model (Any): Loaded Keras model. The function recursively inspects every
            nested layer and nested submodel.
        tf_module (Any): Imported TensorFlow module. It is injected into Lambda
            globals because older checkpoints reference ``tf`` directly inside
            serialized callables.

    Returns:
        None: The function mutates Lambda callables in-place.
    """

    def walk_layers(container: Any) -> Iterator[Any]:
        """Yield layers recursively from a possibly nested Keras model.

        Args:
            container (Any): Keras model or layer container.

        Returns:
            Sequence[Any]: Recursive layer iterator.
        """

        for layer in container.layers:
            yield layer
            if isinstance(layer, tf_module.keras.Model):
                yield from walk_layers(layer)

    for layer in walk_layers(model):
        if isinstance(layer, tf_module.keras.layers.Lambda):
            function = getattr(layer, "function", None) or getattr(layer, "_function", None)
            if callable(function) and hasattr(function, "__globals__"):
                function.__globals__.setdefault("tf", tf_module)
                function.__globals__.setdefault("np", np)
                function.__globals__.setdefault("K", tf_module.keras.backend)


def load_saved_keras_model(model_path: Path) -> tuple[Any, Any]:
    """Load one saved Keras checkpoint used by the throughput evaluation.

    Args:
        model_path (Path): Path to the ``.keras`` file.

    Returns:
        tuple[Any, Any]: Loaded model and the imported TensorFlow module used to
        deserialize it.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ImportError: If TensorFlow is unavailable.
    """

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}.")

    tf_module, custom_objects = _load_tensorflow_components()
    model = tf_module.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects=custom_objects,
    )
    patch_lambda_globals(model=model, tf_module=tf_module)
    return model, tf_module


def prepare_model_inputs(
    *,
    model: Any,
    lidar: np.ndarray,
    coordinates: np.ndarray,
) -> list[np.ndarray] | np.ndarray:
    """Match loaded arrays to the input order expected by a saved Keras model.

    Args:
        model (Any): Loaded Keras model whose input tensors determine the order.
        lidar (np.ndarray): LiDAR tensor with shape ``(num_samples, 20, 200, 10)``.
        coordinates (np.ndarray): Coordinate matrix with shape ``(num_samples, 2)``.

    Returns:
        list[np.ndarray] | np.ndarray: Input payload ready for
        ``model.predict``. Single-input models receive the bare array while
        multi-input models receive a list in model input order.

    Raises:
        ValueError: If an input tensor cannot be matched unambiguously.
    """

    validated_lidar = np.asarray(lidar, dtype=np.float32)
    validated_coordinates = np.asarray(coordinates, dtype=np.float32)
    lidar_shape = tuple(validated_lidar.shape[1:])
    coordinate_shape = tuple(validated_coordinates.shape[1:])

    def is_compatible(expected_shape: Sequence[Any], actual_shape: tuple[int, ...]) -> bool:
        """Check whether a symbolic Keras input shape matches a concrete array.

        Args:
            expected_shape (Sequence[Any]): Symbolic model input shape excluding
                the batch dimension. Entries may be ``None``.
            actual_shape (tuple[int, ...]): Concrete array shape excluding the
                batch dimension.

        Returns:
            bool: ``True`` when every non-``None`` symbolic dimension matches
            the concrete array.
        """

        if len(expected_shape) != len(actual_shape):
            return False
        for expected_dim, actual_dim in zip(expected_shape, actual_shape):
            if expected_dim is not None and int(expected_dim) != int(actual_dim):
                return False
        return True

    prepared_inputs: list[np.ndarray] = []
    for input_tensor in model.inputs:
        expected_shape = tuple(input_tensor.shape[1:])
        input_name = input_tensor.name.split(":")[0]

        # Prefer semantic name matching when available because some checkpoints
        # expose both inputs with shapes that could otherwise be ambiguous.
        if "lidar" in input_name:
            prepared_inputs.append(validated_lidar)
        elif "coord" in input_name:
            prepared_inputs.append(validated_coordinates)
        elif is_compatible(expected_shape, lidar_shape):
            prepared_inputs.append(validated_lidar)
        elif is_compatible(expected_shape, coordinate_shape):
            prepared_inputs.append(validated_coordinates)
        else:
            raise ValueError(
                f"Could not match model input {input_tensor.name} with expected shape "
                f"{input_tensor.shape}. lidar={validated_lidar.shape}, "
                f"coordinates={validated_coordinates.shape}."
            )

    return prepared_inputs[0] if len(prepared_inputs) == 1 else prepared_inputs


def predict_scores_with_saved_model(
    *,
    model_path: Path,
    lidar: np.ndarray,
    coordinates: np.ndarray,
    batch_size: int = MODEL_BATCH_SIZE,
) -> np.ndarray:
    """Run one saved Keras checkpoint on the provided S009 inputs.

    Args:
        model_path (Path): Path to the ``.keras`` checkpoint.
        lidar (np.ndarray): S009 LiDAR tensor with shape ``(num_samples, 20, 200, 10)``.
        coordinates (np.ndarray): Standardized S009 coordinates with shape
            ``(num_samples, 2)``.
        batch_size (int): Prediction batch size. Larger values reduce Python
            overhead but increase memory pressure. Smaller values reduce peak
            memory usage at the cost of more prediction steps.

    Returns:
        np.ndarray: Model score matrix with shape ``(num_samples, 256)``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ImportError: If TensorFlow is unavailable.
        ValueError: If the model output does not have the expected shape.
    """

    model, tf_module = load_saved_keras_model(model_path)
    try:
        prepared_inputs = prepare_model_inputs(
            model=model,
            lidar=lidar,
            coordinates=coordinates,
        )
        scores = np.asarray(model.predict(prepared_inputs, batch_size=batch_size, verbose=0))
    finally:
        tf_module.keras.backend.clear_session()
        gc.collect()

    if scores.ndim != 2 or scores.shape[1] != FULL_NUM_CLASSES:
        raise ValueError(
            f"Expected {model_path} to output shape (N, {FULL_NUM_CLASSES}), "
            f"but received {scores.shape}."
        )
    return scores.astype(np.float32)


def stack_meta_features(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate multiple base-model score matrices for stacking inference.

    Args:
        predictions (Sequence[np.ndarray]): Sequence of base-model score matrices.
            Each matrix must have the same number of samples and the same class
            dimension.

    Returns:
        np.ndarray: Concatenated meta-feature matrix with shape
        ``(num_samples, num_models * num_classes)``.

    Raises:
        ValueError: If the prediction matrices disagree on sample or class count.
    """

    if not predictions:
        raise ValueError("predictions must contain at least one score matrix.")

    validated_predictions = [np.asarray(prediction, dtype=np.float32) for prediction in predictions]
    reference_shape = validated_predictions[0].shape
    for prediction in validated_predictions:
        if prediction.shape != reference_shape:
            raise ValueError(
                "All prediction matrices must share the same shape. "
                f"Expected {reference_shape}, received {prediction.shape}."
            )
    return np.concatenate(validated_predictions, axis=1).astype(np.float32)


def align_meta_features(
    *,
    meta_features: np.ndarray,
    class_ids: np.ndarray,
    num_models: int,
) -> np.ndarray:
    """Slice stacked base-model probabilities to the ensemble's class subset.

    Args:
        meta_features (np.ndarray): Concatenated base-model score matrix with
            shape ``(num_samples, num_models * 256)``.
        class_ids (np.ndarray): Beam IDs kept by the stacked ensemble.
        num_models (int): Number of base-model blocks in ``meta_features``.

    Returns:
        np.ndarray: Restricted meta-feature matrix with shape
        ``(num_samples, num_models * len(class_ids))``.

    Raises:
        ValueError: If shapes are inconsistent.
    """

    validated_meta_features = np.asarray(meta_features, dtype=np.float32)
    validated_class_ids = np.asarray(class_ids, dtype=np.int64)
    if validated_meta_features.ndim != 2:
        raise ValueError(
            "meta_features must have shape (N, M*C), but received "
            f"{validated_meta_features.shape}."
        )
    if num_models <= 0:
        raise ValueError(f"num_models must be positive, but received {num_models}.")
    if validated_meta_features.shape[1] % num_models != 0:
        raise ValueError(
            "meta_features column count must be divisible by num_models. "
            f"Received {validated_meta_features.shape[1]} and {num_models}."
        )

    base_num_classes = validated_meta_features.shape[1] // num_models
    if base_num_classes != FULL_NUM_CLASSES:
        raise ValueError(
            f"Expected base model blocks of size {FULL_NUM_CLASSES}, but received {base_num_classes}."
        )

    aligned_blocks: list[np.ndarray] = []
    for model_index in range(num_models):
        start = model_index * base_num_classes
        end = start + base_num_classes
        aligned_blocks.append(validated_meta_features[:, start:end][:, validated_class_ids])
    return np.concatenate(aligned_blocks, axis=1).astype(np.float32)


def meta_features_to_dataframe(
    *,
    meta_features: np.ndarray,
    class_ids: np.ndarray,
    num_models: int,
) -> Any:
    """Convert aligned meta-features into the column layout expected by AutoGluon.

    Args:
        meta_features (np.ndarray): Restricted meta-feature matrix with shape
            ``(num_samples, num_models * len(class_ids))``.
        class_ids (np.ndarray): Original beam IDs kept by the stacker. These are
            embedded in the column names so the saved predictor sees the same
            feature schema it was trained on.
        num_models (int): Number of base models represented in the feature
            matrix.

    Returns:
        Any: ``pandas.DataFrame`` containing one column per
        ``(model_index, original_class_id)`` pair.

    Raises:
        ImportError: If pandas is unavailable.
        ValueError: If the feature matrix shape is inconsistent.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to build AutoGluon meta-feature tables for the stacked ensemble."
        ) from exc

    validated_meta_features = np.asarray(meta_features, dtype=np.float32)
    validated_class_ids = np.asarray(class_ids, dtype=np.int64)
    expected_num_columns = num_models * validated_class_ids.shape[0]
    if validated_meta_features.shape[1] != expected_num_columns:
        raise ValueError(
            f"Expected {expected_num_columns} meta-feature columns, received "
            f"{validated_meta_features.shape[1]}."
        )

    column_names = [
        f"m{model_index + 1}_class_{int(class_id)}"
        for model_index in range(num_models)
        for class_id in validated_class_ids
    ]
    return pd.DataFrame(validated_meta_features, columns=column_names)


def extract_probability_matrix(
    *,
    probability_frame: Any,
    class_order: Sequence[int],
) -> np.ndarray:
    """Extract ensemble probabilities in a deterministic encoded-class order.

    Args:
        probability_frame (Any): ``pandas.DataFrame`` returned by
            ``TabularPredictor.predict_proba``. Columns may be encoded class IDs
            as integers or strings depending on AutoGluon's serialization.
        class_order (Sequence[int]): Encoded class IDs in the exact order needed
            for downstream expansion back to the original 256-beam space.

    Returns:
        np.ndarray: Probability matrix ordered by ``class_order``.

    Raises:
        KeyError: If the DataFrame columns cannot be aligned with ``class_order``.
    """

    integer_columns = list(class_order)
    if all(column in probability_frame.columns for column in integer_columns):
        return probability_frame[integer_columns].to_numpy(dtype=np.float32)

    string_columns = [str(int(column)) for column in class_order]
    if all(column in probability_frame.columns for column in string_columns):
        return probability_frame[string_columns].to_numpy(dtype=np.float32)

    raise KeyError(
        "Could not align AutoGluon probability columns with the expected class order. "
        f"Missing integer columns {integer_columns} and string columns {string_columns}."
    )


def score_joint_top1_model(
    *,
    baseline_arrays: dict[str, np.ndarray],
    data_seed: int = STANDARD_MODEL_DATA_SEED,
    batch_size: int = MODEL_BATCH_SIZE,
) -> np.ndarray:
    """Score the saved top-1 joint CNN using the repo-local baseline arrays.

    Args:
        baseline_arrays (dict[str, np.ndarray]): Payload returned by
            :func:`load_baseline_raywise_arrays`.
        data_seed (int): Seed used to recreate the S008 training split and fit
            the coordinate standardization transform. Different seeds change the
            preprocessing split and therefore the final score matrix, but they
            do not change the loaded model weights.
        batch_size (int): Prediction batch size for TensorFlow inference.

    Returns:
        np.ndarray: Score matrix with shape ``(9638, 256)``.
    """

    mean, scale = fit_standard_scaler_from_train_split(
        coordinates=baseline_arrays["s008_coord_all"],
        data_seed=data_seed,
    )
    scaled_s009_coord = transform_with_standard_scaler(
        baseline_arrays["s009_coord"],
        mean=mean,
        scale=scale,
    )
    return predict_scores_with_saved_model(
        model_path=JOINT_TOP1_MODEL_PATH,
        lidar=baseline_arrays["s009_lidar"],
        coordinates=scaled_s009_coord,
        batch_size=batch_size,
    )


def score_distilled_top1_model(
    *,
    baseline_arrays: dict[str, np.ndarray],
    data_seed: int = STANDARD_MODEL_DATA_SEED,
    batch_size: int = MODEL_BATCH_SIZE,
) -> np.ndarray:
    """Score the saved top-1 distilled CNN using the baseline S008 scaler split.

    Args:
        baseline_arrays (dict[str, np.ndarray]): Payload returned by
            :func:`load_baseline_raywise_arrays`.
        data_seed (int): Seed used to recreate the S008 training split for the
            standardization transform. Different seeds change only the input
            preprocessing, not the distilled model checkpoint.
        batch_size (int): Prediction batch size for TensorFlow inference.

    Returns:
        np.ndarray: Score matrix with shape ``(9638, 256)``.
    """

    mean, scale = fit_standard_scaler_from_train_split(
        coordinates=baseline_arrays["s008_coord_all"],
        data_seed=data_seed,
    )
    scaled_s009_coord = transform_with_standard_scaler(
        baseline_arrays["s009_coord"],
        mean=mean,
        scale=scale,
    )
    return predict_scores_with_saved_model(
        model_path=DISTILLED_TOP1_MODEL_PATH,
        lidar=baseline_arrays["s009_lidar"],
        coordinates=scaled_s009_coord,
        batch_size=batch_size,
    )


def score_related_work_model(
    *,
    model_path: Path,
    baseline_arrays: dict[str, np.ndarray],
    data_seed: int = STANDARD_MODEL_DATA_SEED,
    batch_size: int = MODEL_BATCH_SIZE,
) -> np.ndarray:
    """Score one related-works model checkpoint on the S009 throughput split.

    Args:
        model_path (Path): Path to a saved related-works ``.keras`` checkpoint.
        baseline_arrays (dict[str, np.ndarray]): Payload returned by
            :func:`load_baseline_raywise_arrays`.
        data_seed (int): Seed used to recreate the S008 training split for the
            standardization transform before S009 inference.
        batch_size (int): Prediction batch size for TensorFlow inference.

    Returns:
        np.ndarray: Score matrix with shape ``(9638, 256)``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ImportError: If TensorFlow is unavailable.
        ValueError: If the model output shape is inconsistent.
    """

    mean, scale = fit_standard_scaler_from_train_split(
        coordinates=baseline_arrays["s008_coord_all"],
        data_seed=data_seed,
    )
    scaled_s009_coord = transform_with_standard_scaler(
        baseline_arrays["s009_coord"],
        mean=mean,
        scale=scale,
    )
    return predict_scores_with_saved_model(
        model_path=model_path,
        lidar=baseline_arrays["s009_lidar"],
        coordinates=scaled_s009_coord,
        batch_size=batch_size,
    )


def score_stacking_ensemble(
    *,
    baseline_arrays: dict[str, np.ndarray],
    data_seed: int | None = None,
    batch_size: int = MODEL_BATCH_SIZE,
    ensemble_results_path: Path = ENSEMBLE_RESULTS_JSON_PATH,
    predictor_dir: Path = ENSEMBLE_PREDICTOR_DIR,
) -> np.ndarray:
    """Score the saved stacking ensemble and expand it to the full 256-beam space.

    Args:
        baseline_arrays (dict[str, np.ndarray]): Payload returned by
            :func:`load_baseline_raywise_arrays`.
        data_seed (int | None): Seed used to fit the coordinate standardization
            transform. When ``None``, the seed stored in the ensemble metadata
            is used. Passing explicit seeds lets the evaluation run repeatedly
            and average the resulting metrics while keeping the rest of the
            ensemble inference path unchanged.
        batch_size (int): Prediction batch size for the three base Keras models.
        ensemble_results_path (Path): JSON metadata saved with the stacked
            ensemble. The file provides the base model paths, the 122 retained
            original beam IDs, and the data seed used when fitting the ensemble.
        predictor_dir (Path): Directory that stores the saved AutoGluon
            predictor.

    Returns:
        np.ndarray: Expanded score matrix with shape ``(9638, 256)``. The 122
        modeled beam IDs receive ensemble probabilities; omitted beam IDs are
        set to a large negative finite score so they never enter a predicted
        shortlist while still satisfying the shared finite-value validation.

    Raises:
        FileNotFoundError: If the ensemble metadata or predictor directory is
            missing.
        ImportError: If TensorFlow, pandas, or AutoGluon is unavailable.
        ValueError: If the saved metadata is inconsistent.
    """

    if not ensemble_results_path.is_file():
        raise FileNotFoundError(f"Ensemble metadata not found: {ensemble_results_path}.")
    if not predictor_dir.is_dir():
        raise FileNotFoundError(f"Ensemble predictor directory not found: {predictor_dir}.")

    with ensemble_results_path.open("r", encoding="utf-8") as file:
        ensemble_results = json.load(file)

    base_model_paths = [Path(path) for path in ensemble_results["config"]["model_paths"]]
    class_ids = np.asarray(ensemble_results["classes"], dtype=np.int64)
    ensemble_seed = int(ensemble_results["config"]["data_seed"])
    effective_seed = ensemble_seed if data_seed is None else int(data_seed)

    mean, scale = fit_standard_scaler_from_train_split(
        coordinates=baseline_arrays["s008_coord_all"],
        data_seed=effective_seed,
    )
    scaled_s009_coord = transform_with_standard_scaler(
        baseline_arrays["s009_coord"],
        mean=mean,
        scale=scale,
    )

    base_predictions = [
        predict_scores_with_saved_model(
            model_path=model_path,
            lidar=baseline_arrays["s009_lidar"],
            coordinates=scaled_s009_coord,
            batch_size=batch_size,
        )
        for model_path in base_model_paths
    ]

    aligned_meta_features = align_meta_features(
        meta_features=stack_meta_features(base_predictions),
        class_ids=class_ids,
        num_models=len(base_model_paths),
    )
    ensemble_features = meta_features_to_dataframe(
        meta_features=aligned_meta_features,
        class_ids=class_ids,
        num_models=len(base_model_paths),
    )

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required to score the saved stacking ensemble. "
            "Activate the project ML environment before calling this function."
        ) from exc

    predictor = TabularPredictor.load(str(predictor_dir))
    restricted_probabilities = extract_probability_matrix(
        probability_frame=predictor.predict_proba(ensemble_features),
        class_order=list(range(class_ids.shape[0])),
    )
    return expand_restricted_class_scores(
        restricted_probabilities,
        class_ids,
        full_num_classes=FULL_NUM_CLASSES,
        fill_value=RESTRICTED_CLASS_FILL_VALUE,
    ).astype(np.float32)


def average_throughput_results(
    *,
    model_name: str,
    results: Sequence[dict[str, Any]],
    evaluation_seeds: Sequence[int],
) -> dict[str, Any]:
    """Average repeated throughput result payloads across evaluation seeds.

    Args:
        model_name (str): Stable model identifier shared by all repeated runs.
        results (Sequence[dict[str, Any]]): Result payloads produced by
            :func:`compute_model_throughput_result` for the same model but with
            different seeds. Every payload must share the same ``k_values`` and
            ``num_samples`` values.
        evaluation_seeds (Sequence[int]): Seeds used for the repeated runs.
            The returned payload stores this list verbatim so the averaged
            metrics remain traceable.

    Returns:
        dict[str, Any]: Averaged throughput result payload with the same schema
        as :func:`compute_model_throughput_result`, plus an
        ``evaluation_seeds`` field.

    Raises:
        ValueError: If ``results`` is empty or if any repeated run disagrees on
            the metric layout.
    """

    if not results:
        raise ValueError(f"No throughput results available to average for {model_name}.")

    reference_result = results[0]
    reference_k_values = list(reference_result["k_values"])
    reference_num_samples = int(reference_result["num_samples"])
    for result in results[1:]:
        if list(result["k_values"]) != reference_k_values:
            raise ValueError(f"Repeated runs for {model_name} do not share the same k_values.")
        if int(result["num_samples"]) != reference_num_samples:
            raise ValueError(f"Repeated runs for {model_name} do not share the same sample count.")

    throughput_ratio = np.mean(
        np.asarray([result["throughput_ratio"] for result in results], dtype=np.float64),
        axis=0,
    )
    topk_accuracy = np.mean(
        np.asarray([result["topk_accuracy"] for result in results], dtype=np.float64),
        axis=0,
    )

    return {
        "model_name": model_name,
        "k_values": reference_k_values,
        "throughput_ratio": throughput_ratio.astype(np.float64).tolist(),
        "topk_accuracy": topk_accuracy.astype(np.float64).tolist(),
        "num_samples": reference_num_samples,
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "num_seeds": len(results),
    }


def evaluate_saved_models_throughput(
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    evaluation_seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    include_related_works: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    oracle_output_path: Path = DEFAULT_ORACLE_OUTPUT_PATH,
) -> dict[str, dict[str, Any]]:
    """Evaluate the saved top-1 models and stacked ensemble on S009 throughput.

    Args:
        k_values (Sequence[int]): Shortlist sizes to evaluate. The default
            ``(1, 3, 5, 10, 20, 30, 40, 50)`` matches the current evaluation
            setup. Larger values are allowed as long as they stay within the
            256-beam codebook.
        evaluation_seeds (Sequence[int]): Seeds used to repeat the full
            evaluation. Each seed re-fits the coordinate standardization split,
            and the final throughput metrics are averaged across runs so the
            reported numbers are less sensitive to one particular split.
        include_related_works (bool): Whether to evaluate the related-works
            checkpoints (related_work_01 to related_work_05). Keeping this
            disabled by default preserves the original runtime and avoids
            requiring those extra checkpoint artifacts unless explicitly needed.
        output_dir (Path): Directory where oracle archives, per-model score
            matrices, JSON results, and the summary CSV are written.
        oracle_output_path (Path): Destination path for the flattened oracle S009
            beam archive. If the file does not exist, it is created first.

    Returns:
        dict[str, dict[str, Any]]: Mapping from model name to the saved
        throughput result payload.

    Raises:
        FileNotFoundError: If required datasets or model artifacts are missing.
        ImportError: If TensorFlow, pandas, or AutoGluon is unavailable when a
            model requires them.
        ValueError: If sample alignment or score-shape checks fail.

    Examples:
        >>> # Requires the project ML environment with TensorFlow and AutoGluon.
        >>> # results = evaluate_saved_models_throughput()
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    validated_evaluation_seeds = tuple(int(seed) for seed in evaluation_seeds)
    if not validated_evaluation_seeds:
        raise ValueError("evaluation_seeds must contain at least one seed.")
    if len(set(validated_evaluation_seeds)) != len(validated_evaluation_seeds):
        raise ValueError("evaluation_seeds must contain unique seed values.")

    if oracle_output_path.is_file():
        oracle_bundle = load_oracle_beam_scores_archive(oracle_output_path)
    else:
        oracle_bundle = build_s009_oracle_beam_scores_archive(output_path=oracle_output_path)

    baseline_arrays = load_baseline_raywise_arrays()
    if not np.array_equal(baseline_arrays["sample_ids"], oracle_bundle["sample_ids"]):
        raise ValueError("Baseline S009 sample IDs do not match the oracle archive sample IDs.")
    if not np.array_equal(baseline_arrays["s009_labels"], oracle_bundle["best_idx"]):
        raise ValueError("Baseline S009 sparse labels do not match the oracle archive labels.")

    score_builders = {
        "joint_1dcnn_top1": score_joint_top1_model,
        "distilled_1dcnn_motpe_top1": score_distilled_top1_model,
        "ensemble_cnn1d_stacking": score_stacking_ensemble,
    }

    if include_related_works:
        score_builders.update(
            {
                "related_work_01": lambda **kwargs: score_related_work_model(
                    model_path=RELATED_WORK_01_MODEL_PATH,
                    **kwargs,
                ),
                "related_work_02": lambda **kwargs: score_related_work_model(
                    model_path=RELATED_WORK_02_MODEL_PATH,
                    **kwargs,
                ),
                "related_work_03": lambda **kwargs: score_related_work_model(
                    model_path=RELATED_WORK_03_MODEL_PATH,
                    **kwargs,
                ),
                "related_work_04": lambda **kwargs: score_related_work_model(
                    model_path=RELATED_WORK_04_MODEL_PATH,
                    **kwargs,
                ),
                "related_work_05": lambda **kwargs: score_related_work_model(
                    model_path=RELATED_WORK_05_MODEL_PATH,
                    **kwargs,
                ),
            }
        )

    saved_results: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for model_name, score_builder in score_builders.items():
        repeated_results: list[dict[str, Any]] = []
        for evaluation_seed in validated_evaluation_seeds:
            scores = score_builder(
                baseline_arrays=baseline_arrays,
                data_seed=evaluation_seed,
            )
            sample_ids = baseline_arrays["sample_ids"]

            save_score_archive(
                output_path=output_dir / f"{model_name}_seed_{evaluation_seed}_scores_s009.npz",
                scores=scores,
                sample_ids=sample_ids,
            )
            repeated_result = compute_model_throughput_result(
                model_name=model_name,
                oracle_scores=oracle_bundle["psi_all"],
                predicted_scores=scores,
                k_values=k_values,
                sample_ids=oracle_bundle["sample_ids"],
                predicted_sample_ids=sample_ids,
            )
            save_throughput_result(
                output_path=output_dir / f"{model_name}_seed_{evaluation_seed}_throughput.json",
                result=repeated_result,
            )
            repeated_results.append(repeated_result)

        result = average_throughput_results(
            model_name=model_name,
            results=repeated_results,
            evaluation_seeds=validated_evaluation_seeds,
        )
        save_throughput_result(
            output_path=output_dir / f"{model_name}_throughput.json",
            result=result,
        )
        saved_results[model_name] = result
        result_rows.append(result)

    write_throughput_summary_csv(
        output_path=output_dir / "throughput_summary.csv",
        results=result_rows,
    )
    return saved_results


if __name__ == "__main__":
    evaluate_saved_models_throughput()
