"""Load Raymobtime raw datasets, split S008, and prepare ensemble predictions."""

import os

# Specify GPU to use (e.g., GPU:0, CPU:-1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

import json
import re
import pickle
from datetime import datetime
from pathlib import Path
from typing import Sequence
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from _load_dataset_raw import (
    load_dataset_raw_sparse_labels_s008_only,
    load_dataset_raw_sparse_labels_s009_only,
)

# ———————————————————————————————————————————————————————————————————————————— #
#                              Global Configuration                             #
# ———————————————————————————————————————————————————————————————————————————— #
TOP_K = (1, 3, 5, 10, 20, 30, 50)

DATA_SEED = 0
TRAIN_RATIO = 0.8

# AutoGluon training knobs for the full ensemble run.
AG_PRESETS = "medium_quality_faster_train"
AG_NUM_STACK_LEVELS = 1
AG_NUM_BAG_FOLDS = 5
AG_NUM_BAG_SETS = 1
AG_USE_BAG_HOLDOUT = True
AG_FIT_STRATEGY = "sequential"
AG_FOLD_FITTING_STRATEGY = "sequential_local"
AG_NUM_FOLDS_PARALLEL = 1
AG_NUM_CPUS = 12
AG_VERBOSITY = 2

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")

S008_COORD_CSV = DATA_ROOT / "Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
S008_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s008/processed_raw_data/lidar_data_s008"
S008_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s008/baseline_data/beam_output/beams_output_s008.npz"

S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

DEFAULT_MODEL_DIR = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_v7_seeds_val_acc_topk_1/optuna_study/model"
)
DEFAULT_TOP_X_MODELS = 10
RESULTS_DIR = Path("runs") / Path(__file__).stem


@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(tf.keras.layers.Layer):
    """Convert raw LiDAR tensors into a 4-channel semantic one-hot map.

    The input uses semantic values encoded in ``{-2, -1, 0, 1}``. This layer
    expands those semantics into explicit channels so downstream convolutions
    can consume a stable representation independent of raw code values.

    Returns:
        tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)`` where the
        channels represent base station, vehicle, obstacle, and free space.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic one-hot encoding to the LiDAR grid.

        Args:
            inputs (tf.Tensor): Raw LiDAR tensor with shape
                ``(batch, 20, 200, 10)`` and semantic values in
                ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Encoded float32 tensor with shape ``(batch, 20, 200, 4)``.
        """
        # Build binary masks for each semantic class used by pretrained models.
        # Keeping this exact mapping preserves compatibility with saved weights.
        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(tf.keras.layers.Layer):
    """Tile 2D coordinates across a flattened spatial sequence.

    Args:
        grid_length (int): Number of spatial positions after flattening the
            LiDAR grid (for example, ``20 * 200``).
    """

    def __init__(self, grid_length: int, **kwargs) -> None:
        """Store the flattened grid length required for coordinate tiling.

        Args:
            grid_length (int): Number of sequence positions produced by the
                LiDAR flattening stage.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.
        """
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Repeat each coordinate pair along the flattened LiDAR axis.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled coordinate tensor with shape
            ``(batch, grid_length, 2)``.
        """
        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict:
        """Return serializable layer configuration.

        Returns:
            dict: Layer configuration including ``grid_length``.
        """
        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


def _prepare_inputs(model: tf.keras.Model, lidar: np.ndarray, coords: np.ndarray):
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
                f"lidar={lidar.shape}, coords={coords.shape}"
            )
        prepared_by_name[input_name] = candidate

    ordered_inputs = [prepared_by_name[t.name.split(":")[0]] for t in model.inputs]
    if len(ordered_inputs) == 1:
        return ordered_inputs[0]
    return ordered_inputs


def resolve_top_model_paths(model_dir: Path, top_x: int) -> list[Path]:
    """Resolve the best `top_x` model files from a model directory.

    This function expects model files named with the convention
    `top_<rank>_... .keras` (for example, `top_1_trial_1116.keras`).
    It parses `<rank>`, sorts models by ascending rank, and returns the
    first `top_x` entries.

    Args:
        model_dir (Path): Directory containing exported Keras model files.
            If this directory does not exist or is not a directory, model
            discovery fails.
        top_x (int): Number of top-ranked models to load. If greater than the
            number of discovered ranked models, the function raises an error.

    Returns:
        list[Path]: Ordered model paths from rank 1 upward, truncated to
        exactly `top_x` elements.

    Raises:
        ValueError: If `top_x` is less than 1.
        FileNotFoundError: If `model_dir` does not exist.
        NotADirectoryError: If `model_dir` is not a directory.
        RuntimeError: If no ranked model files are found or fewer than
            `top_x` ranked model files are available.

    Examples:
        >>> resolve_top_model_paths(Path("/tmp/models"), 2)
        [PosixPath('/tmp/models/top_1_trial_1116.keras'), PosixPath('/tmp/models/top_2_trial_690.keras')]
    """
    if top_x < 1:
        raise ValueError(f"top_x must be >= 1, got {top_x}.")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory path is not a directory: {model_dir}")

    # Select only files that follow the top-rank naming scheme so ordering is explicit.
    # This avoids accidentally mixing in unrelated checkpoints from the same folder.
    ranked_models: list[tuple[int, Path]] = []
    pattern = re.compile(r"^top_(\d+)_.*\.keras$")
    for path in model_dir.glob("*.keras"):
        match = pattern.match(path.name)
        if match:
            ranked_models.append((int(match.group(1)), path))

    if not ranked_models:
        raise RuntimeError(
            "No ranked model files found. Expected files like 'top_<rank>_... .keras' "
            f"in {model_dir}."
        )

    ranked_models.sort(key=lambda item: item[0])
    selected_paths = [path for _, path in ranked_models[:top_x]]
    if len(selected_paths) < top_x:
        raise RuntimeError(
            f"Requested top_x={top_x}, but only found {len(selected_paths)} ranked model file(s) "
            f"in {model_dir}."
        )
    return selected_paths


def resolve_model_selection(
    model_dir: Path | None = None,
    top_x: int | None = None,
) -> tuple[Path, int]:
    """Resolve model folder and number of top models from args/env/defaults.

    Resolution precedence is:
    1) explicit function arguments,
    2) environment variables,
    3) module defaults.

    Environment variables:
    - `RAYWISE_MODEL_DIR`: Overrides default model directory when `model_dir`
      argument is not provided.
    - `RAYWISE_TOP_X_MODELS`: Overrides default number of top models when
      `top_x` argument is not provided.

    Args:
        model_dir (Path | None, optional): Explicit model directory. If set,
            this value has highest precedence and environment/default values
            are ignored for directory selection.
        top_x (int | None, optional): Explicit number of ranked models to use.
            If set, this value has highest precedence and environment/default
            values are ignored for top-X selection.

    Returns:
        tuple[Path, int]: A tuple containing resolved `(model_dir, top_x)`.

    Raises:
        ValueError: If `RAYWISE_TOP_X_MODELS` cannot be parsed as an integer.
        ValueError: If the resolved `top_x` is lower than 1.

    Examples:
        >>> resolve_model_selection()
        (PosixPath('/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model'), 3)
    """
    env_model_dir = os.environ.get("RAYWISE_MODEL_DIR")
    env_top_x = os.environ.get("RAYWISE_TOP_X_MODELS")

    resolved_model_dir = model_dir or (Path(env_model_dir) if env_model_dir else DEFAULT_MODEL_DIR)

    if top_x is not None:
        resolved_top_x = top_x
    elif env_top_x is not None:
        try:
            resolved_top_x = int(env_top_x)
        except ValueError as exc:
            raise ValueError(
                f"Invalid RAYWISE_TOP_X_MODELS value: {env_top_x!r}. Expected an integer >= 1."
            ) from exc
    else:
        resolved_top_x = DEFAULT_TOP_X_MODELS

    if resolved_top_x < 1:
        raise ValueError(f"top_x must be >= 1, got {resolved_top_x}.")

    return resolved_model_dir, resolved_top_x


def predict_with_topk_models(
    models: Sequence[tf.keras.Model],
    lidar_inputs: np.ndarray,
    coord_inputs: np.ndarray,
    coord_scaler: StandardScaler,
    *,
    batch_size: int = 64,
) -> list[np.ndarray]:
    """Predict on the provided data with an arbitrary number of base models.

    Args:
        models (Sequence[tf.keras.Model]): Sequence of loaded Keras models.
            Each model must accept either lidar features, coordinate features,
            or both, as resolved by `_prepare_inputs`.
        lidar_inputs (np.ndarray): Lidar inputs for inference.
        coord_inputs (np.ndarray): Coordinate inputs for inference.
        coord_scaler (StandardScaler): Scaler fit on training coordinates.
            Coordinates are transformed before prediction to match base-model
            training preprocessing.
        batch_size (int, optional): Inference batch size. Larger values can
            improve throughput at the cost of memory usage. Defaults to 64.

    Returns:
        list[np.ndarray]: One prediction matrix per model, preserving input
        model order.

    Raises:
        ValueError: If `models` is empty.
    """
    if len(models) == 0:
        raise ValueError("predict_with_topk_models expects at least one model.")

    lidar = np.asarray(lidar_inputs, dtype=np.float32)
    coords = np.asarray(coord_inputs, dtype=np.float32)
    scaled_coords = coord_scaler.transform(coords)

    outputs: list[np.ndarray] = []
    for model in models:
        inputs = _prepare_inputs(model, lidar, scaled_coords)
        outputs.append(model.predict(inputs, batch_size=batch_size, verbose=0))
    return outputs


def compute_topk_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = TOP_K,
) -> dict[int, float]:
    """Return Top-K accuracy for the provided predictions and labels."""
    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions shape (N, C), got {predictions.shape}")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1-D array")

    total = labels.shape[0]
    if total == 0:
        return {k: 0.0 for k in ks}

    sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
    accuracies: dict[int, float] = {}
    for k in ks:
        k_eff = min(k, sorted_indices.shape[1])
        hits = (sorted_indices[:, :k_eff] == labels[:, None]).any(axis=1)
        accuracies[k] = hits.sum() / total
    return accuracies


def _filter_rare_classes(
    lidar: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop entries whose class appears fewer than `min_samples` times."""
    counts = np.bincount(labels)
    keep_classes = np.where(counts >= min_samples)[0]
    mask = np.isin(labels, keep_classes)
    removed = np.setdiff1d(np.unique(labels), keep_classes)
    if removed.size:
        preview = ", ".join(str(int(v)) for v in removed[:10])
        suffix = "..." if removed.size > 10 else ""
        print(
            f"Dropping {removed.size} rare class(es) with < {min_samples} sample(s): "
            f"{preview}{suffix}"
        )
    return lidar[mask], coords[mask], labels[mask]


def _maybe_stratify(labels: np.ndarray, min_count: int = 2) -> np.ndarray | None:
    """Return labels for stratification if every class has at least `min_count` samples."""
    counts = np.bincount(labels)
    positive = counts[counts > 0]
    if positive.size == 0:
        return None
    min_positive = int(positive.min())
    if min_positive < min_count:
        print(
            f"Warning: cannot stratify train/val split because the rarest class has {min_positive} sample(s); "
            "falling back to a random split."
        )
        return None
    return labels


def stack_meta_features(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate per-model softmax outputs into a 2-D meta feature matrix."""
    if not predictions:
        return np.empty((0, 0), dtype=np.float32)

    base_shape = predictions[0].shape
    feature_list: list[np.ndarray] = []
    for preds in predictions:
        preds = np.asarray(preds, dtype=np.float32)
        if preds.shape[0] != base_shape[0]:
            raise ValueError("All prediction arrays must have the same number of samples.")
        feature_list.append(preds)

    return np.concatenate(feature_list, axis=1)


def meta_features_to_dataframe(
    meta_features: np.ndarray,
    *,
    labels: np.ndarray | None,
    classes: np.ndarray,
    label_column: str = "label",
    expected_models: int,
) -> pd.DataFrame:
    """Return a DataFrame with per-model/per-class probability columns."""
    meta = np.asarray(meta_features, dtype=np.float32)
    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("classes must be non-empty for meta feature DataFrame creation.")
    if meta.shape[1] % num_classes != 0:
        raise ValueError(
            f"Meta feature columns ({meta.shape[1]}) are not divisible by number of classes ({num_classes})."
        )

    inferred_models = meta.shape[1] // num_classes
    if inferred_models != expected_models:
        raise ValueError(
            f"Expected meta features from {expected_models} model(s) but inferred {inferred_models} "
            f"from shape {meta.shape} and {num_classes} classes."
        )

    feature_names = [
        f"m{m_idx + 1}_class_{int(cls)}"
        for m_idx in range(expected_models)
        for cls in classes
    ]
    df = pd.DataFrame(meta, columns=feature_names)
    if labels is not None:
        df[label_column] = labels
    return df


def align_meta_features(
    meta_features: np.ndarray,
    *,
    classes: np.ndarray,
    expected_models: int,
) -> np.ndarray:
    """Slice meta features to only include probabilities for the provided classes."""
    meta = np.asarray(meta_features, dtype=np.float32)
    if meta.shape[0] == 0:
        return meta

    total_cols = meta.shape[1]
    if total_cols % expected_models != 0:
        raise ValueError(
            f"Meta feature columns ({total_cols}) are not divisible by expected_models ({expected_models})."
        )

    base_classes = total_cols // expected_models
    if np.max(classes, initial=-1) >= base_classes:
        raise ValueError(
            f"Found class id outside base model output range. Max class id: {np.max(classes)}, base outputs: {base_classes}."
        )

    slices: list[np.ndarray] = []
    for model_idx in range(expected_models):
        start = model_idx * base_classes
        end = start + base_classes
        model_block = meta[:, start:end]
        slices.append(model_block[:, classes])

    return np.concatenate(slices, axis=1)


def extract_prob_matrix(proba: pd.DataFrame, class_order: Sequence[int]) -> np.ndarray:
    """Return probabilities in the desired class order, handling int or str column names."""
    int_cols = list(class_order)
    if all(col in proba.columns for col in int_cols):
        return proba[int_cols].to_numpy()

    str_cols = [str(int(c)) for c in class_order]
    if all(col in proba.columns for col in str_cols):
        return proba[str_cols].to_numpy()

    raise KeyError(
        "Could not align probability columns with class order. "
        f"Missing columns for {int_cols} or {str_cols}."
    )


def patch_lambda_globals(model: tf.keras.Model) -> None:
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
    """Load a Keras model with project-specific custom layer support.

    Args:
        model_path (Path): Path to a serialized ``.keras`` model file.

    Returns:
        tf.keras.Model: Deserialized model ready for inference.

    Raises:
        OSError: If the model file cannot be read.
        ValueError: If model deserialization fails due to incompatible config.
        TypeError: If custom objects cannot be resolved during loading.
    """
    custom_objects = {
        "tf": tf,
        "LidarToOneHot": LidarToOneHot,
        "CoordTileFlat": CoordTileFlat,
    }
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects=custom_objects,
    )
    patch_lambda_globals(model)
    return model


def main(
    model_dir: Path | None = None,
    top_x_models: int | None = None,
) -> None:
    """Train and evaluate a stacking ensemble with configurable base-model source.

    This function loads S008/S009 raw datasets, prepares train/validation splits,
    builds meta-features from top-ranked base models, and fits an AutoGluon
    multiclass meta-learner.

    Args:
        model_dir (Path | None, optional): Directory containing base model files
            named with the `top_<rank>_... .keras` convention. If provided,
            this value overrides environment and default directory selection.
            If `None`, selection falls back to `RAYWISE_MODEL_DIR`, then to
            the module default path.
        top_x_models (int | None, optional): Number of top-ranked models to
            load from `model_dir`. If provided, this value overrides
            environment/default selection. If `None`, selection falls back to
            `RAYWISE_TOP_X_MODELS`, then to the module default. Larger values
            use more base predictors (potentially improving diversity at higher
            memory/time cost); smaller values reduce compute and memory.

    Returns:
        None: Results are persisted to disk under the run directory. The run
        writes a JSON report, text summary, scaler checkpoint, and AutoGluon
        predictor directory path so downstream scripts can reconstruct the
        exact inference pipeline.

    Raises:
        ValueError: If filtering removes all S008 samples or if split
            constraints are violated.
        RuntimeError: If ranked model discovery fails.
        FileNotFoundError: If the resolved model directory does not exist.
        NotADirectoryError: If the resolved model path is not a directory.
    """
    x_lidar_s008, x_coord_s008, y_s008 = load_dataset_raw_sparse_labels_s008_only(
        s008_coord_csv=str(S008_COORD_CSV),
        s008_lidar_folder=str(S008_LIDAR_FOLDER),
        s008_beam_output_path=str(S008_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
    )

    x_lidar_test, x_coord_test, y_test = load_dataset_raw_sparse_labels_s009_only(
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
    )

    lidar_s008 = np.asarray(x_lidar_s008, dtype=np.float32)
    coord_s008 = np.asarray(x_coord_s008, dtype=np.float32)
    labels_s008 = np.asarray(y_s008, dtype=np.int64)

    run_dir = RESULTS_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    (
        lidar_s008,
        coord_s008,
        labels_s008,
    ) = _filter_rare_classes(lidar_s008, coord_s008, labels_s008, min_samples=2)
    if labels_s008.size == 0:
        raise ValueError("No samples remain after filtering rare classes from S008.")

    classes = np.sort(np.unique(labels_s008))
    class_to_pos = {int(c): i for i, c in enumerate(classes)}
    num_classes = int(classes.size)
    class_order = list(range(num_classes))

    stratify_labels = _maybe_stratify(labels_s008)
    split_kwargs = {"stratify": stratify_labels} if stratify_labels is not None else {}

    (
        lidar_train,
        lidar_val,
        coord_train,
        coord_val,
        labels_train,
        labels_val,
    ) = train_test_split(
        lidar_s008,
        coord_s008,
        labels_s008,
        train_size=TRAIN_RATIO,
        shuffle=True,
        random_state=DATA_SEED,
        **split_kwargs,
    )
    train_classes = np.unique(labels_train)
    all_classes = np.unique(labels_s008)
    missing_train = np.setdiff1d(all_classes, train_classes)
    if missing_train.size:
        preview = ", ".join(str(int(v)) for v in missing_train[:10])
        suffix = "..." if missing_train.size > 10 else ""
        raise ValueError(
            f"Training split is missing {missing_train.size} class(es): {preview}{suffix}"
        )

    coord_scaler = StandardScaler()
    coord_scaler.fit(coord_train)

    print("\n\n\nLoaded raw S008/S009 data with new 80/20 train/val split:")
    print(f"  S008 train shapes -> lidar: {lidar_train.shape}, coord: {coord_train.shape}, labels: {labels_train.shape}")
    print(f"  S008 val shapes   -> lidar: {lidar_val.shape}, coord: {coord_val.shape}, labels: {labels_val.shape}")
    print(f"  Raw S009 test shapes -> lidar: {x_lidar_test.shape}, coord: {x_coord_test.shape}, labels: {y_test.shape}")

    selected_model_dir, selected_top_x = resolve_model_selection(model_dir, top_x_models)
    selected_model_paths = resolve_top_model_paths(selected_model_dir, selected_top_x)
    print("\nSelected base models:")
    for idx, model_path in enumerate(selected_model_paths, start=1):
        print(f"  {idx}. {model_path.name}")

    models = [load_model(p) for p in selected_model_paths]
    predictions = predict_with_topk_models(models, x_lidar_test, x_coord_test, coord_scaler, batch_size=64)

    print("\nTop-K accuracies on S009:")
    base_topk_results: list[dict[str, object]] = []
    for idx, preds in enumerate(predictions, start=1):
        accuracies = compute_topk_accuracy(preds, y_test)
        acc_lines = ", ".join(f"Top-{k}: {accuracies[k]*100:.2f}%" for k in TOP_K)
        print(f"Model {idx} -> {acc_lines}")
        base_topk_results.append(
            {
                "model_index": idx,
                "model_path": str(selected_model_paths[idx - 1]),
                "topk": {f"top_{k}": float(accuracies[k]) for k in TOP_K},
            }
        )

    # ———————————————————————————————————————————————————————————————————————————— #
    #                                   Ensemble                                   #
    # ———————————————————————————————————————————————————————————————————————————— #

    train_meta_preds = predict_with_topk_models(
        models,
        lidar_train,
        coord_train,
        coord_scaler,
        batch_size=64,
    )
    val_meta_preds = predict_with_topk_models(
        models,
        lidar_val,
        coord_val,
        coord_scaler,
        batch_size=64,
    )

    labels_train_enc = np.array([class_to_pos[int(y)] for y in labels_train], dtype=np.int64)
    labels_val_enc = np.array([class_to_pos[int(y)] for y in labels_val], dtype=np.int64)

    meta_train = stack_meta_features(train_meta_preds)
    meta_val = stack_meta_features(val_meta_preds)

    aligned_train = align_meta_features(
        meta_train,
        classes=classes,
        expected_models=len(selected_model_paths),
    )
    aligned_val = align_meta_features(
        meta_val,
        classes=classes,
        expected_models=len(selected_model_paths),
    )

    train_df = meta_features_to_dataframe(
        aligned_train,
        labels=labels_train_enc,
        classes=classes,
        label_column="label",
        expected_models=len(selected_model_paths),
    )
    val_df = meta_features_to_dataframe(
        aligned_val,
        labels=labels_val_enc,
        classes=classes,
        label_column="label",
        expected_models=len(selected_model_paths),
    )

    del models
    tf.keras.backend.clear_session()
    gc.collect()

    print("\nTraining AutoGluon stacked meta-learner on S008 meta-features:")
    predictor_path = RESULTS_DIR / f"autogluon_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    presets = AG_PRESETS
    predictor = TabularPredictor(
        label="label",
        path=str(predictor_path),
        problem_type="multiclass",
        eval_metric="log_loss",
    ).fit(
        train_data=train_df,
        presets=presets,
        num_stack_levels=AG_NUM_STACK_LEVELS,
        num_bag_folds=AG_NUM_BAG_FOLDS,
        num_bag_sets=AG_NUM_BAG_SETS,
        use_bag_holdout=AG_USE_BAG_HOLDOUT,
        fit_strategy=AG_FIT_STRATEGY,
        # num_gpus=1,
        ag_args_ensemble={
            "fold_fitting_strategy": AG_FOLD_FITTING_STRATEGY,
            "num_folds_parallel": AG_NUM_FOLDS_PARALLEL,
        },
        num_cpus=AG_NUM_CPUS,
        verbosity=AG_VERBOSITY,
    )

    class_order = list(range(num_classes))
    val_probs_df = predictor.predict_proba(val_df.drop(columns=["label"]))
    val_probs = extract_prob_matrix(val_probs_df, class_order)
    val_topk = compute_topk_accuracy(val_probs, labels_val_enc)
    val_lines = ", ".join(f"Top-{k}: {val_topk[k]*100:.2f}%" for k in TOP_K)
    print(f"  S008 val (meta) -> {val_lines}")

    train_probs_df = predictor.predict_proba(train_df.drop(columns=["label"]))
    train_probs = extract_prob_matrix(train_probs_df, class_order)
    train_topk = compute_topk_accuracy(train_probs, labels_train_enc)
    train_lines = ", ".join(f"Top-{k}: {train_topk[k]*100:.2f}%" for k in TOP_K)
    print(f"  S008 train (meta) -> {train_lines}")

    meta_test = stack_meta_features(predictions)
    s009_mask = np.isin(y_test, classes)
    if not s009_mask.all():
        unseen = np.setdiff1d(np.unique(y_test), classes)
        prefix = ", ".join(str(int(v)) for v in unseen[:10])
        suffix = "..." if unseen.size > 10 else ""
        print(
            f"  Warning: dropping {np.count_nonzero(~s009_mask)} S009 samples "
            f"with unseen labels: {prefix}{suffix}"
        )

    meta_test = meta_test[s009_mask]
    y_test_filtered = y_test[s009_mask]
    test_topk: dict[int, float] | None = None
    if meta_test.shape[0] == 0:
        print("  No S009 samples left after filtering, skipping meta evaluation.")
    else:
        y_test_enc = np.array([class_to_pos[int(y)] for y in y_test_filtered], dtype=np.int64)
        aligned_test = align_meta_features(
            meta_test,
            classes=classes,
            expected_models=len(selected_model_paths),
        )
        test_df = meta_features_to_dataframe(
            aligned_test,
            labels=None,
            classes=classes,
            label_column="label",
            expected_models=len(selected_model_paths),
        )
        test_probs_df = predictor.predict_proba(test_df)
        test_probs = extract_prob_matrix(test_probs_df, class_order)
        test_topk = compute_topk_accuracy(test_probs, y_test_enc)
        test_lines = ", ".join(f"Top-{k}: {test_topk[k]*100:.2f}%" for k in TOP_K)
        print(f"  S009 test (meta) -> {test_lines}")

    # ———————————————————————————————————————————————————————————————————————————— #
    #                              Persist artifacts                               #
    # ———————————————————————————————————————————————————————————————————————————— #

    print("Saving AutoGluon predictor and results.")
    model_path = predictor.path

    results = {
        "config": {
            "data_seed": DATA_SEED,
            "train_ratio": TRAIN_RATIO,
            "top_k": list(TOP_K),
            "model_dir": str(selected_model_dir),
            "top_x_models": int(selected_top_x),
            "model_paths": [str(p) for p in selected_model_paths],
        },
        "sizes": {
            "s008_train": int(labels_train.shape[0]),
            "s008_val": int(labels_val.shape[0]),
            "s009_test": int(y_test.shape[0]),
            "s009_test_used": int(meta_test.shape[0]),
        },
        "base_model_topk": base_topk_results,
        "meta_train_topk": {f"top_{k}": float(train_topk[k]) for k in TOP_K},
        "meta_val_topk": {f"top_{k}": float(val_topk[k]) for k in TOP_K},
        "meta_test_topk": {f"top_{k}": float(test_topk[k]) for k in TOP_K} if test_topk else None,
        "classes": [int(c) for c in classes],
        "autogluon_presets": presets,
        "autogluon_num_stack_levels": AG_NUM_STACK_LEVELS,
        "autogluon_num_bag_folds": AG_NUM_BAG_FOLDS,
        "autogluon_num_bag_sets": AG_NUM_BAG_SETS,
        "autogluon_use_bag_holdout": AG_USE_BAG_HOLDOUT,
        "autogluon_fit_strategy": AG_FIT_STRATEGY,
        "autogluon_num_cpus": AG_NUM_CPUS,
        "autogluon_path": str(model_path),
        "artifacts": {
            "autogluon_predictor": str(model_path),
        },
    }

    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Persist preprocessing artifacts required to reproduce validation/test
    # inference for this exact run outside the training process.
    scaler_path = run_dir / "coord_scaler.pkl"
    with scaler_path.open("wb") as scaler_file:
        pickle.dump(coord_scaler, scaler_file)

    classes_path = run_dir / "classes.npy"
    np.save(classes_path, classes)

    class_to_pos_path = run_dir / "class_to_pos.json"
    class_to_pos_path.write_text(json.dumps({str(k): int(v) for k, v in class_to_pos.items()}, indent=2))

    # Update results after writing artifacts so consumers can discover them
    # from a single metadata source.
    results["artifacts"].update(
        {
            "coord_scaler": str(scaler_path),
            "classes": str(classes_path),
            "class_to_pos": str(class_to_pos_path),
        }
    )
    results_path.write_text(json.dumps(results, indent=2))

    def _format_topk_line(name: str, topk: dict[int, float]) -> str:
        parts = ", ".join(f"Top-{k}: {topk[k]*100:.2f}%" for k in TOP_K if k in topk)
        return f"{name}: {parts}"

    acc_lines = [
        _format_topk_line("meta_train", train_topk),
        _format_topk_line("meta_val", val_topk),
    ]
    if test_topk:
        acc_lines.append(_format_topk_line("meta_test", test_topk))
    accuracies_path = run_dir / "accuracies.txt"
    accuracies_path.write_text("\n".join(acc_lines))
    print(f"Saved model to {model_path}")
    print(f"Saved results to {results_path}")
    print(f"Saved accuracies to {accuracies_path}")
    print(f"Saved scaler to {scaler_path}")


if __name__ == "__main__":
    main()
