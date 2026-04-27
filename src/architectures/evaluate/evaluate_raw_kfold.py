from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from _load_dataset import load_dataset_sparse_labels
from _load_dataset_raw import load_dataset_raw_sparse_labels_s009_only


DATA_SEED = 0
BATCH_SIZE = 64
TOP_K = (1, 3, 5, 10, 20, 30, 50)

# If True, load the model using registered custom layers (no Lambda layers).
# If False, fall back to the legacy Lambda-based loading flow.
USE_CUSTOM_LAYERS = False

# If True, fit the scaler on baseline S008 coordinate data and apply it to both baseline
# and raw S009 data. This ensures consistent scaling across datasets using a reference distribution.
# If False, fit the scaler independently on the raw S009 coordinate data before evaluation,
# which adapts scaling to the actual distribution of the test set.
USE_BASELINE_SCALER = True

# Directory containing fold models named like "fold_1/model.keras".
MODEL_DIR = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/kfold/train_cnn1d_v7.0_kfold_10_splits/model_training/model"
)

BASELINE_S008_PATH = "./data/s008"
BASELINE_S009_PATH = "./data/s009"

S009_COORD_CSV = Path(
    "/media/matheus/SSD-2/matheus/datasets/RayWise/Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
)
S009_LIDAR_FOLDER = Path(
    "/media/matheus/SSD-2/matheus/datasets/RayWise/Raymobtime_s009/processed_raw_data/lidar_data_s009"
)
S009_BEAM_OUTPUT = Path(
    "/media/matheus/SSD-2/matheus/datasets/RayWise/Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"
)


@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(tf.keras.layers.Layer):
    """Convert the raw LiDAR tensor into a 4-channel semantic one-hot map.

    Returns:
        tf.Tensor: A float32 tensor with shape ``(batch, 20, 200, 4)`` encoding
        base station, vehicle, obstacle, and free-space channels.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic one-hot encoding to the LiDAR grid.

        Args:
            inputs (tf.Tensor): Raw LiDAR tensor with shape ``(batch, 20, 200, 10)``
                where each channel encodes semantic values in ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)``.
        """

        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(tf.keras.layers.Layer):
    """Tile 2D coordinates to align with the flattened LiDAR sequence length.

    Args:
        grid_length (int): Length of the flattened LiDAR grid (e.g., ``20 * 200``).
    """

    def __init__(self, grid_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Tile coordinates across the flattened LiDAR sequence.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled tensor with shape ``(batch, grid_length, 2)``.
        """

        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict:
        """Return the layer configuration for serialization.

        Returns:
            dict: Serializable configuration for the layer.
        """

        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a trained model, optionally registering custom layers.

    Args:
        model_path (Path): Filesystem path to the saved Keras model.

    Returns:
        tf.keras.Model: Loaded model instance ready for inference.
    """

    custom_objects = {"tf": tf}
    if USE_CUSTOM_LAYERS:
        custom_objects.update({
            "LidarToOneHot": LidarToOneHot,
            "CoordTileFlat": CoordTileFlat,
        })

    # Lambda-based models require safe_mode disabled to deserialize callables.
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=USE_CUSTOM_LAYERS,
        compile=False,
        custom_objects=custom_objects,
    )
    return model


def load_kfold_models(model_dir: Path) -> List[tf.keras.Model]:
    """Load all fold models from a k-fold directory.

    Args:
        model_dir (Path): Directory containing fold subfolders with model files.

    Returns:
        list[tf.keras.Model]: Loaded fold models.

    Raises:
        FileNotFoundError: If no fold model files are found.
    """

    model_paths = sorted(model_dir.glob("fold_*/model.keras"))
    if not model_paths:
        raise FileNotFoundError(f"No fold models found under {model_dir}.")

    return [load_model(path) for path in model_paths]


def scale_coordinates(scaler: object, *coord_arrays: np.ndarray) -> list[np.ndarray]:
    """Scale coordinate arrays with a fitted scaler.

    Args:
        scaler (object): Fitted scaler implementing ``transform``.
        *coord_arrays (np.ndarray): Coordinate arrays to transform.

    Returns:
        list[np.ndarray]: Scaled arrays in float32.
    """

    scaled = [scaler.transform(arr) for arr in coord_arrays]
    return [arr.astype(np.float32) for arr in scaled]


def _prepare_inputs(model: tf.keras.Model, lidar: np.ndarray, coords: np.ndarray):
    lidar = np.asarray(lidar, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)

    lidar_shape = tuple(lidar.shape[1:])
    coords_shape = tuple(coords.shape[1:])

    def compatible(expected, actual) -> bool:
        if len(expected) != len(actual):
            return False
        for e, a in zip(expected, actual):
            if e is not None and int(e) != int(a):
                return False
        return True

    prepared = []
    for t in model.inputs:
        expected = tuple(t.shape[1:])
        if compatible(expected, lidar_shape):
            prepared.append(lidar)
        elif compatible(expected, coords_shape):
            prepared.append(coords)
        else:
            raise ValueError(
                f"Cannot match input {t.name} expected {t.shape}. "
                f"lidar={lidar.shape}, coords={coords.shape}"
            )

    return prepared if len(prepared) > 1 else prepared[0]


def _topk_stats(predictions: np.ndarray, labels: np.ndarray, ks: tuple[int, ...]) -> Dict[int, Dict[str, float]]:
    """Compute top-k correctness counts/accuracy for multiple k values."""
    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions with shape (N, C), got {predictions.shape}")
    total = int(labels.shape[0])
    if total == 0:
        return {k: {"correct": 0, "accuracy": 0.0} for k in ks}

    num_classes = predictions.shape[1]
    sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]

    stats: Dict[int, Dict[str, float]] = {}
    for k in ks:
        k_eff = min(k, num_classes)
        topk = sorted_indices[:, :k_eff]
        hits = (topk == labels[:, None]).any(axis=1)
        correct = int(hits.sum())
        stats[k] = {"correct": correct, "accuracy": correct / total}
    return stats


def _validate_topk_inputs(predictions: np.ndarray, labels: np.ndarray) -> None:
    """Validate prediction/label alignment for top-k evaluation.

    Args:
        predictions (np.ndarray): Model outputs with shape ``(N, C)`` where ``C``
            is the number of classes. Values may be logits or probabilities; only
            the ordering is used by the top-k computation.
        labels (np.ndarray): Sparse integer labels with shape ``(N,)``. Each label
            must be an integer in the inclusive range ``[0, C-1]``. If labels are
            shaped ``(N, 1)`` or are one-hot encoded, this check will fail to
            prevent silent mis-evaluation.

    Raises:
        ValueError: If the predictions are not 2D, labels are not 1D, sample
            counts mismatch, or label values are outside the valid class range.
        TypeError: If labels are not integer-compatible values.
    """

    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions with shape (N, C), got {predictions.shape}")

    if labels.ndim != 1:
        raise ValueError(
            "Top-k expects sparse labels with shape (N,), but received labels with "
            f"shape {labels.shape}."
        )

    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(
            "Predictions/labels sample count mismatch: "
            f"predictions={predictions.shape[0]}, labels={labels.shape[0]}."
        )

    if labels.size == 0:
        return

    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.round(labels))):
            raise TypeError("Labels must be integer class IDs for top-k evaluation.")
        labels = labels.astype(int)

    num_classes = predictions.shape[1]
    label_min = int(labels.min())
    label_max = int(labels.max())

    if label_min < 0 or label_max >= num_classes:
        raise ValueError(
            "Label IDs must be in the range [0, C-1]. "
            f"Observed min={label_min}, max={label_max}, C={num_classes}."
        )


def _average_predictions(
    models: Iterable[tf.keras.Model],
    lidar: np.ndarray,
    coords: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Average predictions across multiple fold models.

    Args:
        models (Iterable[tf.keras.Model]): Fold models to ensemble.
        lidar (np.ndarray): LiDAR inputs.
        coords (np.ndarray): Coordinate inputs.
        batch_size (int): Batch size for prediction.

    Returns:
        np.ndarray: Averaged predictions with shape ``(N, C)``.
    """

    preds_sum = None
    model_count = 0

    for model in models:
        inputs = _prepare_inputs(model, lidar, coords)
        preds = model.predict(inputs, batch_size=batch_size, verbose=0)
        if preds_sum is None:
            preds_sum = np.zeros_like(preds, dtype=np.float64)
        preds_sum += preds.astype(np.float64)
        model_count += 1

    if preds_sum is None:
        raise ValueError("No models provided for bagging.")

    return (preds_sum / float(model_count)).astype(np.float32)


def evaluate_split_predictions(
    *,
    name: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
    """Evaluate top-k metrics for LOS/NLOS splits using precomputed predictions.

    Args:
        name (str): Label for the report.
        predictions (np.ndarray): Averaged predictions with shape ``(N, C)``.
        labels (np.ndarray): Sparse integer labels with shape ``(N,)``.
        conditions (np.ndarray): LOS/NLOS labels aligned with ``labels``.
        top_k (tuple[int, ...]): Top-k values to report.

    Returns:
        Dict[str, float]: Summary metrics including per-condition top-k stats.
    """

    _validate_topk_inputs(predictions, labels)
    pred_labels = predictions.argmax(axis=1)

    correct_mask = pred_labels == labels
    total = int(labels.shape[0])
    correct_total = int(correct_mask.sum())

    condition_array = np.asarray(conditions)
    los_mask = condition_array == "LOS"
    nlos_mask = condition_array == "NLOS"

    los_total = int(los_mask.sum())
    nlos_total = int(nlos_mask.sum())

    los_correct = int(np.logical_and(correct_mask, los_mask).sum())
    nlos_correct = int(np.logical_and(correct_mask, nlos_mask).sum())

    topk_overall = _topk_stats(predictions, labels, top_k)
    topk_los = _topk_stats(predictions[los_mask], labels[los_mask], top_k)
    topk_nlos = _topk_stats(predictions[nlos_mask], labels[nlos_mask], top_k)

    return {
        "name": name,
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "los_total": los_total,
        "los_correct": los_correct,
        "los_accuracy": los_correct / los_total if los_total else 0.0,
        "nlos_total": nlos_total,
        "nlos_correct": nlos_correct,
        "nlos_accuracy": nlos_correct / nlos_total if nlos_total else 0.0,
        "topk": topk_overall,
        "los_topk": topk_los,
        "nlos_topk": topk_nlos,
    }


def evaluate_accuracy_predictions(
    *,
    name: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
    """Evaluate top-k metrics using precomputed predictions.

    Args:
        name (str): Label for the report.
        predictions (np.ndarray): Averaged predictions with shape ``(N, C)``.
        labels (np.ndarray): Sparse integer labels with shape ``(N,)``.
        top_k (tuple[int, ...]): Top-k values to report.

    Returns:
        Dict[str, float]: Summary metrics for the evaluation.
    """

    _validate_topk_inputs(predictions, labels)
    pred_labels = predictions.argmax(axis=1)

    correct_mask = pred_labels == labels
    total = int(labels.shape[0])
    correct_total = int(correct_mask.sum())

    topk_overall = _topk_stats(predictions, labels, top_k)

    return {
        "name": name,
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / total if total else 0.0,
        "topk": topk_overall,
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    def _print_topk_block(label: str, topk_stats: Dict[int, Dict[str, float]], total: int, indent: str = "  "):
        for k in sorted(topk_stats.keys()):
            stats = topk_stats[k]
            print(f"{indent}{label}Top-{k}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{total})")

    print(metrics["name"])
    _print_topk_block("", metrics["topk"], metrics["total"])
    print(f"  LOS samples: {metrics['los_total']}")
    _print_topk_block("LOS ", metrics["los_topk"], metrics["los_total"])
    print(f"  NLOS samples: {metrics['nlos_total']}")
    _print_topk_block("NLOS ", metrics["nlos_topk"], metrics["nlos_total"])


def print_simple_accuracy(metrics: Dict[str, float]) -> None:
    print(metrics["name"])
    for k in sorted(metrics["topk"].keys()):
        stats = metrics["topk"][k]
        print(f"  Top-{k}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{metrics['total']})")


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


def _patch_models_if_needed(models: Iterable[tf.keras.Model]) -> None:
    """Patch Lambda globals for legacy models when custom layers are disabled.

    Args:
        models (Iterable[tf.keras.Model]): Loaded fold models.
    """

    if USE_CUSTOM_LAYERS:
        return

    for model in models:
        patch_lambda_globals(model)


def main() -> None:
    models = load_kfold_models(MODEL_DIR)
    _patch_models_if_needed(models)

    (
        s008_coord_input,
        _,
        _,
        x_coord_s009_baseline,
        x_lidar_s009_baseline,
        y_s009_baseline,
    ) = load_dataset_sparse_labels(
        s008_path=str(BASELINE_S008_PATH),
        s009_path=str(BASELINE_S009_PATH),
    )

    x_lidar_s009, x_coord_s009, y_s009, condition_s009 = load_dataset_raw_sparse_labels_s009_only(
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
        remove_null_labels=False,
        return_conditions=True,
    )

    coord_scaler = StandardScaler()

    if USE_BASELINE_SCALER:
        # Fit on baseline S008 data; maintains consistent scaling across both evaluations.
        coord_scaler.fit(s008_coord_input)
    else:
        # Fit on raw S009 data; adapts scaling to the actual test distribution.
        coord_scaler.fit(x_coord_s009)

    x_coord_s009_baseline = scale_coordinates(coord_scaler, x_coord_s009_baseline)[0]
    x_coord_s009 = scale_coordinates(coord_scaler, x_coord_s009)[0]

    baseline_predictions = _average_predictions(
        models,
        x_lidar_s009_baseline,
        x_coord_s009_baseline,
        batch_size=BATCH_SIZE,
    )
    raw_predictions = _average_predictions(
        models,
        x_lidar_s009,
        x_coord_s009,
        batch_size=BATCH_SIZE,
    )

    baseline_metrics = evaluate_accuracy_predictions(
        name="Test on Baseline (Bagged)",
        predictions=baseline_predictions,
        labels=y_s009_baseline,
    )

    s009_metrics = evaluate_split_predictions(
        name="Test on Raw (Bagged)",
        predictions=raw_predictions,
        labels=y_s009,
        conditions=condition_s009,
    )

    print_simple_accuracy(baseline_metrics)
    print_metrics(s009_metrics)


if __name__ == "__main__":
    main()