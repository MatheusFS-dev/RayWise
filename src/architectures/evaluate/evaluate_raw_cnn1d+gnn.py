from pathlib import Path
from typing import Dict

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
# ? From testing, using the baseline scaler is much better (which makes sense)

MODEL_PATH = Path("/media/matheus/SSD-2/matheus/results/raywise/cnn1d+gnn/v0/nas_cnn1d+gnn_v0.2/optuna_study/models/top_1_trial_19.keras")

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


def _legacy_lidar_to_one_hot(inputs: tf.Tensor) -> tf.Tensor:
    """Match the notebook Lambda used to convert raw LiDAR to 4 channels."""

    base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
    vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
    obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
    free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
    return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)


def _legacy_coord_tile_flat(inputs: tf.Tensor, grid_length: int = 20 * 200) -> tf.Tensor:
    """Match the notebook Lambda used to tile 2D coordinates over the LiDAR grid."""

    return tf.tile(tf.expand_dims(inputs, axis=1), [1, grid_length, 1])


def _resize_1d_nearest(inputs: tf.Tensor, target_length: int) -> tf.Tensor:
    """Resize a ``(batch, length, channels)`` tensor along the sequence axis."""

    expanded = tf.expand_dims(inputs, axis=2)
    resized = tf.image.resize(expanded, size=(target_length, 1), method="nearest")
    return tf.squeeze(resized, axis=2)


@tf.keras.utils.register_keras_serializable(package="raywise")
def _func(inputs: tf.Tensor) -> tf.Tensor:
    """Legacy Lambda dispatcher for models saved from the NAS notebook.

    The original notebook serialized several anonymous Lambdas under the same
    generated name ``_func``. During loading, Keras only knows that symbolic
    function name, so this dispatcher infers which transformation to apply from
    the input rank/shape used by this architecture.
    """

    rank = inputs.shape.rank
    last_dim = inputs.shape[-1]
    seq_len = inputs.shape[1] if rank and rank >= 2 else None

    if rank == 4 and last_dim == 10:
        return _legacy_lidar_to_one_hot(inputs)

    if rank == 2 and last_dim == 2:
        return _legacy_coord_tile_flat(inputs)

    if rank == 3:
        if seq_len == 2000:
            return _resize_1d_nearest(inputs, target_length=1000)

        if seq_len == 4000 and last_dim in (380, 750):
            return inputs

        dynamic_len = tf.shape(inputs)[1]
        return tf.cond(
            tf.equal(dynamic_len, 2000),
            lambda: _resize_1d_nearest(inputs, target_length=1000),
            lambda: inputs,
        )

    raise ValueError(
        "Unsupported legacy Lambda input shape while deserializing model: "
        f"rank={rank}, shape={inputs.shape}."
    )


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a trained model, optionally registering custom layers.

    Args:
        model_path (Path): Filesystem path to the saved Keras model.

    Returns:
        tf.keras.Model: Loaded model instance ready for inference.
    """

    custom_objects = {
        "tf": tf,
        "_func": _func,
    }
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


def scale_coordinates(scaler: object, *coord_arrays: np.ndarray) -> list[np.ndarray]:
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

    # Enforce 2D prediction tensor to match (N, C) assumptions in top-k logic.
    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions with shape (N, C), got {predictions.shape}")

    # Require sparse label vector; (N, 1) or one-hot labels will distort top-k hits.
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

    # Ensure labels are integer-like before range validation to avoid silent casting.
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.round(labels))):
            raise TypeError("Labels must be integer class IDs for top-k evaluation.")
        labels = labels.astype(int)

    num_classes = predictions.shape[1]
    label_min = int(labels.min())
    label_max = int(labels.max())

    # Guard against labels outside the model output class range.
    if label_min < 0 or label_max >= num_classes:
        raise ValueError(
            "Label IDs must be in the range [0, C-1]. "
            f"Observed min={label_min}, max={label_max}, C={num_classes}."
        )


def evaluate_split(
    *,
    name: str,
    model: tf.keras.Model,
    lidar: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    batch_size: int,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
    inputs = _prepare_inputs(model, lidar, coords)
    predictions = model.predict(inputs, batch_size=batch_size, verbose=0)
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


def evaluate_accuracy(
    *,
    name: str,
    model: tf.keras.Model,
    lidar: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
    inputs = _prepare_inputs(model, lidar, coords)
    predictions = model.predict(inputs, batch_size=batch_size, verbose=0)
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


def main() -> None:
    model = load_model(MODEL_PATH)

    if not USE_CUSTOM_LAYERS:
        patch_lambda_globals(model)

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

    baseline_metrics = evaluate_accuracy(
        name="Test on Baseline",
        model=model,
        lidar=x_lidar_s009_baseline,
        coords=x_coord_s009_baseline,
        labels=y_s009_baseline,
        batch_size=BATCH_SIZE,
    )

    s009_metrics = evaluate_split(
        name="Test on Raw",
        model=model,
        lidar=x_lidar_s009,
        coords=x_coord_s009,
        labels=y_s009,
        conditions=condition_s009,
        batch_size=BATCH_SIZE,
    )

    print_simple_accuracy(baseline_metrics)
    print_metrics(s009_metrics)


if __name__ == "__main__":
    main()
