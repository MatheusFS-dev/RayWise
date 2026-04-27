"""05 - LIDAR and Position-Aided mmWave Beam Selection with Non-local CNNs and Curriculum Training."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


PROJECT_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from _load_dataset import convert_to_sparse_labels
from _load_dataset_raw import load_dataset_raw_sparse_labels_by_condition, load_dataset_raw_sparse_labels_s009_only
from others.measure_model_stats import measure_and_print_model_stats


DATA_SEED = 0
BATCH_SIZE = 64
EPOCHS = 45
TOP_K = (1, 3, 5, 10, 20, 30, 50)

BASELINE_S008_PATH = "./data/s008"
BASELINE_S009_PATH = "./data/s009"

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")
S008_COORD_CSV = DATA_ROOT / "Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
S008_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s008/processed_raw_data/lidar_data_s008"
S008_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s008/baseline_data/beam_output/beams_output_s008.npz"

S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

RESULTS_DIR = Path("results/related_works/related_work_02")
MODEL_NAME = "related_work_02.keras"

USE_CURRICULUM_RAW = False


def load_baseline_s008_splits_with_soft(
    s008_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beam_train = np.load(Path(s008_path) / "beam_output" / "beam_output_train.npz")["output_classification"]
    coord_train = np.load(Path(s008_path) / "coord_input" / "coord_train.npz")["coordinates"]
    lidar_train = np.load(Path(s008_path) / "lidar_input" / "lidar_train.npz")["input"]

    beam_val = np.load(Path(s008_path) / "beam_output" / "beam_output_val.npz")["output_classification"]
    coord_val = np.load(Path(s008_path) / "coord_input" / "coord_val.npz")["coordinates"]
    lidar_val = np.load(Path(s008_path) / "lidar_input" / "lidar_val.npz")["input"]

    beam_train = np.asarray(beam_train)
    beam_val = np.asarray(beam_val)

    beam_train_flat = np.abs(beam_train).reshape(beam_train.shape[0], -1)
    beam_val_flat = np.abs(beam_val).reshape(beam_val.shape[0], -1)

    y_train = convert_to_sparse_labels(beam_train.astype(np.float32))
    y_val = convert_to_sparse_labels(beam_val.astype(np.float32))

    return (
        lidar_train.astype(np.float32),
        coord_train.astype(np.float32),
        y_train,
        beam_train_flat.astype(np.float32),
        lidar_val.astype(np.float32),
        coord_val.astype(np.float32),
        y_val,
        beam_val_flat.astype(np.float32),
    )


def load_baseline_s009(s009_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beam = np.load(Path(s009_path) / "beam_output" / "beam_output.npz")["output_classification"]
    coord = np.load(Path(s009_path) / "coord_input" / "coord_input.npz")["coordinates"]
    lidar = np.load(Path(s009_path) / "lidar_input" / "lidar_input.npz")["input"]

    beam = np.asarray(beam, dtype=np.float32)
    y = convert_to_sparse_labels(beam)

    return lidar.astype(np.float32), coord.astype(np.float32), y


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


def _non_local_block(x: tf.Tensor, inter_channels: int | None = None) -> tf.Tensor:
    channels = int(x.shape[-1])
    inter_channels = inter_channels or max(1, channels // 2)

    theta = tf.keras.layers.Conv2D(inter_channels, 1, padding="same")(x)
    phi = tf.keras.layers.Conv2D(inter_channels, 1, padding="same")(x)
    g = tf.keras.layers.Conv2D(inter_channels, 1, padding="same")(x)

    theta_shape = tf.keras.layers.Reshape((-1, inter_channels))(theta)
    phi_shape = tf.keras.layers.Reshape((-1, inter_channels))(phi)
    g_shape = tf.keras.layers.Reshape((-1, inter_channels))(g)

    attention = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True))([theta_shape, phi_shape])
    attention = tf.keras.layers.Softmax(axis=-1)(attention)

    y = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([attention, g_shape])
    y = tf.keras.layers.Lambda(
        lambda t: tf.reshape(
            t[0],
            (tf.shape(t[1])[0], tf.shape(t[1])[1], tf.shape(t[1])[2], inter_channels),
        )
    )([y, x])
    y = tf.keras.layers.Conv2D(channels, 1, padding="same")(y)

    return tf.keras.layers.Add()([x, y])


def build_model(num_classes: int = 256) -> tf.keras.Model:
    grid_height = 20  #! AUTHORS DO NOT PROVIDE THIS
    grid_width = 200  #! AUTHORS DO NOT PROVIDE THIS
    lidar_channels = 1

    conv_filters = 5
    conv_kernel = (5, 5)
    conv_stride = (1, 1)
    down_stride = (2, 2)
    final_kernel = (1, 5)
    final_stride = (1, 2)

    beta = 0.5  #! AUTHORS DO NOT PROVIDE THIS

    # Top-view occupancy grid with BS (-1) and vehicle (-2) markers.
    lidar_input = tf.keras.Input(shape=(grid_height, grid_width, lidar_channels), name="lidar")
    x = lidar_input
    x = tf.keras.layers.Conv2D(conv_filters, conv_kernel, strides=conv_stride, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(conv_filters, conv_kernel, strides=conv_stride, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(conv_filters, conv_kernel, strides=down_stride, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(conv_filters, conv_kernel, strides=conv_stride, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(conv_filters, conv_kernel, strides=down_stride, padding="same", activation="relu")(x)
    x = _non_local_block(x)
    # Produces 5x25x1 -> 125 features before the 16-unit linear layer.
    x = tf.keras.layers.Conv2D(1, final_kernel, strides=final_stride, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)

    coord_input = tf.keras.Input(shape=(2,), name="coords")
    merged = tf.keras.layers.Concatenate()([x, coord_input])
    merged = tf.keras.layers.Dense(64, activation="relu")(merged)
    merged = tf.keras.layers.Dense(64, activation="relu")(merged)
    merged = tf.keras.layers.Dense(64, activation="relu")(merged)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(merged)

    model = tf.keras.Model(inputs=[lidar_input, coord_input], outputs=output)
    model.beta = beta
    return model


def _make_soft_labels(beam_flat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(beam_flat, axis=1, keepdims=True) + eps
    return beam_flat / norms


def _mix_labels(y_sparse: np.ndarray, y_soft: np.ndarray, beta: float) -> np.ndarray:
    num_classes = y_soft.shape[1]
    one_hot = tf.keras.utils.to_categorical(y_sparse, num_classes=num_classes)
    return (1.0 - beta) * one_hot + beta * y_soft


def _curriculum_acceptance(epoch: int) -> float:
    group = epoch // 9
    rejection = max(1.0 - 0.2 * group, 0.0)
    return 1.0 - rejection


def _train_with_curriculum(
    model: tf.keras.Model,
    x_lidar: np.ndarray,
    x_coord: np.ndarray,
    y_mix: np.ndarray,
    conditions: np.ndarray,
    x_lidar_val: np.ndarray,
    x_coord_val: np.ndarray,
    y_val_mix: np.ndarray,
) -> tf.keras.callbacks.History:
    history = tf.keras.callbacks.History()
    for epoch in range(EPOCHS):
        accept_nlos = _curriculum_acceptance(epoch)
        rng = np.random.default_rng(DATA_SEED + epoch)
        nlos_mask = conditions == "NLOS"
        los_mask = ~nlos_mask
        nlos_keep = rng.random(nlos_mask.sum()) < accept_nlos

        selected_indices = np.concatenate(
            [np.where(los_mask)[0], np.where(nlos_mask)[0][nlos_keep]]
        )
        rng.shuffle(selected_indices)

        epoch_history = model.fit(
            x=[x_lidar[selected_indices], x_coord[selected_indices]],
            y=y_mix[selected_indices],
            validation_data=([x_lidar_val, x_coord_val], y_val_mix),
            epochs=epoch + 1,
            initial_epoch=epoch,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        history.history.setdefault("loss", []).extend(epoch_history.history.get("loss", []))
        history.history.setdefault("val_loss", []).extend(epoch_history.history.get("val_loss", []))
    return history


def main() -> None:
    tf.keras.utils.set_random_seed(DATA_SEED)
    model = build_model(num_classes=256)
    model.summary()

    measure_and_print_model_stats(
        model,
        name="related_work_02",
        batch_size=1,
        gpu_index=0,
        test_runs=10,
        verbose=1,
    )


if __name__ == "__main__":
    main()
