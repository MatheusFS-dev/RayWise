"""02 - Deep Learning on Multimodal Sensor Data at the Wireless Edge for Vehicular Network."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


PROJECT_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from _load_dataset import convert_to_sparse_labels
from _load_dataset_raw import load_dataset_raw_sparse_labels_s009_only
from others.measure_model_stats import measure_and_print_model_stats


DATA_SEED = 0
BATCH_SIZE = 32
EPOCHS = 100
TOP_K = (1, 3, 5, 10, 20, 30, 50)

BASELINE_S008_PATH = "./data/s008"
BASELINE_S009_PATH = "./data/s009"

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")
S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

RESULTS_DIR = Path("results/related_works/related_work_01")
MODEL_NAME = "related_work_01.keras"


def load_baseline_s008_splits(
    s008_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beam_train = np.load(Path(s008_path) / "beam_output" / "beam_output_train.npz")["output_classification"]
    coord_train = np.load(Path(s008_path) / "coord_input" / "coord_train.npz")["coordinates"]
    lidar_train = np.load(Path(s008_path) / "lidar_input" / "lidar_train.npz")["input"]

    beam_val = np.load(Path(s008_path) / "beam_output" / "beam_output_val.npz")["output_classification"]
    coord_val = np.load(Path(s008_path) / "coord_input" / "coord_val.npz")["coordinates"]
    lidar_val = np.load(Path(s008_path) / "lidar_input" / "lidar_val.npz")["input"]

    beam_train = np.asarray(beam_train, dtype=np.float32)
    beam_val = np.asarray(beam_val, dtype=np.float32)

    y_train = convert_to_sparse_labels(beam_train)
    y_val = convert_to_sparse_labels(beam_val)

    return (
        lidar_train.astype(np.float32),
        coord_train.astype(np.float32),
        y_train,
        lidar_val.astype(np.float32),
        coord_val.astype(np.float32),
        y_val,
    )


def load_baseline_s009(s009_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beam = np.load(Path(s009_path) / "beam_output" / "beam_output.npz")["output_classification"]
    coord = np.load(Path(s009_path) / "coord_input" / "coord_input.npz")["coordinates"]
    lidar = np.load(Path(s009_path) / "lidar_input" / "lidar_input.npz")["input"]

    beam = np.asarray(beam, dtype=np.float32)
    y = convert_to_sparse_labels(beam)

    return lidar.astype(np.float32), coord.astype(np.float32), y


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

    prepared_map: Dict[str, np.ndarray] = {}
    for t in model.inputs:
        expected = tuple(t.shape[1:])
        input_name = t.name.split(":")[0]
        if compatible(expected, lidar_shape):
            prepared_map[input_name] = lidar
        elif compatible(expected, coords_shape):
            prepared_map[input_name] = coords
        else:
            raise ValueError(
                f"Cannot match input {t.name} expected {t.shape}. "
                f"lidar={lidar.shape}, coords={coords.shape}"
            )

    if len(prepared_map) > 1:
        return prepared_map
    return next(iter(prepared_map.values()))


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
    if isinstance(predictions, dict):
        predictions = predictions["fusion"]
    elif isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
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
    if isinstance(predictions, dict):
        predictions = predictions["fusion"]
    elif isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
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


def _identity_block_2d(x: tf.Tensor, filters: int, kernel_size: int) -> tf.Tensor:
    skip = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation=None)(x)
    x = tf.keras.layers.Add()([x, skip])
    return tf.keras.layers.Activation("relu")(x)


def build_model(num_classes: int = 256) -> tf.keras.Model:
    l1_reg = 1e-5
    l2_reg = 1e-4
    dropout_rate = 0.25
    reg = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)

    gps_kernel = 2
    gps_block_repeats = 2

    lidar_filters = 32
    lidar_identity_repeats = 3

    fusion_filters = 30
    fusion_block_repeats = 2

    coord_input = tf.keras.Input(shape=(2,), name="coords")
    coord = tf.keras.layers.Reshape((2, 1))(coord_input)
    for _ in range(gps_block_repeats):
        coord = tf.keras.layers.Conv1D(20, gps_kernel, padding="same", activation="relu")(coord)
        coord = tf.keras.layers.Conv1D(10, gps_kernel, padding="same", activation="relu")(coord)
        coord = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(coord)
    coord = tf.keras.layers.Flatten()(coord)
    coord = tf.keras.layers.Dense(
        4 * num_classes,
        activation="relu",
        kernel_regularizer=reg,
    )(coord)
    coord = tf.keras.layers.Dropout(dropout_rate)(coord)
    coord = tf.keras.layers.Dense(2 * num_classes, activation="relu", kernel_regularizer=reg)(coord)
    coord = tf.keras.layers.Dropout(dropout_rate)(coord)
    coord_pre = tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=reg)(coord)
    coord_head = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="gps_head",
    )(coord_pre)
    coord = tf.keras.layers.Activation("tanh", name="gps_embedding")(coord_pre)

    lidar_input = tf.keras.Input(shape=(20, 200, 10), name="lidar")
    lidar = tf.keras.layers.Conv2D(lidar_filters, (3, 3), padding="same", activation="relu")(lidar_input)
    for _ in range(lidar_identity_repeats):
        lidar = _identity_block_2d(lidar, lidar_filters, 3)
        lidar = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(lidar)
    lidar = tf.keras.layers.Flatten()(lidar)
    lidar = tf.keras.layers.Dense(
        2 * num_classes,
        activation="relu",
        kernel_regularizer=reg,
    )(lidar)
    lidar = tf.keras.layers.Dropout(dropout_rate)(lidar)
    lidar_pre = tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=reg)(lidar)
    lidar_head = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="lidar_head",
    )(lidar_pre)
    lidar = tf.keras.layers.Activation("tanh", name="lidar_embedding")(lidar_pre)

    fusion = tf.keras.layers.Lambda(lambda t: tf.stack(t, axis=-1))([coord, lidar])
    for _ in range(fusion_block_repeats):
        fusion = tf.keras.layers.Conv1D(fusion_filters, 7, padding="same", activation="relu")(fusion)
        fusion = tf.keras.layers.Conv1D(fusion_filters, 5, padding="same", activation="relu")(fusion)
        fusion = tf.keras.layers.MaxPooling1D(pool_size=5, padding="same")(fusion)
    fusion = tf.keras.layers.Flatten()(fusion)
    fusion = tf.keras.layers.Dense(3 * num_classes, activation="relu", kernel_regularizer=reg)(fusion)
    fusion = tf.keras.layers.Dropout(dropout_rate)(fusion)
    fusion = tf.keras.layers.Dense(2 * num_classes, activation="relu", kernel_regularizer=reg)(fusion)
    fusion = tf.keras.layers.Dropout(dropout_rate)(fusion)
    output = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="fusion",
    )(fusion)

    return tf.keras.Model(
        inputs={"lidar": lidar_input, "coords": coord_input},
        outputs=[output, coord_head, lidar_head],
    )


def main() -> None:
    tf.keras.utils.set_random_seed(DATA_SEED)
    model = build_model(num_classes=256)
    model.summary()

    measure_and_print_model_stats(
        model,
        name="related_work_01",
        batch_size=1,
        gpu_index=0,
        test_runs=10,
        verbose=1,
    )


if __name__ == "__main__":
    main()
