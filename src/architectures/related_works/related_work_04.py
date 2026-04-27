"""11 - Efficient mmWave Beam Selection using ViTs and GVEC: GPS-based Virtual Environment Capture."""

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
from _load_dataset_raw import load_dataset_raw_sparse_labels_s009_only
from others.measure_model_stats import measure_and_print_model_stats


DATA_SEED = 0
BATCH_SIZE = 64
EPOCHS = 200
TOP_K = (1, 3, 5, 10, 20, 30, 50)

BASELINE_S008_PATH = "./data/s008"
BASELINE_S009_PATH = "./data/s009"

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")
S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

RESULTS_DIR = Path("results/related_works/related_work_04")
MODEL_NAME = "related_work_04.keras"


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


def _mlp(x: tf.Tensor, hidden_units: list[int], dropout_rate: float) -> tf.Tensor:
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(num_outputs: int = 256) -> tf.keras.Model:
    lidar_input_shape = (72, 72, 3)
    patch_size = (6, 6)
    embed_dim = 32  #! AUTHORS DO NOT PROVIDE THIS
    num_heads = 4
    num_layers = 4
    mlp_units = [64, 32]
    key_dim = 32
    dropout_rate = 0.1  #! AUTHORS DO NOT PROVIDE THIS
    if lidar_input_shape[0] % patch_size[0] or lidar_input_shape[1] % patch_size[1]:
        raise ValueError("lidar_input_shape must be divisible by patch_size")

    coord_input = tf.keras.Input(shape=(2,), name="coords")
    coord = tf.keras.layers.Dense(64, activation="relu")(coord_input)
    coord = tf.keras.layers.Dense(64, activation="relu")(coord)
    coord = tf.keras.layers.Dense(64, activation="relu")(coord)
    coord = tf.keras.layers.Dense(256, activation="relu")(coord)

    lidar_input = tf.keras.Input(shape=lidar_input_shape, name="lidar")
    x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding="valid")(lidar_input)
    num_patches = (lidar_input_shape[0] // patch_size[0]) * (lidar_input_shape[1] // patch_size[1])
    x = tf.keras.layers.Reshape((num_patches, embed_dim))(x)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    x = x + pos_embed

    for _ in range(num_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x1, x1)
        x2 = tf.keras.layers.Add()([attention, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = _mlp(x3, mlp_units, dropout_rate)
        x = tf.keras.layers.Add()([x3, x2])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    vit_features = tf.keras.layers.GlobalAveragePooling1D()(x)
    vit_features = tf.keras.layers.Dense(256, activation="relu")(vit_features)

    fused = tf.keras.layers.Concatenate()([vit_features, coord])
    fused = tf.keras.layers.Dense(256, activation="relu")(fused)
    output = tf.keras.layers.Dense(num_outputs, activation="softmax", name="channel_gains")(fused)

    return tf.keras.Model(inputs=[lidar_input, coord_input], outputs=output)


def main() -> None:
    tf.keras.utils.set_random_seed(DATA_SEED)
    model = build_model(num_outputs=256)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.KLDivergence(),
    )

    measure_and_print_model_stats(
        model,
        name="related_work_04",
        batch_size=1,
        gpu_index=0,
        test_runs=10,
        verbose=1,
    )


if __name__ == "__main__":
    main()
