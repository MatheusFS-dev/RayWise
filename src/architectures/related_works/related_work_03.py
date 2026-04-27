"""10 - Efficient Dynamic mmWave Beam Selection Using Multimodal Attention-Based Approach."""

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
EPOCHS = 100
TOP_K = (1, 3, 5, 10, 20, 30, 50)

BASELINE_S008_PATH = "./data/s008"
BASELINE_S009_PATH = "./data/s009"

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")
S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

RESULTS_DIR = Path("results/related_works/related_work_03")
MODEL_NAME = "related_work_03.keras"


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


def build_model(num_classes: int = 256) -> tf.keras.Model:
    coord_units = (32, 256, 512)
    lidar_filters = (8, 16)
    kernel_size = (5, 5)
    dropout_rate = 0.2

    num_attention_layers = 3  # Figure shows 3 stacked MHA layers
    num_heads = 16  #! AUTHORS DO NOT PROVIDE THIS
    token_shape = (32, 16)  #! AUTHORS DO NOT PROVIDE THIS
    if token_shape[1] % num_heads != 0:
        raise ValueError("token_shape[-1] must be divisible by num_heads")
    key_dim = token_shape[1] // num_heads  # per-head size; total matches embedding dim

    coord_input = tf.keras.Input(shape=(2,), name="coords")
    coord = tf.keras.layers.Dense(coord_units[0], activation="relu")(coord_input)
    coord = tf.keras.layers.Dense(coord_units[1], activation="relu")(coord)
    coord = tf.keras.layers.Dense(coord_units[2], activation="relu")(coord)

    lidar_input = tf.keras.Input(shape=(20, 200, 10), name="lidar")
    lidar = tf.keras.layers.Conv2D(lidar_filters[0], kernel_size, padding="same", activation="relu")(lidar_input)
    lidar = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(lidar)
    lidar = tf.keras.layers.Dropout(dropout_rate)(lidar)
    lidar = tf.keras.layers.Conv2D(lidar_filters[1], kernel_size, padding="same", activation="relu")(lidar)
    lidar = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(lidar)
    lidar = tf.keras.layers.Dropout(dropout_rate)(lidar)
    lidar = tf.keras.layers.Flatten()(lidar)
    lidar = tf.keras.layers.Dense(512, activation="relu")(lidar)

    # Reshape 512-d features into a token sequence for non-trivial attention.
    coord_tokens = tf.keras.layers.Reshape(token_shape)(coord)
    lidar_tokens = tf.keras.layers.Reshape(token_shape)(lidar)

    fused = coord_tokens
    for _ in range(num_attention_layers):
        fused = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
            query=fused,
            value=lidar_tokens,
            key=lidar_tokens,
        )
    fused = tf.keras.layers.Flatten()(fused)
    fused = tf.keras.layers.Dense(512, activation="relu")(fused)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(fused)

    return tf.keras.Model(inputs=[lidar_input, coord_input], outputs=output)


def main() -> None:
    tf.keras.utils.set_random_seed(DATA_SEED)

    (
        x_lidar_train,
        x_coord_train,
        y_train,
        x_lidar_val,
        x_coord_val,
        y_val,
    ) = load_baseline_s008_splits(BASELINE_S008_PATH)

    coord_scaler = StandardScaler()
    coord_scaler.fit(x_coord_train)
    x_coord_train, x_coord_val = scale_coordinates(coord_scaler, x_coord_train, x_coord_val)

    lidar_max = max(float(x_lidar_train.max()), 1.0)
    x_lidar_train = (x_lidar_train / lidar_max).astype(np.float32)
    x_lidar_val = (x_lidar_val / lidar_max).astype(np.float32)

    num_classes = int(max(y_train.max(), y_val.max())) + 1
    model = build_model(num_classes=num_classes)
    model.summary()
    measure_and_print_model_stats(
        model,
        name="related_work_03",
        batch_size=1,
        gpu_index=0,
        test_runs=10,
        verbose=1,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  #! AUTHORS DO NOT PROVIDE THIS
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_1")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_top_1", patience=10, restore_best_weights=True),
    ]

    history = model.fit(
        x=[x_lidar_train, x_coord_train],
        y=y_train,
        validation_data=([x_lidar_val, x_coord_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / MODEL_NAME
    model.save(model_path)

    x_lidar_s009_base, x_coord_s009_base, y_s009_base = load_baseline_s009(BASELINE_S009_PATH)
    x_coord_s009_base = scale_coordinates(coord_scaler, x_coord_s009_base)[0]
    x_lidar_s009_base = (x_lidar_s009_base / lidar_max).astype(np.float32)

    x_lidar_s009_raw, x_coord_s009_raw, y_s009_raw, condition_s009 = load_dataset_raw_sparse_labels_s009_only(
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
        remove_null_labels=False,
        return_conditions=True,
    )
    x_coord_s009_raw = scale_coordinates(coord_scaler, x_coord_s009_raw)[0]
    x_lidar_s009_raw = (x_lidar_s009_raw / lidar_max).astype(np.float32)

    baseline_metrics = evaluate_accuracy(
        name="Test on Baseline",
        model=model,
        lidar=x_lidar_s009_base,
        coords=x_coord_s009_base,
        labels=y_s009_base,
        batch_size=BATCH_SIZE,
    )

    raw_metrics = evaluate_split(
        name="Test on Raw",
        model=model,
        lidar=x_lidar_s009_raw,
        coords=x_coord_s009_raw,
        labels=y_s009_raw,
        conditions=condition_s009,
        batch_size=BATCH_SIZE,
    )

    print_simple_accuracy(baseline_metrics)
    print_metrics(raw_metrics)

    best_val = max(history.history.get("val_top_1", []), default=0.0)
    print(f"Best val top-1: {best_val * 100:.2f}%")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
