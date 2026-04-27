"""Load Raymobtime raw datasets, split S008, and prepare ensemble predictions."""

from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _load_dataset_raw import (
    load_dataset_raw_sparse_labels_s008_only,
    load_dataset_raw_sparse_labels_s009_only,
)

TOP_K = (1, 3, 5, 10, 20, 30, 50)

DATA_SEED = 0
TRAIN_RATIO = 0.8

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")

S008_COORD_CSV = DATA_ROOT / "Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
S008_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s008/processed_raw_data/lidar_data_s008"
S008_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s008/baseline_data/beam_output/beams_output_s008.npz"

S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

MODEL_PATHS = [
    Path("/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model/top_1_trial_1116.keras"),
    Path("/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model/top_2_trial_690.keras"),
    Path("/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model/top_3_trial_460.keras"),
]


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


def predict_with_topk_models(
    models: Sequence[tf.keras.Model],
    lidar_inputs: np.ndarray,
    coord_inputs: np.ndarray,
    coord_scaler: StandardScaler,
    *,
    batch_size: int = 64,
) -> list[np.ndarray]:
    """Predict on the provided data with three top-ranked models."""
    if len(models) != 3:
        raise ValueError("predict_with_topk_models expects exactly three models.")

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
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects={"tf": tf},
    )
    patch_lambda_globals(model)
    return model


def main() -> dict[str, np.ndarray | StandardScaler]:
    """Load Raymobtime raw data, split S008, and fit a coordinate scaler."""
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
    )

    coord_scaler = StandardScaler()
    coord_scaler.fit(coord_train)

    print("\n\n\nLoaded raw S008/S009 data with new 80/20 train/val split:")
    print(f"  S008 train shapes -> lidar: {lidar_train.shape}, coord: {coord_train.shape}, labels: {labels_train.shape}")
    print(f"  S008 val shapes   -> lidar: {lidar_val.shape}, coord: {coord_val.shape}, labels: {labels_val.shape}")
    print(f"  Raw S009 test shapes -> lidar: {x_lidar_test.shape}, coord: {x_coord_test.shape}, labels: {y_test.shape}")

    models = [load_model(p) for p in MODEL_PATHS]
    predictions = predict_with_topk_models(models, x_lidar_test, x_coord_test, coord_scaler, batch_size=64)

    print("\nTop-K accuracies on S009:")
    for idx, preds in enumerate(predictions, start=1):
        accuracies = compute_topk_accuracy(preds, y_test)
        acc_lines = ", ".join(f"Top-{k}: {accuracies[k]*100:.2f}%" for k in TOP_K)
        print(f"Model {idx} -> {acc_lines}")
        
    # ———————————————————————————————————————————————————————————————————————————— #
    #                                   Ensemble                                   #
    # ———————————————————————————————————————————————————————————————————————————— #
    
    
    
    
    # ———————————————————————————————————————————————————————————————————————————— #
