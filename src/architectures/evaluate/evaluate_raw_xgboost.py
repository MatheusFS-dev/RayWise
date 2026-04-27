from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from _load_dataset import load_dataset_sparse_labels
from _load_dataset_raw import load_dataset_raw_sparse_labels_s009_only


DATA_SEED = 0
BATCH_SIZE = 64
TOP_K = (1, 3, 5, 10, 20, 30, 50)
META_TRAIN_RATIO = 0.8

# Base CNN1D models used to generate meta-features.
MODEL_PATHS = [
    Path("/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model/top_1_trial_1116.keras"),
    Path("/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model/top_2_trial_690.keras"),
    Path("/home/matheus/src/RayWise/results/cnn1d/v7/nas_cnn1d_v7.0/optuna_study/model/top_3_trial_460.keras"),
]

# Trained XGBoost meta-learner path (produced by nas_ensemble_cnn1d_v7_xgb_v0.0.py).
XGB_MODEL_PATH = Path(
    "/home/matheus/src/RayWise/results/cnn1d/ensemble/nas_ensemble_cnn1d_v7_xgb_v0.0/xgb_model.json"
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

MIN_SAMPLES_PER_CLASS = 2


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


def load_base_model(model_path: Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects={"tf": tf},
    )
    patch_lambda_globals(model)
    return model


def load_xgb_model(model_path: Path) -> XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing XGBoost model at {model_path}")
    xgb = XGBClassifier()
    xgb.load_model(str(model_path))
    return xgb


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

    prepared_by_name: Dict[str, np.ndarray] = {}
    for t in model.inputs:
        expected = tuple(t.shape[1:])
        input_name = t.name.split(":")[0]
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
                f"Cannot match input {t.name} expected {t.shape}. "
                f"lidar={lidar.shape}, coords={coords.shape}"
            )
        prepared_by_name[input_name] = candidate

    ordered_inputs = [prepared_by_name[t.name.split(":")[0]] for t in model.inputs]
    if len(ordered_inputs) == 1:
        return ordered_inputs[0]
    return ordered_inputs


def _filter_rare_classes(
    features: np.ndarray,
    labels: np.ndarray,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels)
    keep_classes = np.where(counts >= min_samples)[0]
    mask = np.isin(labels, keep_classes)
    return features[mask], labels[mask]


def build_label_encoder(
    s008_lidar: np.ndarray,
    s008_coords: np.ndarray,
    s008_labels: np.ndarray,
    meta_train_ratio: float,
    data_seed: int,
) -> LabelEncoder:
    s008_labels = np.asarray(s008_labels, dtype=np.int64)
    _, _, _, _, y_meta, _ = train_test_split(
        s008_lidar,
        s008_coords,
        s008_labels,
        train_size=meta_train_ratio,
        random_state=data_seed,
        shuffle=True,
    )
    dummy_features = np.zeros((y_meta.shape[0], 1), dtype=np.float32)
    _, y_meta_filtered = _filter_rare_classes(dummy_features, y_meta)
    if y_meta_filtered.size == 0:
        raise ValueError("No labels remain for fitting the label encoder.")

    encoder = LabelEncoder()
    encoder.fit(y_meta_filtered)
    return encoder


def filter_known_labels(
    labels: np.ndarray,
    encoder: LabelEncoder,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    known_mask = np.isin(labels, encoder.classes_)
    if not np.all(known_mask):
        dropped = int((~known_mask).sum())
        missing = np.unique(labels[~known_mask])
        preview = ", ".join(str(int(v)) for v in missing[:10])
        suffix = "..." if missing.size > 10 else ""
        print(f"Dropping {dropped} sample(s) with unseen labels ({preview}{suffix}).")
    filtered_labels = labels[known_mask]
    filtered_arrays = [arr[known_mask] for arr in arrays]
    return filtered_labels, filtered_arrays


def stack_base_predictions(
    model_paths: List[Path],
    lidar_inputs: List[np.ndarray],
    coord_inputs: List[np.ndarray],
    batch_size: int,
) -> List[np.ndarray]:
    if len(lidar_inputs) != len(coord_inputs):
        raise ValueError("lidar_inputs and coord_inputs must have the same length.")

    stacked: List[List[np.ndarray]] = [[] for _ in lidar_inputs]
    for model_path in model_paths:
        model = load_base_model(model_path)
        for idx, (lidar, coords) in enumerate(zip(lidar_inputs, coord_inputs)):
            inputs = _prepare_inputs(model, lidar, coords)
            preds = model.predict(inputs, batch_size=batch_size, verbose=0)
            stacked[idx].append(preds)
        tf.keras.backend.clear_session()

    return [np.concatenate(pred_list, axis=1) for pred_list in stacked]


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
    xgb_model: XGBClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
    predictions = xgb_model.predict_proba(features)
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
    xgb_model: XGBClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
    predictions = xgb_model.predict_proba(features)
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


def main() -> None:
    xgb_model = load_xgb_model(XGB_MODEL_PATH)

    (
        s008_coord_input,
        s008_lidar_input,
        s008_labels,
        x_coord_s009_baseline,
        x_lidar_s009_baseline,
        y_s009_baseline,
    ) = load_dataset_sparse_labels(
        s008_path=str(BASELINE_S008_PATH),
        s009_path=str(BASELINE_S009_PATH),
    )

    coord_scaler = StandardScaler()
    coord_scaler.fit(s008_coord_input)

    label_encoder = build_label_encoder(
        s008_lidar_input,
        s008_coord_input,
        s008_labels,
        META_TRAIN_RATIO,
        DATA_SEED,
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

    x_coord_s009_baseline, x_coord_s009 = scale_coordinates(
        coord_scaler,
        x_coord_s009_baseline,
        x_coord_s009,
    )
    x_lidar_s009_baseline = np.asarray(x_lidar_s009_baseline, dtype=np.float32)
    x_lidar_s009 = np.asarray(x_lidar_s009, dtype=np.float32)

    y_s009_baseline, (
        x_coord_s009_baseline,
        x_lidar_s009_baseline,
    ) = filter_known_labels(
        np.asarray(y_s009_baseline, dtype=np.int64),
        label_encoder,
        x_coord_s009_baseline,
        x_lidar_s009_baseline,
    )
    if y_s009_baseline.size == 0:
        raise ValueError("No baseline samples remain after filtering unseen labels.")

    y_s009, (
        x_coord_s009,
        x_lidar_s009,
        condition_s009,
    ) = filter_known_labels(
        np.asarray(y_s009, dtype=np.int64),
        label_encoder,
        x_coord_s009,
        x_lidar_s009,
        np.asarray(condition_s009),
    )
    if y_s009.size == 0:
        raise ValueError("No raw samples remain after filtering unseen labels.")

    baseline_features, raw_features = stack_base_predictions(
        MODEL_PATHS,
        [x_lidar_s009_baseline, x_lidar_s009],
        [x_coord_s009_baseline, x_coord_s009],
        batch_size=BATCH_SIZE,
    )

    baseline_labels_encoded = label_encoder.transform(y_s009_baseline)
    raw_labels_encoded = label_encoder.transform(y_s009)

    baseline_metrics = evaluate_accuracy(
        name="Test on Baseline",
        xgb_model=xgb_model,
        features=baseline_features,
        labels=baseline_labels_encoded,
        top_k=TOP_K,
    )

    s009_metrics = evaluate_split(
        name="Test on Raw",
        xgb_model=xgb_model,
        features=raw_features,
        labels=raw_labels_encoded,
        conditions=condition_s009,
        top_k=TOP_K,
    )

    print_simple_accuracy(baseline_metrics)
    print_metrics(s009_metrics)


if __name__ == "__main__":
    main()
