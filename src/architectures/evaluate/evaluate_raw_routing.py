import pickle
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

ROUTING_MODE = "hard"  # "hard" or "soft"
CLASSIFIER_THRESHOLD = 0.5
CLASSIFIER_NLOS_INDEX = 1

CLASSIFIER_MODEL_PATH = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_classifier/nas_classifier_NLOS_LOS_v0.0/optuna_study/model/top_1_trial_535.keras"
)
CLASSIFIER_SCALER_PATH = Path("/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_classifier/nas_classifier_NLOS_LOS_v0.0/optuna_study/scaler/top_1_trial_535.pkl")
BEAM_MODEL_LOS_PATH = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_LOS/optuna_study/model/top_1_trial_1160.keras"
)
BEAM_MODEL_NLOS_PATH = Path(
    "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_NLOS/optuna_study/model/top_1_trial_188.keras"
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


@tf.keras.utils.register_keras_serializable(package="RayWise")
class LidarChannelOneHot(tf.keras.layers.Layer):
    def __init__(self, offset: int = 2, depth: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset
        self.depth = depth

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        indices = tf.cast(inputs + self.offset, tf.int32)
        return tf.one_hot(indices, depth=self.depth, dtype=tf.float32)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape(tuple(input_shape) + (self.depth,))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"offset": self.offset, "depth": self.depth})
        return config


def load_beam_model(model_path: Path) -> tf.keras.Model:
    # The saved architecture includes Lambda layers; disable Keras safe_mode to allow loading them.
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects={"tf": tf},
    )
    return model


def load_classifier(model_path: Path) -> tf.keras.Model:
    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=True,
        compile=False,
        custom_objects={"LidarChannelOneHot": LidarChannelOneHot},
    )
    return model


def load_scaler(path: Path) -> StandardScaler:
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


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


def evaluate_split_predictions(
    *,
    name: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    top_k: tuple[int, ...] = TOP_K,
) -> Dict[str, float]:
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


def _beam_output_dim(model: tf.keras.Model) -> int:
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    if output_shape is None or len(output_shape) < 2 or output_shape[-1] is None:
        raise ValueError(f"Unexpected beam model output shape: {model.output_shape}")
    return int(output_shape[-1])


def predict_with_routing(
    *,
    classifier: tf.keras.Model,
    beam_los: tf.keras.Model,
    beam_nlos: tf.keras.Model,
    lidar: np.ndarray,
    coords_beam: np.ndarray,
    coords_classifier: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    clf_inputs = _prepare_inputs(classifier, lidar, coords_classifier)
    clf_probs = classifier.predict(clf_inputs, batch_size=batch_size, verbose=0)
    nlos_probs = clf_probs[:, CLASSIFIER_NLOS_INDEX]

    if ROUTING_MODE == "soft":
        los_inputs = _prepare_inputs(beam_los, lidar, coords_beam)
        nlos_inputs = _prepare_inputs(beam_nlos, lidar, coords_beam)
        los_preds = beam_los.predict(los_inputs, batch_size=batch_size, verbose=0)
        nlos_preds = beam_nlos.predict(nlos_inputs, batch_size=batch_size, verbose=0)
        mix = nlos_probs[:, None].astype(np.float32)
        return los_preds * (1.0 - mix) + nlos_preds * mix

    if ROUTING_MODE != "hard":
        raise ValueError(f"Unknown ROUTING_MODE: {ROUTING_MODE}")

    num_classes = _beam_output_dim(beam_los)
    routed = np.zeros((lidar.shape[0], num_classes), dtype=np.float32)

    nlos_mask = nlos_probs >= CLASSIFIER_THRESHOLD
    los_mask = ~nlos_mask

    if los_mask.any():
        los_inputs = _prepare_inputs(beam_los, lidar[los_mask], coords_beam[los_mask])
        routed[los_mask] = beam_los.predict(los_inputs, batch_size=batch_size, verbose=0)

    if nlos_mask.any():
        nlos_inputs = _prepare_inputs(beam_nlos, lidar[nlos_mask], coords_beam[nlos_mask])
        routed[nlos_mask] = beam_nlos.predict(nlos_inputs, batch_size=batch_size, verbose=0)

    return routed


def main() -> None:
    classifier = load_classifier(CLASSIFIER_MODEL_PATH)
    beam_los = load_beam_model(BEAM_MODEL_LOS_PATH)
    beam_nlos = load_beam_model(BEAM_MODEL_NLOS_PATH)

    patch_lambda_globals(beam_los)
    patch_lambda_globals(beam_nlos)

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

    beam_coord_scaler = StandardScaler()
    beam_coord_scaler.fit(s008_coord_input)
    classifier_coord_scaler = load_scaler(CLASSIFIER_SCALER_PATH)

    x_lidar_s009, x_coord_s009, y_s009, condition_s009 = load_dataset_raw_sparse_labels_s009_only(
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=DATA_SEED,
        report_label_coverage=False,
        remove_null_labels=False,
        return_conditions=True,
    )

    x_coord_s009_baseline_beam = scale_coordinates(beam_coord_scaler, x_coord_s009_baseline)[0]
    x_coord_s009_beam = scale_coordinates(beam_coord_scaler, x_coord_s009)[0]
    x_coord_s009_baseline_clf = scale_coordinates(classifier_coord_scaler, x_coord_s009_baseline)[0]
    x_coord_s009_clf = scale_coordinates(classifier_coord_scaler, x_coord_s009)[0]

    baseline_predictions = predict_with_routing(
        classifier=classifier,
        beam_los=beam_los,
        beam_nlos=beam_nlos,
        lidar=x_lidar_s009_baseline,
        coords_beam=x_coord_s009_baseline_beam,
        coords_classifier=x_coord_s009_baseline_clf,
        batch_size=BATCH_SIZE,
    )

    baseline_metrics = evaluate_accuracy_predictions(
        name="Test on Baseline",
        predictions=baseline_predictions,
        labels=y_s009_baseline,
    )

    raw_predictions = predict_with_routing(
        classifier=classifier,
        beam_los=beam_los,
        beam_nlos=beam_nlos,
        lidar=x_lidar_s009,
        coords_beam=x_coord_s009_beam,
        coords_classifier=x_coord_s009_clf,
        batch_size=BATCH_SIZE,
    )

    s009_metrics = evaluate_split_predictions(
        name="Test on Raw",
        predictions=raw_predictions,
        labels=y_s009,
        conditions=condition_s009,
    )

    print_simple_accuracy(baseline_metrics)
    print_metrics(s009_metrics)


if __name__ == "__main__":
    main()
