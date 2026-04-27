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
RESULTS_DIR = Path("runs") / Path(__file__).stem


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

    models = [load_model(p) for p in MODEL_PATHS]
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
                "model_path": str(MODEL_PATHS[idx - 1]),
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
        expected_models=len(MODEL_PATHS),
    )
    aligned_val = align_meta_features(
        meta_val,
        classes=classes,
        expected_models=len(MODEL_PATHS),
    )

    train_df = meta_features_to_dataframe(
        aligned_train,
        labels=labels_train_enc,
        classes=classes,
        label_column="label",
        expected_models=len(MODEL_PATHS),
    )
    val_df = meta_features_to_dataframe(
        aligned_val,
        labels=labels_val_enc,
        classes=classes,
        label_column="label",
        expected_models=len(MODEL_PATHS),
    )

    del models
    tf.keras.backend.clear_session()
    gc.collect()

    print("\nTraining AutoGluon stacked meta-learner on S008 meta-features:")
    predictor_path = RESULTS_DIR / f"autogluon_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    presets = "medium_quality_faster_train"
    predictor = TabularPredictor(
        label="label",
        path=str(predictor_path),
        problem_type="multiclass",
        eval_metric="log_loss",
    ).fit(
        train_data=train_df,
        presets=presets,
        num_stack_levels=1,
        num_bag_folds=5,
        num_bag_sets=1,
        use_bag_holdout=True,
        fit_strategy="sequential",
        # num_gpus=1,
        ag_args_ensemble={
            "fold_fitting_strategy": "sequential_local",
            "num_folds_parallel": 1,
        },
        num_cpus=12,
        verbosity=2,
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
            expected_models=len(MODEL_PATHS),
        )
        test_df = meta_features_to_dataframe(
            aligned_test,
            labels=None,
            classes=classes,
            label_column="label",
            expected_models=len(MODEL_PATHS),
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
            "model_paths": [str(p) for p in MODEL_PATHS],
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
        "autogluon_path": str(model_path),
        "artifacts": {
            "autogluon_predictor": str(model_path),
        },
    }

    results_path = run_dir / "results.json"
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


main()
