"""Load Raymobtime raw datasets, split S008, and prepare ensemble predictions."""

import os

# Specify GPU to use (e.g., GPU:0, CPU:-1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import optuna
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xgboost import XGBClassifier
import xgboost as xgb

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

    label_encoder = LabelEncoder()
    label_encoder.fit(labels_s008)
    labels_train_enc = label_encoder.transform(labels_train)
    labels_val_enc = label_encoder.transform(labels_val)
    num_classes = len(label_encoder.classes_)

    meta_train = stack_meta_features(train_meta_preds)
    meta_val = stack_meta_features(val_meta_preds)

    print("\nRunning Optuna study for XGBoost meta-learner (objective: val mlogloss)")
    study_storage = f"sqlite:///{(run_dir / 'optuna_study.db').resolve()}"
    study_name = f"{Path(__file__).stem}_xgb_meta"
    sampler = optuna.samplers.TPESampler(seed=DATA_SEED)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=study_storage,
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 5e-2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.5, 6.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e-1, log=True),
        }
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            sampling_method="gradient_based",
            tree_method="hist",
            device="cuda",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=DATA_SEED,
            n_jobs=-1,
            verbosity=1,
            early_stopping_rounds=20,
            **params,
        )

        model.fit(
            meta_train,
            labels_train_enc,
            eval_set=[(meta_val, labels_val_enc)],
            verbose=True,
        )

        booster = model.get_booster()
        best_score = getattr(booster, "best_score", None)
        if best_score is None:
            evals_result = model.evals_result()
            scores = evals_result.get("validation_0", {}).get("mlogloss", [])
            best_idx = getattr(model, "best_iteration", None)
            if best_idx is not None and best_idx < len(scores):
                best_score = scores[best_idx]
            elif scores:
                best_score = scores[-1]
            else:
                best_score = float("inf")
        return float(best_score)

    study.optimize(objective, n_trials=1000)
    best_params = study.best_trial.params
    best_value = float(study.best_value)
    print(f"  Best trial #{study.best_trial.number} -> mlogloss={best_value:.6f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")

    best_params_path = run_dir / "best_params.json"
    best_params_path.write_text(json.dumps(best_params, indent=2))
    print(f"Saved best params to {best_params_path}")

    print("\nTraining XGBoost meta-learner with best Optuna params:")
    xgb_meta = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        sampling_method="gradient_based",
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=DATA_SEED,
        verbosity=2,
        early_stopping_rounds=20,
        **best_params,
    )

    xgb_meta.fit(
        meta_train,
        labels_train_enc,
        eval_set=[(meta_val, labels_val_enc)],
        verbose=True,
    )

    booster = xgb_meta.get_booster()
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None and best_iter >= 0:
        print(f"  → Best XGBoost iteration: {best_iter}")

    val_probs = xgb_meta.predict_proba(meta_val)
    val_topk = compute_topk_accuracy(val_probs, labels_val_enc)
    val_lines = ", ".join(f"Top-{k}: {val_topk[k]*100:.2f}%" for k in TOP_K)
    print(f"  S008 val (meta) -> {val_lines}")
    train_probs = xgb_meta.predict_proba(meta_train)
    train_topk = compute_topk_accuracy(train_probs, labels_train_enc)
    train_lines = ", ".join(f"Top-{k}: {train_topk[k]*100:.2f}%" for k in TOP_K)
    print(f"  S008 train (meta) -> {train_lines}")

    meta_test = stack_meta_features(predictions)
    s009_mask = np.isin(y_test, label_encoder.classes_)
    if not s009_mask.all():
        unseen = np.setdiff1d(np.unique(y_test), label_encoder.classes_)
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
        y_test_enc = label_encoder.transform(y_test_filtered)
        test_probs = xgb_meta.predict_proba(meta_test)
        test_topk = compute_topk_accuracy(test_probs, y_test_enc)
        test_lines = ", ".join(f"Top-{k}: {test_topk[k]*100:.2f}%" for k in TOP_K)
        print(f"  S009 test (meta) -> {test_lines}")

    # ———————————————————————————————————————————————————————————————————————————— #
    #                              Persist artifacts                               #
    # ———————————————————————————————————————————————————————————————————————————— #

    print("Saving XGBoost model and results.")
    model_path = run_dir / "xgb_meta_model.json"
    xgb_meta.save_model(str(model_path))

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
        "classes": [int(c) for c in label_encoder.classes_],
        "optuna": {
            "study_name": study_name,
            "storage": study_storage,
            "best_value": best_value,
            "best_params": best_params,
            "best_params_path": str(best_params_path),
        },
        "xgb_params": xgb_meta.get_params(),
        "artifacts": {
            "xgb_model": str(model_path),
            "optuna_study": str(run_dir / "optuna_study.db"),
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
