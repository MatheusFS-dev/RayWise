"""Seed-stability evaluation aligned to the original Proposed training pipeline."""

import csv
import gc
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
THROUGHPUT_ROOT = SRC_ROOT / "evaluation" / "throughput"
for import_root in (SRC_ROOT, THROUGHPUT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluate_models_throughput import (  # noqa: E402
    _load_tensorflow_components,
    patch_lambda_globals,
    prepare_model_inputs,
)
from utils_beam_metrics import flatten_beam_grid  # noqa: E402


S008_ROOT = SRC_ROOT / "data" / "s008"
S009_ROOT = SRC_ROOT / "data" / "s009"
MODEL_PATH = (
    REPO_ROOT
    / "weights"
    / "nas_best_architectures_cnn1d"
    / "optuna_study"
    / "model"
    / "top_1_trial_4026.keras"
)
OUTPUT_DIR = SRC_ROOT / "evaluation" / "outputs" / "seed_stability_proposed"

DEFAULT_SEEDS = tuple(range(10))
DEFAULT_EPOCHS = 33
DEFAULT_BATCH_SIZE = 64
TOP1_LEARNING_RATE = 0.0028523343462769487


def set_all_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds.

    Args:
        seed (int): Non-negative seed used for all RNG sources.

    Returns:
        None: Updates process-level RNG states in place.

    Raises:
        ValueError: If ``seed`` is negative.
    """

    if seed < 0:
        raise ValueError(f"seed must be non-negative, but received {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    tf_module, _ = _load_tensorflow_components()
    tf_module.keras.utils.set_random_seed(seed)


def load_data_full_s008() -> dict[str, np.ndarray]:
    """Load baseline arrays and build the full-S008 training setup.

    Args:
        None: Paths are taken from module constants.

    Returns:
        dict[str, np.ndarray]: Full-S008 train arrays and S009 test arrays.

    Raises:
        FileNotFoundError: If required baseline files are missing.
        ValueError: If loaded arrays are misaligned.
    """

    paths = {
        "s008_coord_train": S008_ROOT / "coord_input" / "coord_train.npz",
        "s008_coord_val": S008_ROOT / "coord_input" / "coord_val.npz",
        "s008_lidar_train": S008_ROOT / "lidar_input" / "lidar_train.npz",
        "s008_lidar_val": S008_ROOT / "lidar_input" / "lidar_val.npz",
        "s008_beam_train": S008_ROOT / "beam_output" / "beam_output_train.npz",
        "s008_beam_val": S008_ROOT / "beam_output" / "beam_output_val.npz",
        "s009_coord": S009_ROOT / "coord_input" / "coord_input.npz",
        "s009_lidar": S009_ROOT / "lidar_input" / "lidar_input.npz",
        "s009_beam": S009_ROOT / "beam_output" / "beam_output.npz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset archive not found: {path}.")

    s008_coord_train = np.asarray(np.load(paths["s008_coord_train"], allow_pickle=False)["coordinates"], dtype=np.float32)
    s008_coord_val = np.asarray(np.load(paths["s008_coord_val"], allow_pickle=False)["coordinates"], dtype=np.float32)
    s008_lidar_train = np.asarray(np.load(paths["s008_lidar_train"], allow_pickle=False)["input"], dtype=np.float32)
    s008_lidar_val = np.asarray(np.load(paths["s008_lidar_val"], allow_pickle=False)["input"], dtype=np.float32)
    s008_beam_train = np.asarray(np.load(paths["s008_beam_train"], allow_pickle=False)["output_classification"])
    s008_beam_val = np.asarray(np.load(paths["s008_beam_val"], allow_pickle=False)["output_classification"])

    s009_coord = np.asarray(np.load(paths["s009_coord"], allow_pickle=False)["coordinates"], dtype=np.float32)
    s009_lidar = np.asarray(np.load(paths["s009_lidar"], allow_pickle=False)["input"], dtype=np.float32)
    s009_beam = np.asarray(np.load(paths["s009_beam"], allow_pickle=False)["output_classification"])

    s008_y_train = np.argmax(flatten_beam_grid(np.asarray(s008_beam_train.real, dtype=np.float32)), axis=1).astype(np.int64)
    s008_y_val = np.argmax(flatten_beam_grid(np.asarray(s008_beam_val.real, dtype=np.float32)), axis=1).astype(np.int64)
    s009_y = np.argmax(flatten_beam_grid(np.asarray(s009_beam.real, dtype=np.float32)), axis=1).astype(np.int64)

    s008_lidar_full = np.concatenate([s008_lidar_train, s008_lidar_val], axis=0).astype(np.float32)
    s008_coord_full = np.concatenate([s008_coord_train, s008_coord_val], axis=0).astype(np.float32)
    s008_y_full = np.concatenate([s008_y_train, s008_y_val], axis=0).astype(np.int64)

    if not (s008_lidar_full.shape[0] == s008_coord_full.shape[0] == s008_y_full.shape[0]):
        raise ValueError("Full S008 arrays are misaligned.")
    if not (s009_lidar.shape[0] == s009_coord.shape[0] == s009_y.shape[0]):
        raise ValueError("S009 arrays are misaligned.")

    return {
        "s008_lidar": s008_lidar_full,
        "s008_coord": s008_coord_full,
        "s008_y": s008_y_full,
        "s009_lidar": s009_lidar,
        "s009_coord": s009_coord,
        "s009_y": s009_y,
    }


def reseed_initializer_config(config: Any, seed: int) -> Any:
    """Recursively overwrite initializer seeds inside a Keras config object.

    Args:
        config (Any): Nested model/layer config object.
        seed (int): Seed assigned to initializer config entries.

    Returns:
        Any: Config with seeded initializer entries.
    """

    if isinstance(config, dict):
        updated = {key: reseed_initializer_config(value, seed) for key, value in config.items()}
        if "class_name" in updated and isinstance(updated.get("config"), dict):
            class_name = str(updated["class_name"])
            if "Initializer" in class_name or class_name in {
                "GlorotUniform",
                "GlorotNormal",
                "HeUniform",
                "HeNormal",
                "RandomUniform",
                "RandomNormal",
                "TruncatedNormal",
                "Orthogonal",
            }:
                if "seed" in updated["config"]:
                    updated["config"]["seed"] = int(seed)
        return updated
    if isinstance(config, list):
        return [reseed_initializer_config(item, seed) for item in config]
    return config


def compute_topk_accuracy(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute top-k accuracy for sparse labels.

    Args:
        labels (np.ndarray): Label vector with shape ``(N,)``.
        scores (np.ndarray): Score matrix with shape ``(N, C)``.
        k (int): Top-k cutoff.

    Returns:
        float: Top-k accuracy in ``[0, 1]``.

    Raises:
        ValueError: If shapes or ``k`` are invalid.
    """

    if labels.ndim != 1 or scores.ndim != 2 or scores.shape[0] != labels.shape[0]:
        raise ValueError("labels and scores shapes are inconsistent.")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be in [1, {scores.shape[1]}], received {k}.")

    topk_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean((topk_indices == labels[:, None]).any(axis=1).astype(np.float32)))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write ordered CSV rows.

    Args:
        path (Path): Destination path.
        rows (Iterable[dict[str, Any]]): Rows to write.
        fieldnames (list[str]): Header order.

    Returns:
        None: Writes file to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one_seed(data: dict[str, np.ndarray], seed: int, epochs: int, batch_size: int) -> dict[str, Any]:
    """Run one full-S008 Proposed training/evaluation with one seed.

    Args:
        data (dict[str, np.ndarray]): Dataset payload from ``load_data_full_s008``.
        seed (int): Run seed controlling all RNGs.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        dict[str, Any]: Row containing ``seed``, ``top1``, and ``top10``.

    Raises:
        FileNotFoundError: If the template model is missing.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}.")

    set_all_random_seeds(seed)
    tf_module, custom_objects = _load_tensorflow_components()

    scaler_mean = np.mean(data["s008_coord"], axis=0, dtype=np.float64)
    scaler_std = np.std(data["s008_coord"], axis=0, dtype=np.float64)
    scaler_std = np.where(scaler_std == 0.0, 1.0, scaler_std)

    s008_coord = ((data["s008_coord"] - scaler_mean) / scaler_std).astype(np.float32)
    s009_coord = ((data["s009_coord"] - scaler_mean) / scaler_std).astype(np.float32)

    template_model = tf_module.keras.models.load_model(
        str(MODEL_PATH),
        safe_mode=False,
        compile=False,
        custom_objects=custom_objects,
    )
    patch_lambda_globals(model=template_model, tf_module=tf_module)

    seeded_config = reseed_initializer_config(template_model.get_config(), seed)
    model = tf_module.keras.Model.from_config(seeded_config, custom_objects=custom_objects)
    patch_lambda_globals(model=model, tf_module=tf_module)

    model.compile(
        optimizer=tf_module.keras.optimizers.Adam(learning_rate=TOP1_LEARNING_RATE),
        loss=tf_module.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf_module.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_1_accuracy")],
        jit_compile=False,
    )

    # Match the final full-S008 training script: pass arrays directly so Keras
    # owns epoch-level shuffling after the run seed is set.
    model.fit(
        x=(data["s008_lidar"], s008_coord),
        y=data["s008_y"],
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tf_module.keras.callbacks.TerminateOnNaN()],
        verbose=2,
    )

    test_inputs = prepare_model_inputs(model=model, lidar=data["s009_lidar"], coordinates=s009_coord)
    scores = np.asarray(model.predict(test_inputs, batch_size=batch_size, verbose=2), dtype=np.float32)

    top1 = compute_topk_accuracy(data["s009_y"], scores, k=1)
    top10 = compute_topk_accuracy(data["s009_y"], scores, k=10)

    tf_module.keras.backend.clear_session()
    gc.collect()

    return {"seed": int(seed), "top1": float(top1), "top10": float(top10)}


def evaluate_proposed_seed_stability(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float]:
    """Run 10-seed stability evaluation aligned to Proposed original training.

    Args:
        seeds (Sequence[int]): Distinct run seeds.
        epochs (int): Training epochs per seed.
        batch_size (int): Batch size for fit and predict.

    Returns:
        dict[str, float]: Summary containing top1/top10 mean and std.

    Raises:
        ValueError: If ``seeds`` is empty or has duplicates.
    """

    validated_seeds = tuple(int(seed) for seed in seeds)
    if not validated_seeds:
        raise ValueError("seeds must contain at least one seed.")
    if len(set(validated_seeds)) != len(validated_seeds):
        raise ValueError("seeds must be unique.")

    data = load_data_full_s008()
    rows = [run_one_seed(data, seed, epochs, batch_size) for seed in validated_seeds]

    top1_values = np.asarray([row["top1"] for row in rows], dtype=np.float64)
    top10_values = np.asarray([row["top10"] for row in rows], dtype=np.float64)
    summary = {
        "top1_mean": float(np.mean(top1_values)),
        "top1_std": float(np.std(top1_values, ddof=0)),
        "top10_mean": float(np.mean(top10_values)),
        "top10_std": float(np.std(top10_values, ddof=0)),
    }

    write_csv(
        OUTPUT_DIR / "proposed_seed_runs.csv",
        [{"model_name": "Proposed", **row} for row in rows],
        ["model_name", "seed", "top1", "top10"],
    )
    write_csv(
        OUTPUT_DIR / "proposed_seed_summary.csv",
        [{"model_name": "Proposed", "num_seeds": len(validated_seeds), **summary}],
        ["model_name", "num_seeds", "top1_mean", "top1_std", "top10_mean", "top10_std"],
    )
    return summary


if __name__ == "__main__":
    metrics = evaluate_proposed_seed_stability()
    print("Proposed")
    print(f"  Top-1  mean={metrics['top1_mean']:.6f} std={metrics['top1_std']:.6f}")
    print(f"  Top-10 mean={metrics['top10_mean']:.6f} std={metrics['top10_std']:.6f}")
