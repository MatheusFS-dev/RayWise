"""Train a V7 student with top-X teacher ensemble knowledge distillation.

This script keeps the same student architecture used in ``train_v7_top-k.py``
and replaces stacking with teacher-ensemble distillation.
"""

import gc
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from _imports import *

from _load_dataset_raw import (
    load_dataset_raw_sparse_labels_s008_only,
    load_dataset_raw_sparse_labels_s009_only,
)

# Specify GPU to use (e.g., GPU:0, CPU:-1).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

TOP_K = (1, 3, 5, 10, 20, 30, 50)

DATA_SEED = 0
TRAIN_SEED = 0
TRAIN_RATIO = 0.8

EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.0028 # 0.0028523343462769487
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 8

# Distillation hyperparameters.
# `DISTILL_ALPHA` balances hard labels vs teacher soft targets in total loss:
#   total_loss = alpha * hard_ce + (1 - alpha) * kd_kl
# - If alpha is HIGH (closer to 1.0):
#   - Training behaves more like standard supervised learning on S008 labels.
#   - The student is less constrained by teacher probabilities, which can help
#     when teachers are noisy or biased.
#   - Usually improves robustness to bad teacher calibration, but can reduce
#     transfer of teacher class-similarity information.
# - If alpha is LOW (closer to 0.0):
#   - Training emphasizes teacher imitation more strongly.
#   - Can improve ranking/top-k behavior when teachers encode useful dark
#     knowledge, but may also copy teacher errors.
#   - Can become sensitive to teacher quality and teacher ensemble selection.
# Current value (0.2) means 20% hard-label supervision and 80% KD supervision.
DISTILL_ALPHA = 0.5

# `DISTILL_TEMPERATURE` controls how soft the probability distributions are in
# the KL branch (applied to BOTH teacher and student probabilities):
# - If temperature is HIGH (> 1):
#   - Distributions become flatter, exposing relative probabilities among
#     non-top classes (dark knowledge).
#   - Can improve generalization and calibration transfer, but too high may
#     over-smooth targets and weaken class discrimination.
# - If temperature is LOW (near 1):
#   - Distributions are sharper and closer to argmax behavior.
#   - Stronger focus on top class; less signal from secondary classes.
#   - Too low can make KD behave similarly to hard-label training.
# Note: KD loss is scaled by temperature^2 in `train_step`, which keeps gradient
# magnitudes more stable across different temperature values.
DISTILL_TEMPERATURE = 5


from araras.utils.io import select_path

# Teacher model loading configuration.
TOP_X_MODELS = 3
# Auto-discovery source: when EXPLICIT_TEACHER_PATHS is empty, the script scans
# this directory for .keras files, prioritizes names like top_1_trial_*.keras,
# and then selects up to TOP_X_MODELS teachers.
TEACHER_MODELS_DIR = Path(
    select_path(
        select_dir=True,
        description="Select directory containing teacher .keras models for distillation.",
    )
    # "/media/matheus/SSD-2/matheus/results/raywise/nas_cnn1d_v7.0/optuna_study/model"
    # "/media/matheus/SSD-2/matheus/results/raywise/nas_v7_seeds_val_acc_topk_1/optuna_study/model"
)
# Manual override source: if this list is non-empty, these exact checkpoints
# are used first (in listed order) and directory discovery is skipped.
# The final number of teachers still respects TOP_X_MODELS.
EXPLICIT_TEACHER_PATHS: list[Path] = []

DATA_ROOT = Path("/media/matheus/SSD-2/matheus/datasets/RayWise")

S008_COORD_CSV = DATA_ROOT / "Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
S008_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s008/processed_raw_data/lidar_data_s008"
S008_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s008/baseline_data/beam_output/beams_output_s008.npz"

S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

RESULTS_DIR = Path("runs") / Path(__file__).stem

# ———————————————————————————————————————————————————————————————————————————— #
#                            Reproducibility Setup                             #
# ———————————————————————————————————————————————————————————————————————————— #
# Set deterministic seeds for Python, NumPy, and TensorFlow.
set_random_seed(TRAIN_SEED)
tf.config.experimental.enable_op_determinism()

POLICY = mixed_precision.Policy("float32")
mixed_precision.set_global_policy(POLICY)


# ———————————————————————————————————————————————————————————————————————————— #
#                                Model Layers                                  #
# ———————————————————————————————————————————————————————————————————————————— #
@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(layers.Layer):
    """Convert raw LiDAR tensors to a 4-channel semantic one-hot tensor.

    Returns:
        tf.Tensor: A float32 tensor with shape ``(batch, 20, 200, 4)`` where
        channels encode base station, vehicle, obstacle, and free space.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic conversion from dense LiDAR encoding to one-hot maps.

        Args:
            inputs (tf.Tensor): LiDAR tensor of shape ``(batch, 20, 200, 10)``
                with values in ``{-2, -1, 0, 1}`` representing scene elements.

        Returns:
            tf.Tensor: Converted tensor of shape ``(batch, 20, 200, 4)``.

        Raises:
            tf.errors.InvalidArgumentError: If TensorFlow receives an invalid
                tensor shape or dtype during graph execution.
        """

        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer configuration.

        Returns:
            dict[str, Any]: Keras configuration dictionary.
        """

        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(layers.Layer):
    """Tile 2D coordinates across the flattened LiDAR token sequence.

    Args:
        grid_length (int): Number of flattened LiDAR positions. Typical value
            is ``20 * 200 = 4000`` for the V7 model.
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        """Initialize the coordinate tiling layer.

        Args:
            grid_length (int): Number of sequence positions to broadcast each
                coordinate pair over.
            **kwargs (Any): Additional Keras layer kwargs.
        """

        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Tile coordinates to match sequence length expected by Conv1D blocks.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tensor with shape ``(batch, grid_length, 2)``.

        Raises:
            tf.errors.InvalidArgumentError: If TensorFlow cannot broadcast or
                tile because of incompatible tensor rank.
        """

        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer configuration.

        Returns:
            dict[str, Any]: Keras configuration dictionary including
            ``grid_length``.
        """

        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


# ———————————————————————————————————————————————————————————————————————————— #
#                           Distillation Wrapper                               #
# ———————————————————————————————————————————————————————————————————————————— #
class Distiller(tf.keras.Model):
    """Keras model wrapper that trains a student from hard and soft targets.

    The wrapper expects batched targets as ``(hard_labels, teacher_probs)`` and
    combines sparse categorical cross-entropy with KL divergence distillation.
    """

    def __init__(self, student: tf.keras.Model, alpha: float, temperature: float) -> None:
        """Initialize distillation wrapper.

        Args:
            student (tf.keras.Model): Student model to optimize.
            alpha (float): Weight of the hard-label loss branch. If closer to
                ``1.0``, training behaves more like regular supervised learning.
                If closer to ``0.0``, training prioritizes teacher imitation and
                may transfer teacher biases more strongly.
            temperature (float): Softening temperature for teacher and student
                probability distributions. Higher values produce smoother soft
                targets (more class similarity information) and lower values
                keep sharper distributions (closer to argmax behavior).

        Raises:
            ValueError: If ``alpha`` is outside ``[0.0, 1.0]`` or if
                ``temperature <= 0``.
        """

        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0.0, 1.0].")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        super().__init__()
        self.student = student
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.hard_loss_fn = losses.SparseCategoricalCrossentropy()
        self.soft_loss_fn = losses.KLDivergence()

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        """Forward pass delegated to the student.

        Args:
            inputs (Any): Student model inputs.
            training (bool): If ``True``, enables training-time behavior such as
                dropout and batchnorm updates. If ``False``, runs inference mode.

        Returns:
            tf.Tensor: Student class probabilities.
        """

        return self.student(inputs, training=training)

    def _collect_metric_logs(self) -> dict[str, tf.Tensor]:
        """Collect scalar metric logs, flattening any nested metric containers.

        Returns:
            dict[str, tf.Tensor]: Mapping of metric names to scalar metric
            tensors. Container metrics (for example, Keras compile metric
            wrappers) are expanded so callback monitors can reference concrete
            names such as ``top_1_accuracy``.
        """

        logs: dict[str, tf.Tensor] = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                logs.update(result)
            else:
                logs[metric.name] = result
        return logs

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Run one distillation training step.

        Args:
            data (Any): Keras batch tuple ``(x_batch, (y_hard, y_teacher_soft))``
                where ``y_hard`` is integer sparse labels and ``y_teacher_soft``
                is a probability matrix over classes.

        Returns:
            dict[str, tf.Tensor]: Metric dictionary including total loss,
            hard-loss branch, and distillation-loss branch.

        Raises:
            ValueError: If the incoming target tuple does not contain exactly
                two elements.

        Examples:
            ``distiller.fit(x=train_inputs, y=(labels, teacher_probs), epochs=5)``
        """

        x_batch, y_batch = data
        if not isinstance(y_batch, tuple) or len(y_batch) != 2:
            raise ValueError("Distiller expects targets as (y_hard, y_teacher_soft).")
        y_hard, y_teacher = y_batch

        # Compute student outputs and both loss branches in a single tape pass.
        with tf.GradientTape() as tape:
            y_pred = self.student(x_batch, training=True)
            hard_loss = self.hard_loss_fn(y_hard, y_pred)

            # Re-temperature probabilities via log-space to avoid requiring logits.
            eps = tf.keras.backend.epsilon()
            student_soft = tf.nn.softmax(tf.math.log(tf.clip_by_value(y_pred, eps, 1.0)) / self.temperature, axis=-1)
            teacher_soft = tf.nn.softmax(tf.math.log(tf.clip_by_value(y_teacher, eps, 1.0)) / self.temperature, axis=-1)
            distill_loss = self.soft_loss_fn(teacher_soft, student_soft) * (self.temperature**2)

            # Blend supervised learning and teacher imitation according to alpha.
            total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss

        # Optimize only the student parameters; teacher probabilities are fixed targets.
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Update metrics via the compiled container, but collect values from
        # self.metrics for compatibility across Keras versions.
        self.compiled_metrics.update_state(y_hard, y_pred)
        metric_logs = self._collect_metric_logs()
        metric_logs.update(
            {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "distill_loss": distill_loss,
            }
        )
        return metric_logs

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Run one evaluation step against hard labels only.

        Args:
            data (Any): Keras batch tuple ``(x_batch, y_hard)``.

        Returns:
            dict[str, tf.Tensor]: Evaluation metrics dictionary.
        """

        x_batch, y_hard = data
        y_pred = self.student(x_batch, training=False)
        hard_loss = self.hard_loss_fn(y_hard, y_pred)
        self.compiled_metrics.update_state(y_hard, y_pred)
        metric_logs = self._collect_metric_logs()
        metric_logs["loss"] = hard_loss
        return metric_logs


# ———————————————————————————————————————————————————————————————————————————— #
#                           Student Architecture                               #
# ———————————————————————————————————————————————————————————————————————————— #
def build_v7_student_model(*, show_summary: bool = True) -> Model:
    """Build the same V7 student architecture used in ``train_v7_top-k.py``.

    Args:
        show_summary (bool): If ``True``, prints model summary. If ``False`,
            skips summary output for cleaner logs and slightly lower startup time.

    Returns:
        Model: Uncompiled student model that outputs 256-way class probabilities.

    Examples:
        ``student = build_v7_student_model(show_summary=False)``
    """

    initializer = tf.keras.initializers.GlorotUniform(seed=TRAIN_SEED)

    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")
    one_hot_lidar = LidarToOneHot(name="lidar_transform_to_one_hot")(x_lidar_input)
    x_lidar_flat = layers.Reshape((20 * 200, 4), name="lidar_flatten_4_channels")(one_hot_lidar)

    x_coord_input = layers.Input(shape=(2,), name="coord_input")
    x_coord = CoordTileFlat(20 * 200, name="coord_tile_flat")(x_coord_input)

    # Fuse per-token LiDAR semantics with repeated coordinate channels.
    combined = layers.Concatenate(axis=-1, name="combine_lidar_coord")([x_lidar_flat, x_coord])

    x = layers.Conv1D(
        filters=128,
        kernel_size=9,
        padding="same",
        kernel_initializer=initializer,
        name="conv1d_0",
    )(combined)
    x = layers.BatchNormalization(name="conv1d_0_bn")(x)
    x = layers.Activation("silu", name="conv1d_0_act")(x)
    x = layers.MaxPooling1D(pool_size=4, name="max_pool_0")(x)

    x = layers.Conv1D(
        filters=256,
        kernel_size=4,
        padding="same",
        kernel_initializer=initializer,
        name="conv1d_1",
    )(x)
    x = layers.BatchNormalization(name="conv1d_1_bn")(x)
    x = layers.Activation("tanh", name="conv1d_1_act")(x)
    x = layers.MaxPooling1D(pool_size=2, name="max_pool_1")(x)

    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = layers.Dense(175, kernel_initializer=initializer, name="dense_0")(x)
    x = layers.Activation("tanh", name="dense_0_act")(x)
    x = layers.Dropout(0.1, name="dense_0_dropout")(x)

    outputs = layers.Dense(256, activation="softmax", kernel_initializer=initializer, name="output")(x)
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))

    if show_summary:
        model.summary()
    return model


# ———————————————————————————————————————————————————————————————————————————— #
#                              Utility Functions                               #
# ———————————————————————————————————————————————————————————————————————————— #
def compute_topk_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = TOP_K,
) -> dict[int, float]:
    """Compute top-k accuracies for probabilistic multi-class predictions.

    Args:
        predictions (np.ndarray): Probability matrix with shape ``(N, C)``.
        labels (np.ndarray): Integer sparse labels with shape ``(N,)``.
        ks (tuple[int, ...]): Requested top-k values. Larger values than
            ``C`` are clipped to ``C``.

    Returns:
        dict[int, float]: Mapping from each requested ``k`` to accuracy in
        ``[0.0, 1.0]``.

    Raises:
        ValueError: If predictions are not 2D or labels are not 1D.
    """

    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions shape (N, C), got {predictions.shape}.")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1-D array.")

    if labels.shape[0] == 0:
        return {k: 0.0 for k in ks}

    sorted_indices = np.argsort(predictions, axis=1)[:, ::-1]
    accuracies: dict[int, float] = {}
    for k in ks:
        # Clip requested k to available classes to avoid index errors.
        k_eff = min(k, sorted_indices.shape[1])
        hits = (sorted_indices[:, :k_eff] == labels[:, None]).any(axis=1)
        accuracies[k] = float(np.mean(hits))
    return accuracies


def _filter_rare_classes(
    lidar: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop samples from classes that are too rare for stable splitting/training.

    Args:
        lidar (np.ndarray): LiDAR feature tensor.
        coords (np.ndarray): Coordinate feature tensor.
        labels (np.ndarray): Sparse integer labels.
        min_samples (int): Minimum samples required per class to keep it.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered ``(lidar, coords,
        labels)`` arrays with matching first dimension.
    """

    counts = np.bincount(labels)
    keep_classes = np.where(counts >= min_samples)[0]
    # Apply one shared boolean mask so all feature/label arrays stay aligned.
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
    """Return labels for stratification when all classes are sufficiently populated.

    Args:
        labels (np.ndarray): Sparse integer labels.
        min_count (int): Minimum count per class required to stratify.

    Returns:
        np.ndarray | None: Original labels if stratification is safe, otherwise
        ``None`` to force random split.
    """

    counts = np.bincount(labels)
    positive = counts[counts > 0]
    if positive.size == 0:
        return None
    min_positive = int(positive.min())
    if min_positive < min_count:
        print(
            f"Warning: cannot stratify split because the rarest class has {min_positive} sample(s); "
            "falling back to random split."
        )
        return None
    return labels


def resolve_teacher_model_paths(
    *,
    explicit_paths: Sequence[Path],
    models_dir: Path,
    top_x: int,
) -> list[Path]:
    """Resolve teacher model paths from explicit list or top-ranked filenames.

    Args:
        explicit_paths (Sequence[Path]): Explicit teacher model paths. If this
            sequence is non-empty, only these paths are used and ``models_dir``
            discovery is skipped.
        models_dir (Path): Directory containing saved teacher ``.keras`` files.
            Used only when ``explicit_paths`` is empty.
        top_x (int): Number of teacher models to load. If fewer candidates are
            available, all available models are returned.

    Returns:
        list[Path]: Resolved teacher model paths in deterministic order.

    Raises:
        ValueError: If ``top_x`` is not positive.
        FileNotFoundError: If no teacher model file can be resolved.

    Examples:
        ``resolve_teacher_model_paths(explicit_paths=[], models_dir=Path("models"), top_x=3)``
    """

    if top_x <= 0:
        raise ValueError("top_x must be a positive integer.")

    if explicit_paths:
        # Explicit paths override auto-discovery to support exact reproducibility.
        missing = [p for p in explicit_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing explicit teacher model(s): {missing}")
        return list(explicit_paths[:top_x])

    if not models_dir.exists():
        raise FileNotFoundError(f"Teacher model directory does not exist: {models_dir}")

    candidates = list(models_dir.glob("*.keras"))
    if not candidates:
        raise FileNotFoundError(f"No .keras models found in: {models_dir}")

    # Prefer NAS-ranked files named like top_1_trial_*.keras so model quality
    # ordering follows Optuna ranking when available.
    ranked: list[tuple[int, Path]] = []
    unranked: list[Path] = []
    rank_pattern = re.compile(r"top_(\d+)_trial_.*\.keras$")
    for path in candidates:
        match = rank_pattern.match(path.name)
        if match:
            ranked.append((int(match.group(1)), path))
        else:
            unranked.append(path)

    ranked_sorted = [path for _, path in sorted(ranked, key=lambda item: item[0])]
    fallback_sorted = sorted(unranked)
    selected = (ranked_sorted + fallback_sorted)[:top_x]

    if not selected:
        raise FileNotFoundError(f"Unable to resolve teacher models from: {models_dir}")
    return selected


def patch_lambda_globals(model: tf.keras.Model) -> None:
    """Patch Lambda layer globals so custom deserialization works reliably.

    Args:
        model (tf.keras.Model): Loaded model potentially containing Lambda
            layers referencing missing global symbols.

    Returns:
        None: The function mutates Lambda function globals in-place.
    """

    # Recursively walk nested models so Lambda patches are applied everywhere.
    def walk(module: tf.keras.Model):
        for layer in module.layers:
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
    """Load a teacher model from disk with Lambda compatibility patches.

    Args:
        model_path (Path): Path to a saved ``.keras`` model.

    Returns:
        tf.keras.Model: Loaded model in inference-ready mode.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Teacher model file does not exist: {model_path}")

    model = tf.keras.models.load_model(
        str(model_path),
        safe_mode=False,
        compile=False,
        custom_objects={"tf": tf},
    )
    patch_lambda_globals(model)
    return model


def _prepare_inputs(model: tf.keras.Model, lidar: np.ndarray, coords: np.ndarray) -> np.ndarray | list[np.ndarray]:
    """Map provided LiDAR and coordinate arrays to model input signature.

    Args:
        model (tf.keras.Model): Teacher model whose input tensors determine
            input ordering and assignment.
        lidar (np.ndarray): LiDAR tensor with shape compatible with LiDAR input.
        coords (np.ndarray): Scaled coordinate tensor with shape compatible with
            coordinate input.

    Returns:
        np.ndarray | list[np.ndarray]: Single array for single-input models, or
        list ordered according to ``model.inputs`` for multi-input models.

    Raises:
        ValueError: If no compatible assignment can be inferred.
    """

    lidar = np.asarray(lidar, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    lidar_shape = tuple(lidar.shape[1:])
    coords_shape = tuple(coords.shape[1:])

    # Accept dynamic dimensions (None) while validating static dimensions.
    def compatible(expected: tuple[int, ...], actual: tuple[int, ...]) -> bool:
        if len(expected) != len(actual):
            return False
        for exp, act in zip(expected, actual):
            if exp is not None and int(exp) != int(act):
                return False
        return True

    # Map each symbolic input to the right array using name first, then shape fallback.
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
                f"lidar={lidar.shape}, coords={coords.shape}."
            )
        prepared_by_name[input_name] = candidate

    ordered_inputs = [prepared_by_name[t.name.split(":")[0]] for t in model.inputs]
    if len(ordered_inputs) == 1:
        return ordered_inputs[0]
    return ordered_inputs


def predict_with_models(
    models: Sequence[tf.keras.Model],
    lidar_inputs: np.ndarray,
    coord_inputs: np.ndarray,
    coord_scaler: StandardScaler,
    *,
    batch_size: int,
) -> list[np.ndarray]:
    """Run inference for all teachers on a dataset split.

    Args:
        models (Sequence[tf.keras.Model]): Teacher models to evaluate.
        lidar_inputs (np.ndarray): LiDAR samples.
        coord_inputs (np.ndarray): Raw coordinate samples (not scaled yet).
        coord_scaler (StandardScaler): Fitted scaler from the S008 train split.
        batch_size (int): Inference batch size.

    Returns:
        list[np.ndarray]: Per-model probability matrices with shape ``(N, C)``.

    Raises:
        ValueError: If ``models`` is empty.
    """

    if not models:
        raise ValueError("At least one teacher model is required.")

    lidar = np.asarray(lidar_inputs, dtype=np.float32)
    coords = np.asarray(coord_inputs, dtype=np.float32)
    # Teachers receive normalized coordinates using the scaler fit on train split only.
    scaled_coords = coord_scaler.transform(coords)

    outputs: list[np.ndarray] = []
    for model in models:
        inputs = _prepare_inputs(model, lidar, scaled_coords)
        outputs.append(model.predict(inputs, batch_size=batch_size, verbose=0))
    return outputs


def average_teacher_predictions(predictions: Sequence[np.ndarray]) -> np.ndarray:
    """Average per-model probability outputs into ensemble soft targets.

    Args:
        predictions (Sequence[np.ndarray]): Teacher probability matrices. All
            matrices must share the same shape ``(N, C)``.

    Returns:
        np.ndarray: Ensemble probability matrix with shape ``(N, C)``.

    Raises:
        ValueError: If input list is empty or shapes are inconsistent.
    """

    if not predictions:
        raise ValueError("Cannot average teacher predictions from an empty sequence.")

    # Enforce consistent shape before averaging so soft targets remain valid.
    reference_shape = predictions[0].shape
    stacked = []
    for probs in predictions:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != reference_shape:
            raise ValueError(
                f"All teacher outputs must have the same shape. Expected {reference_shape}, got {probs.shape}."
            )
        stacked.append(probs)

    return np.mean(np.stack(stacked, axis=0), axis=0).astype(np.float32)


def _format_topk_line(name: str, topk: dict[int, float]) -> str:
    """Build a single readable line containing top-k metrics.

    Args:
        name (str): Prefix name for the metric row.
        topk (dict[int, float]): Top-k mapping from ``k`` to accuracy in
            ``[0.0, 1.0]``.

    Returns:
        str: Human-readable line like ``name: Top-1: 80.00%, Top-5: 92.00%``.
    """

    parts = ", ".join(f"Top-{k}: {topk[k] * 100:.2f}%" for k in TOP_K if k in topk)
    return f"{name}: {parts}"


def _to_json_serializable(value: Any) -> Any:
    """Convert TensorFlow/NumPy-rich objects into JSON-serializable Python types.

    Args:
        value (Any): Arbitrary nested object that may include TensorFlow
            tensors, NumPy scalars/arrays, dictionaries, tuples, and klists.

    Returns:
        Any: A structure containing only JSON-compatible primitive types,
        dictionaries, and lists.

    Raises:
        TypeError: If a value cannot be converted to a JSON-serializable type.
    """

    # Preserve JSON-native scalar types without modification.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # TensorFlow tensors are converted to NumPy first, then normalized below.
    if tf.is_tensor(value):
        return _to_json_serializable(value.numpy())

    # NumPy scalar values become plain Python scalars (int/float/bool).
    if isinstance(value, np.generic):
        return value.item()

    # NumPy arrays are recursively normalized element-wise.
    if isinstance(value, np.ndarray):
        return [_to_json_serializable(item) for item in value.tolist()]

    # Keep dictionary keys as strings so JSON encoding is deterministic.
    if isinstance(value, dict):
        return {str(key): _to_json_serializable(item) for key, item in value.items()}

    # Normalize tuples/lists to JSON arrays.
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(item) for item in value]

    raise TypeError(f"Unsupported type for JSON serialization: {type(value).__name__}")


# ———————————————————————————————————————————————————————————————————————————— #
#                              Training Pipeline                               #
# ———————————————————————————————————————————————————————————————————————————— #
def main() -> None:
    """Train a V7 student using top-X ensemble teacher distillation.

    Returns:
        None: Artifacts and metrics are written to disk under ``runs``.

    Raises:
        ValueError: If training data becomes empty after filtering or if splits
            are inconsistent.
        FileNotFoundError: If teacher model files cannot be resolved.
    """

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

    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    lidar_s008, coord_s008, labels_s008 = _filter_rare_classes(
        lidar_s008,
        coord_s008,
        labels_s008,
        min_samples=2,
    )
    if labels_s008.size == 0:
        raise ValueError("No S008 samples remain after filtering rare classes.")

    classes = np.sort(np.unique(labels_s008))
    stratify_labels = _maybe_stratify(labels_s008)
    split_kwargs = {"stratify": stratify_labels} if stratify_labels is not None else {}

    (
        lidar_train,
        lidar_val,
        coord_train,
        coord_val,
        labels_train_raw,
        labels_val_raw,
    ) = train_test_split(
        lidar_s008,
        coord_s008,
        labels_s008,
        train_size=TRAIN_RATIO,
        shuffle=True,
        random_state=DATA_SEED,
        **split_kwargs,
    )

    missing_train = np.setdiff1d(np.unique(labels_s008), np.unique(labels_train_raw))
    if missing_train.size:
        preview = ", ".join(str(int(v)) for v in missing_train[:10])
        suffix = "..." if missing_train.size > 10 else ""
        raise ValueError(f"Training split is missing {missing_train.size} class(es): {preview}{suffix}")

    coord_scaler = StandardScaler()
    # Fit only on train coordinates to prevent validation/test leakage.
    coord_scaler.fit(coord_train)
    coord_train_scaled = coord_scaler.transform(coord_train).astype(np.float32)
    coord_val_scaled = coord_scaler.transform(coord_val).astype(np.float32)

    # Keep labels in the original 256-class id space so they stay compatible
    # with both teacher outputs and the fixed 256-way student output head.
    labels_train = np.asarray(labels_train_raw, dtype=np.int64)
    labels_val = np.asarray(labels_val_raw, dtype=np.int64)

    print("Loaded raw S008/S009 data with S008 train/val split:")
    print(f"  S008 train -> lidar: {lidar_train.shape}, coord: {coord_train_scaled.shape}, labels: {labels_train.shape}")
    print(f"  S008 val   -> lidar: {lidar_val.shape}, coord: {coord_val_scaled.shape}, labels: {labels_val.shape}")
    print(f"  S009 test  -> lidar: {np.asarray(x_lidar_test).shape}, coord: {np.asarray(x_coord_test).shape}, labels: {np.asarray(y_test).shape}")

    teacher_paths = resolve_teacher_model_paths(
        explicit_paths=EXPLICIT_TEACHER_PATHS,
        models_dir=TEACHER_MODELS_DIR,
        top_x=TOP_X_MODELS,
    )
    print("Teacher models selected:")
    for index, model_path in enumerate(teacher_paths, start=1):
        print(f"  {index}. {model_path}")

    teachers = [load_model(path) for path in teacher_paths]

    # Generate teacher outputs once per split and reuse them for distillation.
    teacher_train_preds = predict_with_models(
        teachers,
        lidar_train,
        coord_train,
        coord_scaler,
        batch_size=BATCH_SIZE,
    )
    teacher_val_preds = predict_with_models(
        teachers,
        lidar_val,
        coord_val,
        coord_scaler,
        batch_size=BATCH_SIZE,
    )
    teacher_test_preds = predict_with_models(
        teachers,
        np.asarray(x_lidar_test, dtype=np.float32),
        np.asarray(x_coord_test, dtype=np.float32),
        coord_scaler,
        batch_size=BATCH_SIZE,
    )

    print("Teacher Top-K accuracies on S009:")
    base_topk_results: list[dict[str, Any]] = []
    for index, probs in enumerate(teacher_test_preds, start=1):
        topk = compute_topk_accuracy(probs, np.asarray(y_test, dtype=np.int64))
        print(f"  Teacher {index} -> {_format_topk_line('S009', topk)}")
        base_topk_results.append(
            {
                "teacher_index": index,
                "teacher_path": str(teacher_paths[index - 1]),
                "s009_topk": {f"top_{k}": float(topk[k]) for k in TOP_K},
            }
        )

    # Build averaged ensemble soft targets in the original class space.
    ensemble_soft_train = average_teacher_predictions(teacher_train_preds)
    ensemble_soft_val = average_teacher_predictions(teacher_val_preds)

    y_test_np = np.asarray(y_test, dtype=np.int64)
    s009_mask = np.isin(y_test_np, classes)
    if not s009_mask.all():
        unseen = np.setdiff1d(np.unique(y_test_np), classes)
        prefix = ", ".join(str(int(v)) for v in unseen[:10])
        suffix = "..." if unseen.size > 10 else ""
        print(
            f"Warning: dropping {np.count_nonzero(~s009_mask)} S009 sample(s) with unseen labels: "
            f"{prefix}{suffix}"
        )

    x_lidar_test_f = np.asarray(x_lidar_test, dtype=np.float32)[s009_mask]
    x_coord_test_f = coord_scaler.transform(np.asarray(x_coord_test, dtype=np.float32)[s009_mask]).astype(np.float32)
    y_test_f = np.asarray(y_test_np[s009_mask], dtype=np.int64)
    ensemble_soft_test = average_teacher_predictions(teacher_test_preds)[s009_mask]

    # Build the same architecture as train_v7_top-k and optimize it via KD.
    student = build_v7_student_model(show_summary=True)
    distiller = Distiller(student=student, alpha=DISTILL_ALPHA, temperature=DISTILL_TEMPERATURE)
    distiller.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_1_accuracy")],
        jit_compile=False,
    )

    callback_list: list[callbacks.Callback] = [
        callbacks.EarlyStopping(
            monitor="val_top_1_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_top_1_accuracy",
            mode="max",
            patience=REDUCE_LR_PATIENCE,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = distiller.fit(
        x=(lidar_train, coord_train_scaled),
        y=(labels_train, ensemble_soft_train),
        validation_data=((lidar_val, coord_val_scaled), labels_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        verbose=2,
    )

    # Evaluate the student alone (without Distiller wrapper) for deployment parity.
    train_probs = student.predict((lidar_train, coord_train_scaled), batch_size=BATCH_SIZE, verbose=0)
    val_probs = student.predict((lidar_val, coord_val_scaled), batch_size=BATCH_SIZE, verbose=0)

    train_topk = compute_topk_accuracy(train_probs, labels_train)
    val_topk = compute_topk_accuracy(val_probs, labels_val)

    test_topk: dict[int, float] | None = None
    if x_lidar_test_f.shape[0] > 0:
        test_probs = student.predict((x_lidar_test_f, x_coord_test_f), batch_size=BATCH_SIZE, verbose=0)
        test_topk = compute_topk_accuracy(test_probs, y_test_f)
        print(f"Student distilled S009 -> {_format_topk_line('S009', test_topk)}")
    else:
        print("No S009 samples left after filtering unseen labels; skipping student S009 evaluation.")

    student_model_path = run_dir / "student_distilled.keras"
    student.save(student_model_path)

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(_to_json_serializable(history.history), indent=2))

    results = {
        "config": {
            "data_seed": DATA_SEED,
            "train_seed": TRAIN_SEED,
            "train_ratio": TRAIN_RATIO,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "top_k": list(TOP_K),
            "teacher_top_x": TOP_X_MODELS,
            "teacher_model_paths": [str(path) for path in teacher_paths],
            "distill_alpha": DISTILL_ALPHA,
            "distill_temperature": DISTILL_TEMPERATURE,
        },
        "sizes": {
            "s008_train": int(labels_train.shape[0]),
            "s008_val": int(labels_val.shape[0]),
            "s009_test": int(y_test_np.shape[0]),
            "s009_test_used": int(y_test_f.shape[0]),
        },
        "classes": [int(c) for c in classes],
        "teacher_topk": base_topk_results,
        "student_train_topk": {f"top_{k}": float(train_topk[k]) for k in TOP_K},
        "student_val_topk": {f"top_{k}": float(val_topk[k]) for k in TOP_K},
        "student_test_topk": {f"top_{k}": float(test_topk[k]) for k in TOP_K} if test_topk else None,
        "artifacts": {
            "student_model": str(student_model_path),
            "history_json": str(history_path),
        },
    }

    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(_to_json_serializable(results), indent=2))

    lines = [
        _format_topk_line("student_train", train_topk),
        _format_topk_line("student_val", val_topk),
    ]
    if test_topk:
        lines.append(_format_topk_line("student_test", test_topk))
    accuracies_path = run_dir / "accuracies.txt"
    accuracies_path.write_text("\n".join(lines))

    print(f"Saved distilled student model to: {student_model_path}")
    print(f"Saved results to: {results_path}")
    print(f"Saved accuracies to: {accuracies_path}")

    # Release teacher graph memory after training to reduce GPU memory pressure.
    del teachers
    gc.collect()
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
