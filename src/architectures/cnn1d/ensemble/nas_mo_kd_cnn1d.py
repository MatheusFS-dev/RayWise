"""
Multi-Objective NAS with Knowledge Distillation for CNN1D Ray Propagation Models.

This script uses Optuna with MOTPE sampler to jointly optimize:
1. Maximize validation top-1 accuracy (via knowledge distillation from teacher ensemble)
2. Minimize model FLOPs (architecture efficiency)

The student models are TFLite-compatible CNN1D architectures trained on LiDAR ray
propagation predictions using soft targets from pre-trained teacher ensembles.

- Authored by Matheus Ferreira Silva
- GitHub https://github.com/MatheusFS-dev
"""

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Sys Path Configuration                            #
# ———————————————————————————————————————————————————————————————————————————— #
import sys
from pathlib import Path
from time import perf_counter

# Ensure the repository src/ directory is on sys.path when running this file
# directly, so local package imports resolve consistently across entrypoints.
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Setup and Configuration                           #
# ———————————————————————————————————————————————————————————————————————————— #
import os
import re

# Specify GPU to use (e.g., GPU:0, CPU:-1):
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use CUDA async allocator to reduce fragmentation:
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Allow TensorFlow to allocate GPU memory as needed:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR):
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

# %%—————————————————————————————————————————————————————————————————————————— #
#                                    Imports                                   #
# ———————————————————————————————————————————————————————————————————————————— #
from imports_keras import *  # Centralized file containing all imports
from _load_dataset import load_dataset_sparse_labels

# Suppress TensorFlow logging:
tf.get_logger().setLevel("ERROR")

# %%—————————————————————————————————————————————————————————————————————————— #
#                                Run Parameters                                #
# ———————————————————————————————————————————————————————————————————————————— #
# Execution mode switch:
DO_ONLY_SEARCH_SPACE = False

# Number of trials to run:
NUM_TRIALS = 2000

# Training configuration:
EPOCHS = 100
BATCH_SIZE = 64

# Knowledge distillation hyperparameters:
DISTILL_ALPHA_MIN = 0.1
DISTILL_ALPHA_MAX = 0.9
DISTILL_TEMPERATURE_MIN = 1.0
DISTILL_TEMPERATURE_MAX = 10.0

# Optuna sampler seed:
SAMPLER_SEED = 0

STEPS_PER_EXECUTION = 1

# Enable or disable XLA compilation:
USE_JIT_COMPILE = True

# Number of top trials to save:
TOP_K = 10

# Trial ranking order:
ORDER = "descending"

# Key to rank trials by:
RANK_KEY = "val_top_1_accuracy"

# Multi-objective optimization directions:
# 1. Maximize top-1 accuracy
# 2. Minimize FLOPs (keep under 70M)
DIRECTION = ["maximize", "minimize"]

# Callback hyperparameters:
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
PRUNING_INTERVAL = 3

# Large penalty values for infeasible models:
PENALTY_VAL_ACCURACY = 0.0
PENALTY_FLOPS = 1e15

# Hard resource constraints:
MAX_FLOPS = 70_000_000  # 70M FLOPs (below 70M threshold)

POLICY = mixed_precision.Policy("float32")
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# Keep data split and initializer seed fixed across trials:
DATA_SEED = 8128
TRAIN_SEED = 5264

# Teacher ensemble configuration:
TEACHER_MODELS_DIR = Path("/media/matheus/SSD-2/matheus/results/raywise/")
EXPLICIT_TEACHER_PATHS: list[Path] = []
TEACHER_COUNT_MIN = 1
TEACHER_COUNT_MAX = 10

# Dataset configuration:
S008_DATASET_PATH = PROJECT_SRC_DIR / "data" / "s008"
S009_DATASET_PATH = PROJECT_SRC_DIR / "data" / "s009"

# Keep startup failures explicit now that paths are fixed and non-interactive.
if not TEACHER_MODELS_DIR.is_dir():
    raise FileNotFoundError(f"Teacher models directory not found: {TEACHER_MODELS_DIR}")
if not S008_DATASET_PATH.is_dir():
    raise FileNotFoundError(f"S008 dataset directory not found: {S008_DATASET_PATH}")
if not S009_DATASET_PATH.is_dir():
    raise FileNotFoundError(f"S009 dataset directory not found: {S009_DATASET_PATH}")

# Run directory:
RUN_DIR = f"runs/{get_caller_stem()}_mo_kd"


# ———————————————————————————————————————————————————————————————————————————— #
#                               Utility Functions                              #
# ———————————————————————————————————————————————————————————————————————————— #
def log_print(function_name: str, message: str) -> None:
    """Print a standardized log line.

    Args:
        function_name (str): Function or block identifier that emits the
            message, preserving greppability across runs.
        message (str): Human-readable message body.

    Returns:
        None: Writes one formatted line to stdout.
    """
    print(f"[{Path(__file__).stem}:{function_name}] {message}")


def log_print_red(function_name: str, message: str) -> None:
    """Print a standardized red log line for highlighted runtime information.

    Args:
        function_name (str): Function or block identifier that emits the
            message.
        message (str): Human-readable message body to be highlighted in red.

    Returns:
        None: Writes one ANSI-colored formatted line to stdout.
    """
    print(f"\033[31m[{Path(__file__).stem}:{function_name}] {message}\033[0m")


def append_trial_micro_timing_row(csv_path: str, row_data: dict[str, Any]) -> None:
    """Append one trial timing record to a CSV file.

    Args:
        csv_path (str): Destination CSV path for micro-task timings.
        row_data (dict[str, Any]): Flat mapping of timing keys to scalar values
            for one trial.

    Returns:
        None: Writes one row to ``csv_path``.
    """
    pd.DataFrame([row_data]).to_csv(
        csv_path,
        mode="a",
        index=False,
        header=not os.path.exists(csv_path),
    )


def mark_step(
    step_timings: dict[str, float | int | str],
    step_name: str,
    step_start: float,
) -> None:
    """Record elapsed time for a named micro task.

    Args:
        step_timings (dict[str, float | int | str]): Mutable accumulator that
            stores timing values by step name.
        step_name (str): Stable key used as a CSV column name.
        step_start (float): ``perf_counter`` timestamp captured before the
            step begins.

    Returns:
        None: Updates ``step_timings`` in-place.
    """
    step_timings[step_name] = perf_counter() - step_start


@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(layers.Layer):
    """Convert raw LiDAR tensors into a 4-channel semantic representation.

    Returns:
        tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)`` encoding
        base station, vehicle, obstacle, and free-space channels.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic one-hot conversion to raw LiDAR grids.

        Args:
            inputs (tf.Tensor): Raw LiDAR tensor with shape ``(batch, 20, 200, 10)``
                and semantic values in ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)``.
        """
        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)

    def get_config(self) -> dict[str, Any]:
        """Return serializable Keras configuration.

        Returns:
            dict[str, Any]: Serializable layer configuration.
        """
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(layers.Layer):
    """Tile 2D coordinates to match the flattened LiDAR token length.

    Args:
        grid_length (int): Number of flattened LiDAR positions.
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        """Initialize the coordinate tiling layer.

        Args:
            grid_length (int): Number of positions to broadcast coordinates over.
            **kwargs (Any): Additional Keras layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Broadcast coordinates across all flattened LiDAR positions.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled tensor with shape ``(batch, grid_length, 2)``.
        """
        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict[str, Any]:
        """Return serializable Keras configuration.

        Returns:
            dict[str, Any]: Serializable layer configuration including grid length.
        """
        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


def load_training_data(
    *,
    s008_path: str,
    s009_path: str,
    data_seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    StandardScaler,
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Load baseline arrays through ``_load_dataset`` and create train/val splits.

    Args:
        s008_path (str): Path to the baseline S008 dataset directory.
        s009_path (str): Path to the baseline S009 dataset directory.
        data_seed (int): Random seed used for the S008 train/validation split.

    Returns:
        tuple: Training split, validation split, fitted coordinate scaler, and
        scaled S009 data for optional downstream evaluation.

    Raises:
        ValueError: If the loaded arrays are misaligned.
    """
    (
        s008_coord_input,
        s008_lidar_input,
        s008_y_all,
        s009_coord_input,
        s009_lidar_input,
        s009_y,
    ) = load_dataset_sparse_labels(s008_path=s008_path, s009_path=s009_path)

    (
        x_s008_lidar_train,
        x_s008_lidar_val,
        x_s008_coord_train,
        x_s008_coord_val,
        y_s008_train,
        y_s008_val,
    ) = train_test_split(
        s008_lidar_input,
        s008_coord_input,
        s008_y_all,
        test_size=0.2,
        random_state=data_seed,
        shuffle=True,
    )

    # Fit the coordinate scaler only on the training split to avoid leakage.
    coord_scaler = StandardScaler()
    coord_scaler.fit(x_s008_coord_train)

    x_s008_coord_train = coord_scaler.transform(x_s008_coord_train)
    x_s008_coord_val = coord_scaler.transform(x_s008_coord_val)
    s009_coord_input = coord_scaler.transform(s009_coord_input)

    if (
        x_s008_lidar_train.shape[0] != x_s008_coord_train.shape[0]
        or x_s008_lidar_train.shape[0] != y_s008_train.shape[0]
        or x_s008_lidar_val.shape[0] != x_s008_coord_val.shape[0]
        or x_s008_lidar_val.shape[0] != y_s008_val.shape[0]
    ):
        raise ValueError("Loaded dataset arrays are misaligned. Check preprocessing and inputs.")

    return (
        (x_s008_lidar_train, x_s008_coord_train, y_s008_train),
        (x_s008_lidar_val, x_s008_coord_val, y_s008_val),
        coord_scaler,
        (s009_lidar_input, s009_coord_input, s009_y),
    )


def load_teacher_models(
    teacher_dir: Path,
    explicit_paths: list[Path],
    max_teachers: int,
) -> list[tf.keras.Model]:
    """Load and return the top teacher models for ensemble distillation.

    Priorities ranked filenames (e.g., top_1_trial_*.keras) and falls back to
    alphabetical ordering. Returns up to ``max_teachers`` models.

    Args:
        teacher_dir (Path): Directory containing .keras teacher model files.
        explicit_paths (list[Path]): Explicit list of teacher paths to use first.
            If non-empty, these are prioritized before directory discovery.
        max_teachers (int): Maximum number of teachers to load.

    Returns:
        list[tf.keras.Model]: List of loaded teacher Keras models.

    Raises:
        FileNotFoundError: If ``teacher_dir`` does not exist.
        ValueError: If no teacher models can be found or loaded.
    """
    teachers = []

    if explicit_paths:
        for path in explicit_paths[:max_teachers]:
            if path.exists():
                teachers.append(tf.keras.models.load_model(str(path)))
                log_print("load_teacher_models", f"Loaded teacher: {path.name}")
            if len(teachers) >= max_teachers:
                break
    else:
        # Discover teacher models in directory, prioritizing ranked filenames.
        if not teacher_dir.exists():
            raise FileNotFoundError(f"Teacher directory does not exist: {teacher_dir}")

        keras_files = sorted(teacher_dir.glob("*.keras"))
        if not keras_files:
            raise ValueError(f"No .keras files found in {teacher_dir}")

        # Sort by ranked prefix (top_1, top_2, etc.) if present:
        def sort_key(path: Path) -> tuple[int, str]:
            name = path.stem
            match = re.match(r"top_(\d+)", name)
            if match:
                rank = int(match.group(1))
            else:
                rank = int(1e9)  # Non-ranked files go to end
            return (rank, name)

        keras_files = sorted(keras_files, key=sort_key)

        for path in keras_files[:max_teachers]:
            teachers.append(tf.keras.models.load_model(str(path)))
            log_print("load_teacher_models", f"Loaded teacher: {path.name}")

    if not teachers:
        raise ValueError(f"Could not load any teacher models from {teacher_dir}")

    return teachers


def precompute_teacher_soft_targets_by_count(
    teachers: list[tf.keras.Model],
    x_lidar: np.ndarray,
    x_coord: np.ndarray,
    batch_size: int = 64,
    max_teachers: int | None = None,
) -> dict[int, np.ndarray]:
    """Precompute ensemble soft targets for every teacher-count prefix.

    Computes teacher predictions once per teacher and stores cumulative
    ensemble averages for ``k = 1..max_teachers``. This lets Optuna tune the
    number of teachers per trial without re-running teacher inference.

    Args:
        teachers (list[tf.keras.Model]): List of teacher models to ensemble.
        x_lidar (np.ndarray): LiDAR tensor with shape ``(N, 20, 200, 10)``.
        x_coord (np.ndarray): Coordinate matrix with shape ``(N, 2)``.
        batch_size (int): Batch size for teacher inference.
        max_teachers (int | None): Optional cap on the number of teachers used
            to build the cache. If ``None``, all loaded teachers are used. If
            provided, only the first ``max_teachers`` teachers are used, which
            reduces setup time and memory while restricting the search space.

    Returns:
        dict[int, np.ndarray]: Mapping where key ``k`` stores averaged soft
        targets for the first ``k`` teachers.

    Raises:
        ValueError: If ``teachers`` is empty, inputs are empty, or
            ``max_teachers`` is smaller than 1.
    """
    if not teachers:
        raise ValueError("At least one teacher model is required.")
    if x_lidar.shape[0] == 0 or x_coord.shape[0] == 0:
        raise ValueError("Teacher soft-target inputs must have at least one sample.")
    if max_teachers is not None and max_teachers < 1:
        raise ValueError("max_teachers must be >= 1 when provided.")

    effective_max_teachers = len(teachers) if max_teachers is None else min(max_teachers, len(teachers))

    # Compute one prediction pass per teacher and maintain a running sum so we
    # can materialize all prefix averages without repeated inference.
    soft_targets_by_count: dict[int, np.ndarray] = {}
    running_sum: np.ndarray | None = None

    for teacher_idx, teacher in enumerate(teachers[:effective_max_teachers], start=1):
        pred = teacher.predict([x_lidar, x_coord], batch_size=batch_size, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        teacher_pred = np.asarray(pred, dtype=np.float32)

        if running_sum is None:
            running_sum = teacher_pred
        else:
            running_sum = running_sum + teacher_pred

        soft_targets_by_count[teacher_idx] = (running_sum / float(teacher_idx)).astype(np.float32)

    return soft_targets_by_count


# %%—————————————————————————————————————————————————————————————————————————— #
#                         Distillation Loss Layer                              #
# ———————————————————————————————————————————————————————————————————————————— #
@tf.keras.utils.register_keras_serializable(package="raywise")
class Distiller(tf.keras.Model):
    """Keras model wrapper that trains a student from hard and soft targets.

    The wrapper expects batched targets as ``(hard_labels, teacher_probs)`` and
    combines sparse categorical cross-entropy with KL divergence distillation.
    """

    def __init__(self, student: tf.keras.Model, alpha: float, temperature: float) -> None:
        """Initialize distillation wrapper.

        Args:
            student (tf.keras.Model): Student model to optimize.
            alpha (float): Weight of the hard-label loss branch. If ``1.0``,
                training behaves like regular supervised learning. If ``0.0``,
                training prioritizes teacher imitation.
            temperature (float): Softening temperature for probability
                distributions. Higher values expose more class-similarity
                information (dark knowledge); lower values keep sharper
                distributions.

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
                dropout and batchnorm updates.

        Returns:
            tf.Tensor: Student class probabilities.
        """
        return self.student(inputs, training=training)

    def _collect_metric_logs(self) -> dict[str, tf.Tensor]:
        """Collect scalar metric logs, flattening nested metric containers.

        Returns:
            dict[str, tf.Tensor]: Mapping of metric names to scalar metric tensors.
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

        Combines hard-label supervision (sparse categorical cross-entropy) with
        soft-target knowledge (KL divergence from teacher predictions). The loss
        is weighted by ``alpha`` (hard branch) and ``1 - alpha`` (soft branch).

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
        """
        x_batch, y_batch = data
        if not isinstance(y_batch, tuple) or len(y_batch) != 2:
            raise ValueError("Distiller expects targets as (y_hard, y_teacher_soft).")
        y_hard, y_teacher = y_batch

        # Compute student outputs and both loss branches in a single gradient tape.
        with tf.GradientTape() as tape:
            y_pred = self.student(x_batch, training=True)
            hard_loss = self.hard_loss_fn(y_hard, y_pred)

            # Apply temperature scaling to both distributions in log-space to avoid
            # requiring logits. Clipping ensures numerical stability.
            eps = tf.keras.backend.epsilon()
            student_soft = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(y_pred, eps, 1.0)) / self.temperature,
                axis=-1,
            )
            teacher_soft = tf.nn.softmax(
                tf.math.log(tf.clip_by_value(y_teacher, eps, 1.0)) / self.temperature,
                axis=-1,
            )
            distill_loss = self.soft_loss_fn(teacher_soft, student_soft) * (self.temperature**2)

            # Blend hard supervision and teacher imitation:
            total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss

        # Update student parameters only; teacher predictions are fixed targets.
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Collect and update metrics:
        self.compiled_metrics.update_state(y_hard, y_pred)
        metric_logs = self._collect_metric_logs()
        metric_logs.update({
            "loss": total_loss,
            "hard_loss": hard_loss,
            "distill_loss": distill_loss,
        })

        return metric_logs

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Run one evaluation step against hard labels.

        Args:
            data (Any): Keras batch tuple ``(x_batch, y_hard)``.

        Returns:
            dict[str, tf.Tensor]: Evaluation metrics dictionary.
        """
        x_batch, y_batch = data
        y_hard = y_batch[0] if isinstance(y_batch, tuple) else y_batch
        y_pred = self.student(x_batch, training=False)
        hard_loss = self.hard_loss_fn(y_hard, y_pred)
        self.compiled_metrics.update_state(y_hard, y_pred)
        metric_logs = self._collect_metric_logs()
        metric_logs["loss"] = hard_loss
        return metric_logs


# %%—————————————————————————————————————————————————————————————————————————— #
#                            Model Definition                                  #
# ———————————————————————————————————————————————————————————————————————————— #
def build_compact_cnn1d_student(
    trial: optuna.Trial,
    *,
    num_classes: int = 256,
    show_summary: bool = True,
    **kwargs: Any,
) -> tf.keras.Model:
    """Build a compact CNN1D student model for LiDAR+coordinate distillation.

    The architecture keeps the same LiDAR and coordinate preprocessing flow as
    the existing V7 family, but shrinks the convolutional and dense capacity so
    Optuna can search for models under the target FLOPs budget.

    Args:
        trial (optuna.Trial): Current trial for hyperparameter suggestions.
        num_classes (int): Number of output classes (beam indices). Default 256.
        show_summary (bool): If ``True``, prints model summary.
        **kwargs (Any): Additional keyword arguments (train_seed).

    Returns:
        tf.keras.Model: Uncompiled Keras model that outputs class probabilities.

    Raises:
        ValueError: If invalid hyperparameter combinations are suggested.
    """
    initializer = initializers.GlorotUniform(seed=kwargs.get("train_seed", 0))

    # Sample a compact search space that stays close to the proven V7 design
    # while allowing meaningful FLOPs/accuracy tradeoffs.
    conv_kernel_sizes = [3, 5, 7, 9]
    conv_pool_sizes = [2, 4]
    num_conv_blocks = trial.suggest_int("num_conv_blocks", 1, 3)
    num_dense_layers = trial.suggest_int("num_dense_layers", 0, 2)
    activation_name = trial.suggest_categorical("activation", ["relu", "silu", "tanh"])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.05)
    global_pooling = trial.suggest_categorical("global_pooling", ["avg", "max"])

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

    # Distillation hyperparameters:
    distill_alpha = trial.suggest_float("distill_alpha", DISTILL_ALPHA_MIN, DISTILL_ALPHA_MAX, step=0.1)
    distill_temperature = trial.suggest_float(
        "distill_temperature",
        DISTILL_TEMPERATURE_MIN,
        DISTILL_TEMPERATURE_MAX,
        step=0.5,
    )

    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")
    x_coord_input = layers.Input(shape=(2,), name="coord_input")

    one_hot_lidar = LidarToOneHot(name="lidar_transform_to_one_hot")(x_lidar_input)
    x_lidar_flat = layers.Reshape((20 * 200, 4), name="lidar_flatten_4_channels")(one_hot_lidar)
    x_coord_tiled = CoordTileFlat(20 * 200, name="coord_tile_flat")(x_coord_input)
    x = layers.Concatenate(axis=-1, name="combine_lidar_coord")([x_lidar_flat, x_coord_tiled])

    for block_idx in range(num_conv_blocks):
        filters = trial.suggest_int(f"conv_{block_idx}_filters", 32, 160, step=32)
        kernel_size = trial.suggest_categorical(f"conv_{block_idx}_kernel_size", conv_kernel_sizes)
        pool_size = trial.suggest_categorical(f"conv_{block_idx}_pool_size", conv_pool_sizes)

        # Each block increases receptive field while reducing sequence length,
        # which is the main lever for staying below the FLOPs target.
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=initializer,
            name=f"conv1d_{block_idx}",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"conv1d_{block_idx}_bn")(x)
        x = layers.Activation(activation_name, name=f"conv1d_{block_idx}_act")(x)
        x = layers.MaxPooling1D(pool_size=pool_size, name=f"max_pool_{block_idx}")(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"conv1d_{block_idx}_dropout")(x)

    if global_pooling == "avg":
        x = layers.GlobalAveragePooling1D(name="global_average_pooling")(x)
    else:
        x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)

    for layer_idx in range(num_dense_layers):
        units = trial.suggest_int(f"dense_{layer_idx}_units", 64, 256, step=32)
        x = layers.Dense(
            units=units,
            kernel_initializer=initializer,
            name=f"dense_{layer_idx}",
        )(x)
        x = layers.Activation(activation_name, name=f"dense_{layer_idx}_act")(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"dense_{layer_idx}_dropout")(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        kernel_initializer=initializer,
        name="output",
    )(x)

    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,), name="compact_cnn1d_student")

    if show_summary:
        model.summary()

    return model


def get_multiobjective_fit_callbacks(
    *,
    trial_number: int,
    tensorboard_dir: str,
    monitor: str,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    mode: str = "auto",
) -> list[callbacks.Callback]:
    """Return Keras callbacks that are safe for Optuna multi-objective studies.

    Optuna's pruning callbacks call ``trial.report()``, which is unsupported in
    multi-objective optimization. This helper keeps only pure Keras callbacks.

    Args:
        trial_number (int): Trial number used for TensorBoard log subdirectory.
        tensorboard_dir (str): Base TensorBoard directory for the study.
        monitor (str): Validation metric monitored by callbacks.
        early_stopping_patience (int): Patience for early stopping.
        reduce_lr_patience (int): Patience for ReduceLROnPlateau.
        mode (str): Metric optimization mode for monitored callbacks.

    Returns:
        list[callbacks.Callback]: Safe callback list for ``model.fit``.
    """
    fit_callbacks: list[callbacks.Callback] = [
        callbacks.TerminateOnNaN(),
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            mode=mode,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=reduce_lr_patience,
            mode=mode,
            factor=0.2,
            min_lr=1e-10,
            verbose=0,
        ),
    ]

    if tensorboard_dir:
        fit_callbacks.append(
            callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_dir, f"trial_{trial_number}"),
                histogram_freq=1,
                write_graph=False,
                write_images=True,
                update_freq="epoch",
            )
        )

    return fit_callbacks


# %%—————————————————————————————————————————————————————————————————————————— #
#                              Objective Function                              #
# ———————————————————————————————————————————————————————————————————————————— #
def objective(
    trial: optuna.Trial,
    **kwargs: Any,
) -> tuple[float, float]:
    """Multi-objective function for knowledge distillation NAS.

    Optimizes a CNN1D student model trained on soft targets from teacher
    ensemble, jointly maximizing validation top-1 accuracy and minimizing FLOPs.

    Args:
        trial (optuna.Trial): Current trial for hyperparameter suggestions.
        **kwargs (Any): Additional keyword arguments containing directory paths.

    Returns:
        tuple[float, float]: Pair of objectives
            ``(val_top_1_accuracy, flops)``.
            The study maximizes validation top-1 accuracy and minimizes FLOPs.

    Raises:
        optuna.TrialPruned: If model is infeasible (FLOPs exceed budget) or
            if training encounters recoverable errors.
    """
    log_print("objective", f"\n\nRunning trial {trial.number}...")
    clear_session()

    trial_t0 = perf_counter()
    step_timings: dict[str, float | int | str] = {}
    trial_status = "unknown"
    train_seed = TRAIN_SEED
    batch_size = BATCH_SIZE

    # Set reproducibility seeds:
    set_random_seed(train_seed)

    # Extract directory paths:
    backup_dir = kwargs["backup"]
    model_dir = kwargs["model"]
    fig_dir = kwargs["fig"]
    logs_dir = kwargs["logs"]
    history_dir = kwargs["history"]
    tensorboard_dir = kwargs["tensorboard"]

    try:
        # ———————— Data Preprocessing ————————
        step_t0 = perf_counter()
        x_train = [FIXED_X_LIDAR_TRAIN, FIXED_X_COORD_TRAIN]
        x_val = [FIXED_X_LIDAR_VAL, FIXED_X_COORD_VAL]
        y_train_labels = FIXED_Y_TRAIN_LABELS
        y_val_labels = FIXED_Y_VAL_LABELS
        num_teachers = trial.suggest_int("num_teachers", TEACHER_COUNT_MIN, FIXED_MAX_TEACHER_COUNT)
        y_train_soft = FIXED_Y_TRAIN_SOFT_BY_COUNT[num_teachers]
        trial.set_user_attr("num_teachers", int(num_teachers))
        mark_step(step_timings, "dataset_binding", step_t0)

        step_t0 = perf_counter()
        validate_steps_per_execution(
            name=["train", "val"],
            data_size=[y_train_labels.shape[0], y_val_labels.shape[0]],
            batch_size=batch_size,
            steps_per_execution=STEPS_PER_EXECUTION,
        )
        mark_step(step_timings, "validate_steps_per_execution", step_t0)

        # ———————— Model Construction ————————
        step_t0 = perf_counter()
        model = build_compact_cnn1d_student(
            trial=trial,
            num_classes=FIXED_NUM_CLASSES,
            show_summary=False,
            train_seed=train_seed,
        )
        mark_step(step_timings, "model_build", step_t0)

        # ———————— Resource Estimation ————————
        # Prune oversized candidates before training to keep the study focused
        # on architectures that satisfy the deployment budget.
        step_t0 = perf_counter()
        prune_model_by_config(
            trial=trial,
            model=model,
            thresholds={
                "model_size": 1024,
                "memory_mb": 16384,
                "param": 2e8,
                "flops": float(MAX_FLOPS),
            },
            bytes_per_param=BYTES_PER_PARAM,
            batch_size=1,
        )
        mark_step(step_timings, "resource_estimation", step_t0)

        # ———————— Distillation Configuration ————————
        distill_alpha = trial.params.get("distill_alpha", 0.5)
        distill_temperature = trial.params.get("distill_temperature", 5.0)

        # Wrap student with distillation loss:
        distiller = Distiller(
            student=model,
            alpha=distill_alpha,
            temperature=distill_temperature,
        )

        # Get optimizer from trial parameters:
        optimizer_name = trial.params.get("optimizer", "adam")
        learning_rate = trial.params.get("learning_rate", 1e-3)

        if optimizer_name == "sgd":
            optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == "adamw":
            optimizer = optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate=learning_rate)

        # Compile distiller with top-1 accuracy metric:
        distiller.compile(
            optimizer=optimizer,
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[
                metrics.SparseCategoricalAccuracy(name="top_1_accuracy"),
                metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
                metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
            ],
            jit_compile=USE_JIT_COMPILE,
            steps_per_execution=STEPS_PER_EXECUTION,
        )

        # ———————— Training ————————
        step_t0 = perf_counter()
        history = distiller.fit(
            x=x_train,
            y=(y_train_labels, y_train_soft),
            validation_data=(x_val, y_val_labels),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=get_multiobjective_fit_callbacks(
                trial_number=trial.number,
                tensorboard_dir=tensorboard_dir,
                monitor="val_top_1_accuracy",
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                reduce_lr_patience=REDUCE_LR_PATIENCE,
                mode="max",
            ),
            verbose=2,
        )
        mark_step(step_timings, "model_fit", step_t0)

        # ———————— Model Save ————————
        step_t0 = perf_counter()
        model.save(os.path.join(model_dir, f"trial_{trial.number}.keras"))
        mark_step(step_timings, "model_save", step_t0)

        # ———————— Metrics Collection ————————
        step_t0 = perf_counter()
        set_user_attr_model_stats(
            trial=trial,
            model=model,
            bytes_per_param=BYTES_PER_PARAM,
            batch_size=1,
            test_runs=10,
            device="gpu/0",
            stats_to_measure=(
                "parameters",
                "model_size",
                "flops",
                "macs",
                "summary",
                "inference_latency",
            ),
            extra_attrs=None,
            verbose=1,
        )
        flops = float(trial.user_attrs.get("flops", PENALTY_FLOPS))
        trial.set_user_attr("flops_batch1", flops)
        mark_step(step_timings, "model_stats", step_t0)

        # ———————— Best Epoch Extraction ————————
        step_t0 = perf_counter()
        val_accuracy_history = history.history.get("val_top_1_accuracy", [])
        if val_accuracy_history:
            best_idx = np.argmax(val_accuracy_history)
            best_val_accuracy = val_accuracy_history[best_idx]
        else:
            best_idx = -1
            best_val_accuracy = 0.0

        trial.set_user_attr("best_epoch", int(best_idx) + 1)
        trial.set_user_attr("val_top_1_accuracy", float(best_val_accuracy))
        mark_step(step_timings, "best_epoch_and_attrs", step_t0)

        # ———————— History Save ————————
        step_t0 = perf_counter()
        history_path = os.path.join(history_dir, f"trial_{trial.number}.csv")
        history_data = {
            "epoch": list(range(1, len(history.history.get("loss", [])) + 1)),
            "train_loss": history.history.get("loss", []),
            "train_top_1_accuracy": history.history.get("top_1_accuracy", []),
            "val_loss": history.history.get("val_loss", []),
            "val_top_1_accuracy": history.history.get("val_top_1_accuracy", []),
        }

        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_path, index=False)
        mark_step(step_timings, "history_save", step_t0)

        trial_status = "complete"

        return float(best_val_accuracy), float(flops)

    except optuna.exceptions.TrialPruned:
        trial_status = "pruned"
        raise
    except Exception as e:
        trial_status = "failed_exception"
        log_trial_error(
            trial=trial,
            exc=e,
            logs_dir=logs_dir,
            prune_on={
                tf.errors.ResourceExhaustedError: None,
                tf.errors.InternalError: None,
                tf.errors.UnavailableError: None,
            },
            propagate={
                optuna.exceptions.TrialPruned: None,
            },
            force_crash_oom=5,
        )
        return float(PENALTY_VAL_ACCURACY), float(PENALTY_FLOPS)
    finally:
        total_elapsed = perf_counter() - trial_t0
        step_timings["total_seconds"] = total_elapsed
        step_timings["trial_number"] = trial.number
        step_timings["status"] = trial_status

        timings_csv_path = os.path.join(logs_dir, "micro_task_timings.csv")
        append_trial_micro_timing_row(timings_csv_path, step_timings)


# %%—————————————————————————————————————————————————————————————————————————— #
#                             Dataset and Teachers Setup                       #
# ———————————————————————————————————————————————————————————————————————————— #
clear()
log_print("setup", f"Using S008 dataset: {S008_DATASET_PATH}")
log_print("setup", f"Using S009 dataset: {S009_DATASET_PATH}")
log_print("setup", "Loading baseline arrays through _load_dataset...")

(
    train_split,
    val_split,
    FIXED_COORD_SCALER,
    FIXED_S009_SPLIT,
) = load_training_data(
    s008_path=str(S008_DATASET_PATH),
    s009_path=str(S009_DATASET_PATH),
    data_seed=DATA_SEED,
)

FIXED_X_LIDAR_TRAIN, FIXED_X_COORD_TRAIN, FIXED_Y_TRAIN_LABELS = train_split
FIXED_X_LIDAR_VAL, FIXED_X_COORD_VAL, FIXED_Y_VAL_LABELS = val_split
FIXED_S009_LIDAR, FIXED_S009_COORD, FIXED_S009_LABELS = FIXED_S009_SPLIT
FIXED_NUM_CLASSES = int(max(FIXED_Y_TRAIN_LABELS.max(), FIXED_Y_VAL_LABELS.max(), FIXED_S009_LABELS.max()) + 1)

fixed_train_dataset_gb = (
    FIXED_X_LIDAR_TRAIN.nbytes
    + FIXED_X_COORD_TRAIN.nbytes
    + FIXED_Y_TRAIN_LABELS.nbytes
) / 1e9
log_print_red("setup", f"Loaded training split memory footprint: {fixed_train_dataset_gb:.6f} GB")

log_print(
    "setup",
    (
        "Cached fixed datasets: "
        f"lidar_train={FIXED_X_LIDAR_TRAIN.shape}, lidar_val={FIXED_X_LIDAR_VAL.shape}, "
        f"coord_train={FIXED_X_COORD_TRAIN.shape}, coord_val={FIXED_X_COORD_VAL.shape}, "
        f"y_train={FIXED_Y_TRAIN_LABELS.shape}, y_val={FIXED_Y_VAL_LABELS.shape}"
    ),
)

log_print("setup", "Loading teacher models for knowledge distillation...")
teachers = load_teacher_models(
    teacher_dir=TEACHER_MODELS_DIR,
    explicit_paths=EXPLICIT_TEACHER_PATHS,
    max_teachers=TEACHER_COUNT_MAX,
)
log_print("setup", f"Loaded {len(teachers)} teacher models")

FIXED_MAX_TEACHER_COUNT = min(TEACHER_COUNT_MAX, len(teachers))
if FIXED_MAX_TEACHER_COUNT < TEACHER_COUNT_MIN:
    raise ValueError(
        "Not enough teacher models to satisfy teacher-count search space: "
        f"available={len(teachers)}, minimum_required={TEACHER_COUNT_MIN}."
    )

log_print(
    "setup",
    (
        "Teacher-count search space configured as "
        f"[{TEACHER_COUNT_MIN}, {FIXED_MAX_TEACHER_COUNT}] based on available models"
    ),
)

# Precompute teacher outputs once so trials only search student architectures,
# while still allowing trial-specific teacher-count selection.
FIXED_Y_TRAIN_SOFT_BY_COUNT = precompute_teacher_soft_targets_by_count(
    teachers=teachers,
    x_lidar=FIXED_X_LIDAR_TRAIN,
    x_coord=FIXED_X_COORD_TRAIN,
    batch_size=BATCH_SIZE,
    max_teachers=FIXED_MAX_TEACHER_COUNT,
)


# %%—————————————————————————————————————————————————————————————————————————— #
#                                     Study                                    #
# ———————————————————————————————————————————————————————————————————————————— #
if not DO_ONLY_SEARCH_SPACE:
    log_print("main", "Running multi-objective Optuna study with MOTPE sampler...")

    study = run_study(
        objective=objective,
        run_dir=RUN_DIR,
        num_trials=NUM_TRIALS,
        sampler_seed=SAMPLER_SEED,
        direction=DIRECTION,
        top_k=TOP_K,
        rank_key=RANK_KEY,
        order=ORDER,
        convergence_epoch_column="train_loss",
        convergence_epoch_direction="minimize",
        init_study_dirs=[
            "args",
            "fig",
            "backup",
            "history",
            "scaler",
            "model",
            "logs",
            "tensorboard",
        ],
        cleanup_paths=[
            ("model", "trial_{trial_id}.keras"),
            ("fig", "trial_{trial_id}.png"),
            ("history", "trial_{trial_id}.csv"),
            ("scaler", "trial_{trial_id}.pkl"),
            ("tensorboard", "trial_{trial_id}"),
        ],
        rename_paths=[
            ("model", ".keras"),
            ("fig", ".png"),
            ("history", ".csv"),
            ("scaler", ".pkl"),
        ],
        extra_attrs=[
            "best_epoch",
            "val_top_1_accuracy",
            "flops_batch1",
            "num_teachers",
        ],
        variance_threshold=None,
        prune_threshold=None,
        patience=None,
    )

    log_print("main", "Study completed successfully!")
    log_print("main", f"Pareto front size: {len(study.best_trials)}")
