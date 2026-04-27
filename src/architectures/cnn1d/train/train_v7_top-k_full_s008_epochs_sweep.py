"""
Model Training (Epoch Sweep)

- Authored by Matheus Ferreira Silva
- GitHub https://github.com/MatheusFS-dev
"""

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Sys Path Configuration                            #
# ———————————————————————————————————————————————————————————————————————————— #
import sys
from pathlib import Path

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

# Specify GPU to use (e.g., GPU:0, CPU:-1):
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Disable all auto-JIT clustering at the process level:
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"

# Enable deterministic operations for reproducibility:
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# If it fails to determine best cudnn convolution algorithm:
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR):
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# %%—————————————————————————————————————————————————————————————————————————— #
#                                    Imports                                   #
# ———————————————————————————————————————————————————————————————————————————— #
from _imports import *  # Centralized imports for the project

# %%—————————————————————————————————————————————————————————————————————————— #
#                                Run Parameters                                #
# ———————————————————————————————————————————————————————————————————————————— #
# Epoch sweep values to test.
EPOCHS = [33, 34, 35, 36, 37]
BATCH_SIZE = 64

# Top-k value to train and report.
TOP_K = 1

# Validation mode:
# - False: train on full S008 without validation.
# - True: split S008 into 80/20, then train on merged 100% (80+20)
#   while still using the 20% split as validation (100/20 mode).
USE_VALIDATION_SPLIT = True

DATA_SEED = 0
TRAIN_SEED = 0

# Global scaler fit mode:
# - True: fit scaler with all S008 coordinates (current implementation).
# - False: fit scaler with only 80% of S008 coordinates (legacy-like behavior).
USE_FULL_S008_FOR_SCALER = True

# Set Python, NumPy, Keras and TensorFlow seeds
set_random_seed(TRAIN_SEED)

# Reproducibility settings for TensorFlow:
# Note: must have same inputs and hardware
# Warning: this affects overall performance
tf.config.experimental.enable_op_determinism()

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = False

POLICY = mixed_precision.Policy("float32")
mixed_precision.set_global_policy(POLICY)

# Include top-k configuration in the run directory for easy experiment tracking.
RUN_DIR = f"runs/sweep/{get_caller_stem()}_topk_{TOP_K}"  # (e.g. "runs/train_1_topk_5")


# %%—————————————————————————————————————————————————————————————————————————— #
#                        Data Loading and Preprocessing                        #
# ———————————————————————————————————————————————————————————————————————————— #
@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(layers.Layer):
    """Convert the raw LiDAR tensor into a 4-channel semantic one-hot map.

    Args:
        name (str | None): Optional layer name forwarded to Keras. If ``None``,
            Keras assigns a default name.

    Returns:
        tf.Tensor: A float32 tensor with shape ``(batch, 20, 200, 4)`` encoding
        base station, vehicle, obstacle, and free-space channels.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic one-hot encoding to the LiDAR grid.

        Args:
            inputs (tf.Tensor): Raw LiDAR tensor with shape ``(batch, 20, 200, 10)``
                where each channel encodes semantic values in ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)``.
        """

        # Build semantic channels using boolean reductions to preserve the original logic.
        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)

    def get_config(self) -> dict:
        """Return the layer configuration for serialization.

        Returns:
            dict: Serializable configuration for the layer.
        """

        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(layers.Layer):
    """Tile 2D coordinates to align with the flattened LiDAR sequence length.

    Args:
        grid_length (int): Length of the flattened LiDAR grid (e.g., ``20 * 200``).
        name (str | None): Optional layer name forwarded to Keras. If ``None``,
            Keras assigns a default name.

    Returns:
        tf.Tensor: A float32 tensor with shape ``(batch, grid_length, 2)``.
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Tile coordinates across the flattened LiDAR sequence.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled tensor with shape ``(batch, grid_length, 2)``.
        """

        # Expand and tile to align coordinates with each LiDAR token.
        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict:
        """Return the layer configuration for serialization.

        Returns:
            dict: Serializable configuration for the layer.
        """

        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


def load_training_data(
    *,
    s008_path: str,
    s009_path: str,
    use_full_s008_for_scaler: bool,
    use_validation_split: bool,
    validation_size: float = 0.2,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    StandardScaler,
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Load and preprocess data for full-s008 training and s009 evaluation.

    Args:
        s008_path (str): Path to the baseline S008 dataset folder.
        s009_path (str): Path to the baseline S009 dataset folder.
        use_full_s008_for_scaler (bool): Controls which S008 subset is used to
            fit the coordinate scaler.
            If ``True``, the scaler is fit on all S008 coordinates, which matches
            this script's full-s008 training setup and typically gives more
            stable mean/variance estimates because it uses more samples.
            If ``False``, the scaler is fit on a deterministic 80% split of S008
            coordinates (random_state=``DATA_SEED``), which matches the legacy
            train/validation-style scaler fitting behavior but may produce
            slightly different normalization statistics due to fewer fit samples.
        use_validation_split (bool): Controls whether an explicit S008
            validation split is created.
            If ``True``, the function first creates an 80/20 split of S008 using
            ``validation_size`` and ``DATA_SEED``, then builds the effective
            training set by merging train+validation back together (100%), while
            still returning the 20% split as validation for monitoring
            (100/20 mode). This intentionally reuses validation samples in
            training and can overestimate generalization in validation metrics.
            If ``False``, no validation split is created, and training uses the
            full S008 set without validation outputs.
        validation_size (float): Fraction of S008 reserved for validation when
            ``use_validation_split`` is ``True``. Must be in the open interval
            ``(0.0, 1.0)``.

    Returns:
        tuple: A tuple containing:
            - Full S008 training data ``(x_lidar_train, x_coord_train, y_train)``.
            - Optional S008 validation data
              ``(x_lidar_val, x_coord_val, y_val)`` if enabled, otherwise ``None``.
            - The fitted ``StandardScaler`` for coordinates.
            - Full S009 baseline data ``(x_lidar_s009, x_coord_s009, y_s009)``
              after scaling for evaluation.

    Raises:
        ValueError: If loaded arrays do not align in length.
        ValueError: If ``validation_size`` is not in ``(0.0, 1.0)`` when
            ``use_validation_split`` is enabled.
    """

    # Load baseline S008/S009 arrays and convert beam tensors to sparse labels.
    (
        s008_coord_input,
        s008_lidar_input,
        s008_y,
        s009_coord_input,
        s009_lidar_input,
        s009_y,
    ) = load_dataset_sparse_labels(s008_path=s008_path, s009_path=s009_path)

    if use_validation_split:
        # Validate split ratio early to fail fast on misconfiguration.
        if not 0.0 < validation_size < 1.0:
            raise ValueError("validation_size must be in the open interval (0.0, 1.0).")

        # Create a deterministic 80/20 split for validation bookkeeping.
        (
            x_s008_lidar_train_80,
            x_s008_lidar_val,
            x_s008_coord_train_80,
            x_s008_coord_val,
            y_s008_train_80,
            y_s008_val,
        ) = train_test_split(
            s008_lidar_input,
            s008_coord_input,
            s008_y,
            test_size=validation_size,
            random_state=DATA_SEED,
            shuffle=True,
        )

        # Build the requested 100/20 mode by merging the validation samples back
        # into training, while still returning the same 20% split for monitoring.
        x_s008_lidar_train = np.concatenate([x_s008_lidar_train_80, x_s008_lidar_val], axis=0)
        x_s008_coord_train = np.concatenate([x_s008_coord_train_80, x_s008_coord_val], axis=0)
        y_s008_train = np.concatenate([y_s008_train_80, y_s008_val], axis=0)
    else:
        # Use the complete S008 split for training with no validation branch.
        x_s008_lidar_train = s008_lidar_input
        x_s008_coord_train = s008_coord_input
        y_s008_train = s008_y

        # Keep a consistent return type by setting validation outputs to None.
        x_s008_lidar_val = None
        x_s008_coord_val = None
        y_s008_val = None

    # Choose scaler fitting data according to the global mode flag.
    # This keeps the training set unchanged (always full S008) while allowing
    # normalization to mimic either the legacy 80% fit or the current full fit.
    if use_full_s008_for_scaler:
        scaler_fit_coords = x_s008_coord_train
    else:
        scaler_fit_coords, _ = train_test_split(
            x_s008_coord_train,
            train_size=0.8,
            shuffle=True,
            random_state=DATA_SEED,
        )

    # Fit the scaler using the selected S008 subset and apply it consistently
    # to both the full S008 training coordinates and S009 evaluation coordinates.
    coord_scaler = StandardScaler()
    coord_scaler.fit(scaler_fit_coords)

    x_s008_coord_train = coord_scaler.transform(x_s008_coord_train)
    if use_validation_split and x_s008_coord_val is not None:
        x_s008_coord_val = coord_scaler.transform(x_s008_coord_val)
    s009_coord_input = coord_scaler.transform(s009_coord_input)

    # Validate dataset alignment to avoid silent mis-training.
    if (
        x_s008_lidar_train.shape[0] != x_s008_coord_train.shape[0]
        or x_s008_lidar_train.shape[0] != y_s008_train.shape[0]
    ):
        raise ValueError("Loaded dataset arrays are misaligned. Check preprocessing and inputs.")

    if use_validation_split and (
        x_s008_lidar_val is None
        or x_s008_coord_val is None
        or y_s008_val is None
        or x_s008_lidar_val.shape[0] != x_s008_coord_val.shape[0]
        or x_s008_lidar_val.shape[0] != y_s008_val.shape[0]
    ):
        raise ValueError("Validation dataset arrays are misaligned. Check preprocessing and inputs.")

    return (
        (x_s008_lidar_train, x_s008_coord_train, y_s008_train),
        (
            (x_s008_lidar_val, x_s008_coord_val, y_s008_val)
            if use_validation_split
            else None
        ),
        coord_scaler,
        (s009_lidar_input, s009_coord_input, s009_y),
    )


def build_training_metric(top_k: int) -> tf.keras.metrics.Metric:
    """Create the single top-k metric used for model compilation.

    Args:
        top_k (int): The ``k`` value for top-k accuracy. Must be a positive
            integer. When ``top_k`` is 1, this matches standard sparse
            categorical accuracy but is reported explicitly for clarity.

    Returns:
        tf.keras.metrics.Metric: Sparse top-k accuracy metric configured for
            the requested ``k``.

    Raises:
        ValueError: If ``top_k`` is not a positive integer.
    """

    # Validate input up front to avoid silent misconfiguration.
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("TOP_K must be a positive integer.")

    return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=top_k,
        name=f"top_{top_k}_accuracy",
    )


def validate_epoch_sweep(epochs: list[int]) -> list[int]:
    """Validate and normalize epoch values used in the sweep.

    Args:
        epochs (list[int]): Epoch counts to test.
            Each value must be a strictly positive integer. Repeated values are
            removed while preserving the first occurrence order so the sweep is
            deterministic and avoids duplicate training runs.

    Returns:
        list[int]: A deduplicated, ordered list of positive epoch counts.

    Raises:
        ValueError: If ``epochs`` is empty.
        ValueError: If any value is not a strictly positive integer.

    Examples:
        >>> validate_epoch_sweep([5, 10, 5, 20])
        [5, 10, 20]
    """

    if not epochs:
        raise ValueError("EPOCHS must contain at least one value.")

    unique_epochs: list[int] = []
    seen_epochs: set[int] = set()

    for value in epochs:
        # Fail fast on invalid values so training does not start in a bad state.
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Each value in EPOCHS must be a positive integer.")

        # Preserve user-provided order while skipping duplicates.
        if value not in seen_epochs:
            unique_epochs.append(value)
            seen_epochs.add(value)

    return unique_epochs


def get_last_history_metric(history_dict: dict[str, list[float]], metric_name: str) -> float:
    """Return the last value of a history metric or NaN when unavailable.

    Args:
        history_dict (dict[str, list[float]]): Keras history mapping where keys
            are metric names and values are per-epoch measurements.
        metric_name (str): Metric key to fetch.

    Returns:
        float: Last value for ``metric_name``.
            If the metric key does not exist or has no values, returns
            ``np.nan`` so downstream CSV writing keeps a consistent numeric
            schema instead of raising exceptions.
    """

    metric_values = history_dict.get(metric_name)
    if metric_values is None or len(metric_values) == 0:
        return float(np.nan)

    return float(metric_values[-1])


# %%—————————————————————————————————————————————————————————————————————————— #
#                               Model Definition                               #
# ———————————————————————————————————————————————————————————————————————————— #
def build_model(show_summary: bool = False) -> Model:
    """Build the fixed CNN1D model that matches nas_cnn1d_v7.0.

    Args:
        show_summary (bool): If ``True``, prints the Keras model summary to
            stdout, which can add overhead during repeated runs. If ``False``,
            skips printing to keep training logs minimal.

    Returns:
        Model: Compiled Keras model that mirrors the nas_cnn1d_v7.0 top-1 trial
        architecture and hyperparameters. The model reports top-k accuracy based
        on the global ``TOP_K`` value.
    """

    # Use a deterministic initializer to match the original NAS trial setup.
    initializer = tf.keras.initializers.GlorotUniform(seed=TRAIN_SEED)

    # —————————————————————————————————— Inputs —————————————————————————————————— #
    # LiDAR input follows the baseline tensor shape.
    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")

    # Inline one-hot encoding that matches the NAS pre-processing pipeline.
    one_hot_lidar = LidarToOneHot(name="lidar_transform_to_one_hot")(x_lidar_input)

    # Flatten the LiDAR grid into a 1D sequence of 4-channel tokens.
    x_lidar_flat: layers.Layer = layers.Reshape(
        (20 * 200, 4),
        name="lidar_flatten_4_channels",
    )(one_hot_lidar)

    # GPS input contains the 2D coordinates.
    x_coord_input = layers.Input(shape=(2,), name="coord_input")

    # Tile coordinates to align with the 1D LiDAR sequence length.
    x_coord: layers.Layer = CoordTileFlat(20 * 200, name="coord_tile_flat")(x_coord_input)

    # Fuse LiDAR and coordinate channels for convolutional processing.
    combined = layers.Concatenate(axis=-1, name="combine_lidar_coord")([x_lidar_flat, x_coord])

    # ——————————————————————————————— Hidden Layers —————————————————————————————— #
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

    x = layers.Dense(
        175,
        kernel_initializer=initializer,
        name="dense_0",
    )(x)
    x = layers.Activation("tanh", name="dense_0_act")(x)
    x = layers.Dropout(0.1, name="dense_0_dropout")(x)

    outputs = layers.Dense(
        256,
        activation="softmax",
        kernel_initializer=initializer,
        name="output",
    )(x)

    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))

    if show_summary:
        model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0028523343462769487),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[build_training_metric(TOP_K)],
        jit_compile=USE_JIT_COMPILE,
    )

    return model


# %%—————————————————————————————————————————————————————————————————————————— #
#                            Training and Evaluation                           #
# ———————————————————————————————————————————————————————————————————————————— #
logs_dir = ""

try:
    # Create the standard study directory structure so logs/results stay
    # consistent with the rest of the project organization.
    (
        study_dir,
        args_dir,
        fig_dir,
        backup_dir,
        history_dir,
        scaler_dir,
        model_dir,
        logs_dir,
        tensorboard_dir,
    ) = init_study_dirs(RUN_DIR, study_name="epochs_sweep_training")

    epochs_to_test = validate_epoch_sweep(EPOCHS)

    # Load data once because all sweep runs use the same split and scaler logic.
    (
        (x_lidar_train, x_coord_train, y_train),
        val_data,
        _coord_scaler,
        (x_lidar_s009, x_coord_s009, y_s009),
    ) = load_training_data(
        s008_path="./data/s008",
        s009_path="./data/s009",
        use_full_s008_for_scaler=USE_FULL_S008_FOR_SCALER,
        use_validation_split=USE_VALIDATION_SPLIT,
    )

    results_rows: list[dict[str, float | int]] = []

    for index, epochs_value in enumerate(epochs_to_test):
        # Reset graph/session between runs to avoid residual state leakage.
        tf.keras.backend.clear_session()
        set_random_seed(TRAIN_SEED)

        # Only print model summary in the first run to keep logs readable.
        model = build_model(show_summary=(index == 0))

        fit_kwargs: dict[str, Any] = {
            "x": (x_lidar_train, x_coord_train),
            "y": y_train,
            "epochs": epochs_value,
            "batch_size": BATCH_SIZE,
            "callbacks": [callbacks.TerminateOnNaN()],
            "verbose": 2,
        }

        if USE_VALIDATION_SPLIT and val_data is not None:
            x_lidar_val, x_coord_val, y_val = val_data
            fit_kwargs["validation_data"] = ((x_lidar_val, x_coord_val), y_val)

        history = model.fit(**fit_kwargs)

        # Evaluate each trained model on full S009 to provide the test-side metric.
        s009_metrics = model.evaluate(
            (x_lidar_s009, x_coord_s009),
            y_s009,
            batch_size=BATCH_SIZE,
            verbose=2,
            return_dict=True,
        )

        row: dict[str, float | int] = {
            "epochs": int(epochs_value),
            "train_loss": get_last_history_metric(history.history, "loss"),
            f"train_top_{TOP_K}_accuracy": get_last_history_metric(
                history.history,
                f"top_{TOP_K}_accuracy",
            ),
            "test_loss": float(s009_metrics.get("loss", np.nan)),
            f"test_top_{TOP_K}_accuracy": float(
                s009_metrics.get(f"top_{TOP_K}_accuracy", np.nan)
            ),
        }

        if USE_VALIDATION_SPLIT:
            row["val_loss"] = get_last_history_metric(history.history, "val_loss")
            row[f"val_top_{TOP_K}_accuracy"] = get_last_history_metric(
                history.history,
                f"val_top_{TOP_K}_accuracy",
            )

        results_rows.append(row)

    # Save one CSV row per epoch value tested.
    epochs_sweep_path = os.path.join(history_dir, "epochs_sweep_results.csv")
    pd.DataFrame(results_rows).to_csv(epochs_sweep_path, index=False)

    print(f"\nEpoch sweep CSV saved to: {epochs_sweep_path}\n")

except Exception as e:
    print(f"\n An error occurred: {e}\n")
    traceback.print_exc()

    if logs_dir:
        with open(os.path.join(logs_dir, "training_error.log"), "a") as f:
            f.write(f"An error occurred during epoch sweep training:\n{e}\n{traceback.format_exc()}\n\n")
