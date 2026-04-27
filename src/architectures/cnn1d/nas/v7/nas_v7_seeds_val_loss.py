"""
NAS Optuna

- Authored by Matheus Ferreira Silva
- GitHub https://github.com/MatheusFS-dev

Check Optuna Integration documentation for more features:
https://optuna-integration.readthedocs.io/en/stable/reference/index.html
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

# Use CUDA async allocator to reduce fragmentation:
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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
from _imports import *  # Centralized file containing all imports

# %%—————————————————————————————————————————————————————————————————————————— #
#                                Run Parameters                                #
# ———————————————————————————————————————————————————————————————————————————— #
NUM_TRIALS = 500
EPOCHS = 100

SAMPLER_SEED = 0

# steps_per_execution can break validation when a loop has too few batches,
# leading to missing val_* logs or empty evaluate() returns.
# Ensure steps_per_execution <= min_batches across train, val, and eval loops.
# num_batches = ceil(num_samples / batch_size)
# Example: 1473 samples, batch_size=64 -> ceil(1473/64)=24 batches,
# so steps_per_execution must be <= 24.
STEPS_PER_EXECUTION = 1

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = True

# Top-k value to train and report.
TOP_K_METRIC = 10

# Number of top trials to save
TOP_K_TRIALS = 3

# Order to rank trials by:
# "ascending" -> the lowest value is the best
# "descending" -> the highest value is the best
ORDER = "ascending"

# Key to rank trials by:
# "value" -> objective trial value
# other e.g., "test_accuracy" -> user params
RANK_KEY = "s009_topk"

# Direction of optimization:
# "minimize" -> the lowest value is the best
# "maximize" -> the highest value is the best
DIRECTION = "minimize"

POLICY = mixed_precision.Policy("float32")
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# Set to an existing dir to resume training
RUN_DIR = f"runs/{get_caller_stem()}_topk_{TOP_K_METRIC}"  # (e.g. "runs/nas_1_topk_10")


# %%—————————————————————————————————————————————————————————————————————————— #
#                        Data Loading and Preprocessing                        #
# ———————————————————————————————————————————————————————————————————————————— #
@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(layers.Layer):
    """Convert the raw LiDAR tensor into a 4-channel semantic one-hot map.

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

        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(layers.Layer):
    """Tile 2D coordinates to align with the flattened LiDAR sequence length.

    Args:
        grid_length (int): Length of the flattened LiDAR grid (e.g., ``20 * 200``).
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        """Initialize the tiling layer with a fixed sequence length.

        Args:
            grid_length (int): Length of the flattened LiDAR grid sequence.
            **kwargs (Any): Additional keyword arguments forwarded to Keras.
        """

        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Tile coordinates across the flattened LiDAR sequence.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled tensor with shape ``(batch, grid_length, 2)``.
        """

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
    data_seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    StandardScaler,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Load and preprocess data following the nas_cnn1d_v7.0 notebook.

    Args:
        s008_path (str): Path to the baseline S008 dataset folder.
        s009_path (str): Path to the baseline S009 dataset folder.
        data_seed (int): Random seed used for the train/validation split to
            ensure deterministic shuffling and reproducibility.

    Returns:
        tuple: A tuple containing:
            - Training split ``(x_lidar_train, x_coord_train, y_train)``.
            - Validation split ``(x_lidar_val, x_coord_val, y_val)``.
            - The fitted ``StandardScaler`` for coordinates.
            - Full S009 baseline data ``(x_lidar_s009, x_coord_s009, y_s009)``
              after scaling.
            - Full S008 baseline data ``(x_lidar_s008, x_coord_s008, y_s008)``
              after scaling.

    Raises:
        ValueError: If the loaded arrays do not align in length, indicating a
        corrupted dataset or mismatched inputs.
    """

    (
        s008_coord_input,
        s008_lidar_input,
        s008_y_train,
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
        s008_y_train,
        test_size=0.2,
        random_state=data_seed,
        shuffle=True,
    )

    coord_scaler = StandardScaler()
    coord_scaler.fit(x_s008_coord_train)

    x_s008_coord_train = coord_scaler.transform(x_s008_coord_train)
    x_s008_coord_val = coord_scaler.transform(x_s008_coord_val)
    s009_coord_input = coord_scaler.transform(s009_coord_input)
    s008_coord_input = coord_scaler.transform(s008_coord_input)

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
        (s008_lidar_input, s008_coord_input, s008_y_train),
    )


# %%—————————————————————————————————————————————————————————————————————————— #
#                                Hyperparameters                               #
# ———————————————————————————————————————————————————————————————————————————— #


# %%—————————————————————————————————————————————————————————————————————————— #
#                               Model Definition                               #
# ———————————————————————————————————————————————————————————————————————————— #
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

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("TOP_K_METRIC must be a positive integer.")

    return tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=top_k,
        name=f"top_{top_k}_accuracy",
    )


def build_model(*, show_summary: bool = True, train_seed: int) -> tf.keras.Model:
    """Build the fixed CNN1D model that matches nas_cnn1d_v7.0.

    Args:
        show_summary (bool): If ``True``, prints the Keras model summary to
            stdout, which can add overhead during repeated runs. If ``False``,
            skips printing to keep training logs minimal.
        train_seed (int): Random seed used to initialize weights deterministically.

    Returns:
        tf.keras.Model: Compiled Keras model with fixed architecture and
        top-k metric configured by ``TOP_K_METRIC``.
    """

    # ———————————————————————————————————————————————————————————————————————————— #
    #                              Model Construction                              #
    # ———————————————————————————————————————————————————————————————————————————— #

    initializer = tf.keras.initializers.GlorotUniform(seed=train_seed)

    # —————————————————————————————————— Inputs —————————————————————————————————— #
    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")
    one_hot_lidar = LidarToOneHot(name="lidar_transform_to_one_hot")(x_lidar_input)
    x_lidar_flat: layers.Layer = layers.Reshape(
        (20 * 200, 4),
        name="lidar_flatten_4_channels",
    )(one_hot_lidar)

    x_coord_input = layers.Input(shape=(2,), name="coord_input")
    x_coord: layers.Layer = CoordTileFlat(20 * 200, name="coord_tile_flat")(x_coord_input)

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

    # ———————————————————————————————— Compilation ——————————————————————————————— #
    model.summary() if show_summary else None
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0028523343462769487),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[build_training_metric(TOP_K_METRIC)],
        jit_compile=USE_JIT_COMPILE,  # For XLA speedup, does not support determinism
        steps_per_execution=STEPS_PER_EXECUTION,
    )

    return model


# %%—————————————————————————————————————————————————————————————————————————— #
#                              Objective Function                              #
# ———————————————————————————————————————————————————————————————————————————— #
def objective(
    trial: optuna.Trial,
    **kwargs: Any,
) -> float:
    """
    Objective function for Optuna to optimize a Neural Network NN on any-input data.

    Args:
        trial (optuna.Trial): Current trial for hyperparameter suggestions.
        **kwargs: Additional keyword arguments.

    Returns:
        float: Final value used for optimization.
    """
    (print(f"Running trial {trial.number}..."), clear_session())

    # —————————————————————————————— Reproducibility ————————————————————————————— #
    DATA_SEED = trial.suggest_int("data_seed", 0, 10000)
    TRAIN_SEED = trial.suggest_int("train_seed", 0, 10000)

    # Set Python, NumPy, Keras and TensorFlow seeds
    set_random_seed(TRAIN_SEED)

    # ——————————————————————————————————— Setup —————————————————————————————————— #
    backup_dir = kwargs["backup"]
    model_dir = kwargs["model"]
    fig_dir = kwargs["fig"]
    tensorboard_dir = kwargs["tensorboard"]
    logs_dir = kwargs["logs"]
    history_dir = kwargs["history"]
    scaler_dir = kwargs["scaler"]

    # ———————————————————————————————————————————————————————————————————————————— #

    try:
        # ———————————————————————————————————————————————————————————————————————————— #
        #                              Data Preprocessing                              #
        # ———————————————————————————————————————————————————————————————————————————— #
        (
            (x_lidar_train, x_coord_train, y_train),
            (x_lidar_val, x_coord_val, y_val),
            coord_scaler,
            (x_lidar_s009, x_coord_s009, y_s009),
            (x_lidar_s008, x_coord_s008, y_s008),
        ) = load_training_data(
            s008_path="./data/s008",
            s009_path="./data/s009",
            data_seed=DATA_SEED,
        )

        scaler_path = os.path.join(scaler_dir, f"trial_{trial.number}.pkl")
        with open(scaler_path, "wb") as scaler_file:
            pickle.dump(coord_scaler, scaler_file)

        # ———————————————————————————————————————————————————————————————————————————— #
        #                        Model Construction and Training                       #
        # ———————————————————————————————————————————————————————————————————————————— #
        model = build_model(show_summary=False, train_seed=TRAIN_SEED)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        prune_model_by_config(
            trial=trial,
            model=model,
            thresholds={
                # "model_size": 50,  # Maximum model size in MB
                "memory_mb": 8000,  # Maximum memory training usage in MB
                # "param": 1e6,  # Maximum number of parameters
                # "flops": 1e9,  # Maximum number of FLOPs
            },
            bytes_per_param=BYTES_PER_PARAM,
            batch_size=batch_size,
        )

        # Check if steps_per_execution is valid
        validate_steps_per_execution(
            name=["x_lidar_train", "x_lidar_val"],
            data_size=[x_lidar_train.shape[0], x_lidar_val.shape[0]],
            batch_size=batch_size,
            steps_per_execution=STEPS_PER_EXECUTION,
        )

        history = model.fit(
            x=[x_lidar_train, x_coord_train],
            y=y_train,
            validation_data=([x_lidar_val, x_coord_val], y_val),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=get_callbacks_study(
                trial=trial,
                monitor="val_loss",
                early_stopping_patience=10,
                reduce_lr_patience=3,
                pruning_interval=5,
                #! Can cause high memory usage
                # tensorboard_logs=tensorboard_dir,
            ),
            verbose=2,
        )

        model.save(os.path.join(model_dir, f"trial_{trial.number}.keras"))

        # ———————————————————————————————————————————————————————————————————————————— #
        #                              Save Trial Results                              #
        # ———————————————————————————————————————————————————————————————————————————— #

        # ——————————————————————————— Model characteristics —————————————————————————— #
        set_user_attr_model_stats(
            trial=trial,
            model=model,
            bytes_per_param=BYTES_PER_PARAM,
            batch_size=1,  # ? Benchmarking per-sample stats
            test_runs=10,
            device="gpu/0",
            stats_to_measure=(
                "parameters",
                "model_size",
                "flops",
                "macs",
                "summary",
                "inference_latency",
                # "cpu_util_percent",
                # "cpu_power_rapl_w",
                # "ram_used_bytes",
                # "ram_util_percent",
                # "gpu_util_percent",
                # "gpu_mem_used_bytes",
                # "gpu_power_w",
            ),
            extra_attrs=None,
            verbose=1,
        )

        s009_loss, s009_topk = model.evaluate(
            [x_lidar_s009, x_coord_s009],
            y_s009,
            batch_size=batch_size,
            verbose=2,
        )
        s008_loss, s008_topk = model.evaluate(
            [x_lidar_s008, x_coord_s008],
            y_s008,
            batch_size=batch_size,
            verbose=2,
        )

        # ———————————————————————————————————————————————————————————————————————————— #
        #                               Extra Attributes                               #
        # ———————————————————————————————————————————————————————————————————————————— #
        if "minimize" in DIRECTION:
            # The best epoch is the one with the lowest validation loss.
            best_idx = int(np.argmin(history.history["val_loss"]))
        else:
            # The best epoch is the one with the highest validation loss.
            best_idx = int(np.argmax(history.history["val_loss"]))

        best_train_loss = history.history["loss"][best_idx]
        best_val_loss = history.history["val_loss"][best_idx]
        best_train_topk = history.history[f"top_{TOP_K_METRIC}_accuracy"][best_idx]
        best_val_topk = history.history[f"val_top_{TOP_K_METRIC}_accuracy"][best_idx]

        trial.set_user_attr("best_epoch", int(best_idx) + 1)
        trial.set_user_attr("best_train_loss", float(best_train_loss))
        trial.set_user_attr("best_val_loss", float(best_val_loss))
        trial.set_user_attr("best_train_topk", float(best_train_topk))
        trial.set_user_attr("best_val_topk", float(best_val_topk))
        trial.set_user_attr("s008_loss", float(s008_loss))
        trial.set_user_attr("s009_loss", float(s009_loss))
        trial.set_user_attr("s008_topk", float(s008_topk))
        trial.set_user_attr("s009_topk", float(s009_topk))

        # TODO: implement a k-fold cross-validation procedure and report mean and stddev of scores
        # That, only if the model was trained and not pruned early.

        # ? just a tip, implement an emsemble of the top-k models

        # ——————————————————————————————— Save history ——————————————————————————————— #
        history_path = os.path.join(history_dir, f"trial_{trial.number}.csv")
        history_data = {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            f"train_top_{TOP_K_METRIC}_accuracy": history.history[f"top_{TOP_K_METRIC}_accuracy"],
            f"val_top_{TOP_K_METRIC}_accuracy": history.history[f"val_top_{TOP_K_METRIC}_accuracy"],
        }

        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_path, index=False)

        # ————————————————————————— Finish the current trial ————————————————————————— #
        if len(history.history["val_loss"]) > 1:  # Termination Judgement Report
            report_cross_validation_scores(trial, scores=history.history["val_loss"])

        return float(best_val_loss)
    except ValueError as e:
        # Catch invalid model configurations
        # e.g., when a pooling operation results in negative dimension size
        if "Negative dimension size" in str(e):
            raise optuna.TrialPruned("Pruned, invalid pooling config") from e
        raise
    except Exception as e:
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
            force_crash_oom=5,  # Crash after X occurrences of OOM
        )


# %%—————————————————————————————————————————————————————————————————————————— #
#                                     Study                                    #
# ———————————————————————————————————————————————————————————————————————————— #
study = run_study(
    objective=objective,
    run_dir=RUN_DIR,
    num_trials=NUM_TRIALS,
    sampler_seed=SAMPLER_SEED,
    direction=DIRECTION,
    top_k=TOP_K_TRIALS,
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
        ("tensorboard", "trial_{trial_id}"),
        ("scaler", "trial_{trial_id}.pkl"),
    ],
    rename_paths=[
        ("model", ".keras"),
        ("fig", ".png"),
        ("history", ".csv"),
        ("scaler", ".pkl"),
    ],
    extra_attrs=[
        "best_epoch",
        "best_train_loss",
        "best_val_loss",
        "best_train_topk",
        "best_val_topk",
        "s008_loss",
        "s009_loss",
        "s008_topk",
        "s009_topk",
    ],
    variance_threshold=None,  # 1e-10
    prune_threshold=None,  # 50
    patience=None,  # 100
)
