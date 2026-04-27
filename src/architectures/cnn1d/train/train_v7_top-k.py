"""
Model Training

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
EPOCHS = 50
BATCH_SIZE = 64

# Top-k value to train and report.
TOP_K = 1

DATA_SEED = 0
TRAIN_SEED = 0

# Set Python, NumPy, Keras and TensorFlow seeds
set_random_seed(TRAIN_SEED)

# Reproducibility settings for TensorFlow:
# Note: must have same inputs and hardware
# Warning: this affects overall performance
tf.config.experimental.enable_op_determinism()

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = False

POLICY = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# Set to an existing dir to resume training.
# Include top-k configuration in the run directory for easy experiment tracking.
RUN_DIR = f"runs/{get_caller_stem()}_topk_{TOP_K}"  # (e.g. "runs/train_1_topk_5")

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
    data_seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    StandardScaler,
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
              after scaling, matching the notebook preprocessing flow.

    Raises:
        ValueError: If the loaded arrays do not align in length, indicating a
        corrupted dataset or mismatched inputs.
    """

    # Load baseline S008/S009 arrays and convert beam tensors to sparse labels.
    (
        s008_coord_input,
        s008_lidar_input,
        s008_y_train,
        s009_coord_input,
        s009_lidar_input,
        s009_y,
    ) = load_dataset_sparse_labels(s008_path=s008_path, s009_path=s009_path)

    # Split S008 into train/validation sets as done in the NAS notebook.
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

    # Fit the scaler on the S008 training coordinates only, then reuse it for all splits.
    coord_scaler = StandardScaler()
    coord_scaler.fit(x_s008_coord_train)

    x_s008_coord_train = coord_scaler.transform(x_s008_coord_train)
    x_s008_coord_val = coord_scaler.transform(x_s008_coord_val)
    s009_coord_input = coord_scaler.transform(s009_coord_input)

    # Validate dataset alignment to avoid silent mis-training.
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

# %%—————————————————————————————————————————————————————————————————————————— #
#                               Model Definition                               #
# ———————————————————————————————————————————————————————————————————————————— #
def build_model(show_summary: bool = True) -> Model:
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

    # ———————————————————————————————————————————————————————————————————————————— #
    #                              Model Construction                              #
    # ———————————————————————————————————————————————————————————————————————————— #

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
    # First convolutional block.
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

    # Second convolutional block.
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

    # Global max pooling matches the NAS-selected pooling type.
    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)

    # Dense block selected by the NAS trial.
    x = layers.Dense(
        175,
        kernel_initializer=initializer,
        name="dense_0",
    )(x)
    x = layers.Activation("tanh", name="dense_0_act")(x)
    x = layers.Dropout(0.1, name="dense_0_dropout")(x)

    # —————————————————————————————————— Outputs ————————————————————————————————— #
    outputs = layers.Dense(
        256,
        activation="softmax",
        kernel_initializer=initializer,
        name="output",
    )(x)

    # —————————————————————————— Set Inputs and Outputs —————————————————————————— #
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))

    # ———————————————————————————————— Compilation ——————————————————————————————— #
    if show_summary:
        model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0028523343462769487),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[build_training_metric(TOP_K)],
        jit_compile=USE_JIT_COMPILE,  # For XLA speedup; avoid when determinism is required.
    )

    return model

# %%—————————————————————————————————————————————————————————————————————————— #
#                            Training and Evaluation                           #
# ———————————————————————————————————————————————————————————————————————————— #
try:
    # ——————————————————————————————————— Setup —————————————————————————————————— #
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
    ) = init_study_dirs(RUN_DIR, study_name="model_training")

    # Load data using the same preprocessing recipe as nas_cnn1d_v7.0.
    (
        (x_lidar_train, x_coord_train, y_train),
        (x_lidar_val, x_coord_val, y_val),
        coord_scaler,
        (x_lidar_s009, x_coord_s009, y_s009),
    ) = load_training_data(
        s008_path="./data/s008",
        s009_path="./data/s009",
        data_seed=DATA_SEED,
    )

    # Persist the scaler so evaluation uses the identical coordinate normalization.
    scaler_path = os.path.join(scaler_dir, "coord_scaler.pkl")
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(coord_scaler, scaler_file)

    # —————————————————————————————— Train the Model ————————————————————————————— #

    model = build_model(show_summary=True)

    history = model.fit(
        x=(x_lidar_train, x_coord_train),
        y=y_train,
        validation_data=((x_lidar_val, x_coord_val), y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks_model(
            backup_dir=os.path.join(backup_dir, "training"),
            checkpoint_dir=os.path.join(backup_dir, "checkpoints"),
            early_stopping_patience=10,
            reduce_lr_patience=3,
            #! Can cause high memory usage
            # tensorboard_logs=tensorboard_dir,
        ),
        verbose=2,
    )

    # Evaluate on full S009 baseline set to mirror the NAS notebook workflow.
    s009_metrics = model.evaluate(
        (x_lidar_s009, x_coord_s009),
        y_s009,
        batch_size=BATCH_SIZE,
        verbose=2,
        return_dict=True,
    )
    s009_loss = float(s009_metrics.get("loss", np.nan))
    s009_topk = float(s009_metrics.get(f"top_{TOP_K}_accuracy", np.nan))

    model.save(os.path.join(model_dir, "model.keras"))

    # ——————————————————————————————— Save history ——————————————————————————————— #
    history_path = os.path.join(history_dir, "history.csv")
    history_data = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
    }

    history_df = pd.DataFrame(history_data)
    history_df.to_csv(history_path, index=False)

    # ———————————————————————————————— Model Stats ——————————————————————————————— #
    write_model_stats_to_file(
        model=model,
        file_path=os.path.join(args_dir, "model_stats.txt"),
        batch_size=1,
        bytes_per_param=tf.dtypes.as_dtype(POLICY.variable_dtype).size,
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
        extra_attrs={
            "final_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
            "s009_loss": s009_loss,
            f"s009_top_{TOP_K}_accuracy": s009_topk,
        },
        test_runs=10,
        verbose=1,
    )
    # ———————————————————————————————————————————————————————————————————————————— #
except Exception as e:
    print(f"\n An error occurred: {e}\n")
    traceback.print_exc()

    with open(os.path.join(logs_dir, "training_error.log"), "a") as f:
        f.write(f"An error occurred during training:\n{e}\n{traceback.format_exc()}\n\n")
finally:
    # Clean up directories
    shutil.rmtree(backup_dir, ignore_errors=True)
    if not os.listdir(logs_dir):
        os.rmdir(logs_dir)
