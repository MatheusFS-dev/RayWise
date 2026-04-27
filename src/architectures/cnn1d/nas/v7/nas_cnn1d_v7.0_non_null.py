# %% [markdown]
# # NAS - Optuna
# 
# - **Authored by:** Matheus Ferreira Silva 
# - **GitHub:**: https://github.com/MatheusFS-dev

# %% [markdown]
# ## 1. Setup and Configuration

# %% [markdown]
# ### 1.1. Environment Variables

# %%
import os

# Specify GPU to use (e.g., GPU:0, CPU:-1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use CUDA async allocator to reduce fragmentation:
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Allow TensorFlow to allocate GPU memory as needed
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress TensorFlow logging (0: ALL, 1: INFO, 2: WARNING, 3: ERROR):
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # If it fails to determine best cudnn convolution algorithm
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# %%
# # Disable all auto-JIT clustering at the process level
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=-1"

# %% [markdown]
# ### 1.2. Imports

# %%
from _imports import * # Centralized file containing all imports
from _load_dataset import convert_to_sparse_labels

# %% [markdown]
# ### 1.3. GPU Management

# %%
get_gpu_info()

# %% [markdown]
# ## 2. Run Parameters 

# %%
NUM_TRIALS = 1000
EPOCHS = 50

SAMPLER_SEED = 0

STEPS_PER_EXECUTION = 1

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = True

# %%
# Number of top trials to save
TOP_K = 3

# Order to rank trials by:
# "ascending" -> the lowest value is the best
# "descending" -> the highest value is the best
ORDER = "descending"

# Key to rank trials by:
# "value" -> objective trial value
# other e.g., "test_accuracy" -> user params
RANK_KEY = "s009_accuracy"

# Direction of optimization:
# "minimize" -> the lowest value is the best
# "maximize" -> the highest value is the best
DIRECTION = "minimize"

# %%
POLICY = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# %%
# Set to an existing dir to resume training
RUN_DIR = f"runs/{get_caller_stem()}"  # (e.g. "runs/nas_1")

# %% [markdown]
# ## 3. Data Loading and Preprocessing

# %%
@tf.keras.utils.register_keras_serializable(package="raywise")
class LidarToOneHot(layers.Layer):
    """Convert raw LiDAR tensors into a 4-channel semantic representation.

    Args:
        name (str | None): Optional layer name forwarded to Keras. If ``None``,
            Keras assigns a default name.

    Returns:
        tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)`` containing
        base station, vehicle, obstacle, and free-space semantic channels.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply semantic one-hot style transformations to LiDAR values.

        Args:
            inputs (tf.Tensor): LiDAR tensor with shape ``(batch, 20, 200, 10)``
                and semantic values in ``{-2, -1, 0, 1}``.

        Returns:
            tf.Tensor: Float32 tensor with shape ``(batch, 20, 200, 4)``.
        """

        # Build each semantic map explicitly so the model graph remains serializable.
        base_station = tf.cast(tf.reduce_any(tf.equal(inputs, -2), axis=-1, keepdims=True), tf.float32)
        vehicle = tf.cast(tf.reduce_any(tf.equal(inputs, -1), axis=-1, keepdims=True), tf.float32)
        obstacle = tf.cast(tf.reduce_any(tf.equal(inputs, 1), axis=-1, keepdims=True), tf.float32)
        free_space = tf.cast(tf.reduce_all(tf.equal(inputs, 0), axis=-1, keepdims=True), tf.float32)
        return tf.concat([base_station, vehicle, obstacle, free_space], axis=-1)

    def get_config(self) -> dict:
        """Return layer configuration for Keras serialization.

        Returns:
            dict: Serializable layer configuration.
        """

        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="raywise")
class CoordTileFlat(layers.Layer):
    """Tile 2D coordinates to align with the flattened LiDAR token length.

    Args:
        grid_length (int): Number of LiDAR tokens after flattening the
            ``20 x 200`` grid.
        name (str | None): Optional layer name forwarded to Keras. If ``None``,
            Keras assigns a default name.

    Returns:
        tf.Tensor: Float32 tensor with shape ``(batch, grid_length, 2)``.
    """

    def __init__(self, grid_length: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.grid_length = int(grid_length)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Broadcast each coordinate pair across all flattened LiDAR tokens.

        Args:
            inputs (tf.Tensor): Coordinate tensor with shape ``(batch, 2)``.

        Returns:
            tf.Tensor: Tiled tensor with shape ``(batch, grid_length, 2)``.
        """

        return tf.tile(tf.expand_dims(inputs, axis=1), [1, self.grid_length, 1])

    def get_config(self) -> dict:
        """Return layer configuration for Keras serialization.

        Returns:
            dict: Serializable layer configuration.
        """

        config = super().get_config()
        config.update({"grid_length": self.grid_length})
        return config


def _load_baseline_split_arrays(
    *,
    split_name: str,
    beam_path: str,
    coord_path: str,
    lidar_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one baseline split and remove rows whose beam tensors are null.

    A null beam sample is a row whose entire ``(8, 32)`` beam tensor is zero.

    Args:
        split_name (str): Human-readable split identifier for logging.
        beam_path (str): Path to the ``.npz`` file containing
            ``output_classification``.
        coord_path (str): Path to the ``.npz`` file containing ``coordinates``.
        lidar_path (str): Path to the ``.npz`` file containing LiDAR ``input``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered
        ``(coordinates, lidar, sparse_labels)`` arrays.

    Raises:
        ValueError: If arrays have inconsistent lengths or no non-null samples
            remain after filtering.
    """

    beam_scores = np.load(beam_path)["output_classification"].astype(np.float32)
    coordinates = np.load(coord_path)["coordinates"].astype(np.float32)
    lidar_tensor = np.load(lidar_path)["input"].astype(np.float32)

    if not (beam_scores.shape[0] == coordinates.shape[0] == lidar_tensor.shape[0]):
        raise ValueError(
            f"{split_name} arrays are misaligned: "
            f"beams={beam_scores.shape[0]}, coords={coordinates.shape[0]}, lidar={lidar_tensor.shape[0]}"
        )

    # Keep only samples that contain at least one non-zero beam score.
    non_null_mask = np.any(beam_scores != 0.0, axis=(1, 2))
    kept_samples = int(np.count_nonzero(non_null_mask))
    removed_samples = int(non_null_mask.size - kept_samples)
    print(f"{split_name}: removed {removed_samples} null-beam samples; kept {kept_samples} samples.")

    if kept_samples == 0:
        raise ValueError(f"{split_name} has no non-null beam samples after filtering.")

    filtered_beams = beam_scores[non_null_mask]
    filtered_coords = coordinates[non_null_mask]
    filtered_lidar = lidar_tensor[non_null_mask]
    sparse_labels = convert_to_sparse_labels(filtered_beams)
    return filtered_coords, filtered_lidar, sparse_labels


def load_dataset_sparse_labels_non_null(
    *,
    s008_path: str,
    s009_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load S008/S009 sparse-label data with non-null and class-remap filtering.

    The preprocessing removes null-beam samples, remaps S008 classes to a
    contiguous range, and filters S009 to only classes seen in S008.

    Args:
        s008_path (str): Path to baseline S008 dataset folder.
        s009_path (str): Path to baseline S009 dataset folder.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        ``(s008_coord_input, s008_lidar_input, s008_y_train, s009_coord_input,
        s009_lidar_input, s009_y, num_non_null_beams)``.

    Raises:
        ValueError: If no non-null classes are available in S008 after filtering.
    """

    s008_train_coord, s008_train_lidar, s008_train_y = _load_baseline_split_arrays(
        split_name="s008_train",
        beam_path=os.path.join(s008_path, "beam_output", "beam_output_train.npz"),
        coord_path=os.path.join(s008_path, "coord_input", "coord_train.npz"),
        lidar_path=os.path.join(s008_path, "lidar_input", "lidar_train.npz"),
    )
    s008_val_coord, s008_val_lidar, s008_val_y = _load_baseline_split_arrays(
        split_name="s008_val",
        beam_path=os.path.join(s008_path, "beam_output", "beam_output_val.npz"),
        coord_path=os.path.join(s008_path, "coord_input", "coord_val.npz"),
        lidar_path=os.path.join(s008_path, "lidar_input", "lidar_val.npz"),
    )
    s009_coord_input, s009_lidar_input, s009_y = _load_baseline_split_arrays(
        split_name="s009",
        beam_path=os.path.join(s009_path, "beam_output", "beam_output.npz"),
        coord_path=os.path.join(s009_path, "coord_input", "coord_input.npz"),
        lidar_path=os.path.join(s009_path, "lidar_input", "lidar_input.npz"),
    )

    s008_coord_input = np.concatenate((s008_train_coord, s008_val_coord), axis=0)
    s008_lidar_input = np.concatenate((s008_train_lidar, s008_val_lidar), axis=0)
    s008_y_train = np.concatenate((s008_train_y, s008_val_y), axis=0)

    present_beams_s008 = np.unique(s008_y_train.astype(np.int64))
    if present_beams_s008.size == 0:
        raise ValueError("No valid S008 beam classes were found after non-null preprocessing.")

    # Remap S008 labels to contiguous ids to minimize output dimensionality.
    label_map = np.full(int(present_beams_s008.max()) + 1, -1, dtype=np.int64)
    label_map[present_beams_s008] = np.arange(present_beams_s008.size, dtype=np.int64)
    s008_y_train = label_map[s008_y_train.astype(np.int64)]

    # Keep S009 only where labels exist in the S008 non-null class space.
    valid_s009_mask = s009_y.astype(np.int64) < label_map.shape[0]
    valid_s009_mask &= label_map[s009_y.astype(np.int64)] >= 0
    s009_lidar_input = s009_lidar_input[valid_s009_mask]
    s009_coord_input = s009_coord_input[valid_s009_mask]
    s009_y = label_map[s009_y[valid_s009_mask].astype(np.int64)]

    num_non_null_beams = int(present_beams_s008.size)
    print(f"Using {num_non_null_beams} non-null beam classes (from S008 examples).")

    return (
        s008_coord_input,
        s008_lidar_input,
        s008_y_train,
        s009_coord_input,
        s009_lidar_input,
        s009_y,
        num_non_null_beams,
    )


(
    s008_coord_input,
    s008_lidar_input,
    s008_y_train,
    s009_coord_input,
    s009_lidar_input,
    s009_y,
    num_non_null_beams,
) = load_dataset_sparse_labels_non_null(s008_path="./data/s008", s009_path="./data/s009")


# %% [markdown]
# ## Hyperparameters

# %%
kparams = KParams(
    activation_choices={
        "relu": tf.keras.activations.relu,
        "gelu": tf.keras.activations.gelu,
        "silu": tf.keras.activations.silu,
        "elu": tf.keras.activations.elu,
        "sigmoid": tf.keras.activations.sigmoid,
        "tanh": tf.keras.activations.tanh,
        "none": None,
    },
    regularizer_choices={
        # "l2": tf.keras.regularizers.l2,
        "none": None,
    },
    optimizer_choices={
        # "sgd": tf.keras.optimizers.SGD(momentum=0.9),
        # "adam": tf.keras.optimizers.Adam(),
        "adamw": tf.keras.optimizers.AdamW(weight_decay=1e-4),
        # "lion": tf.keras.optimizers.Lion(beta_1=0.9, beta_2=0.99),
        # "rmsprop": tf.keras.optimizers.RMSprop(),
    },
    # scaler_choices={
    #     "standard": StandardScaler,
    #     "minmax_0_1": lambda: MinMaxScaler(feature_range=(0, 1)),
    #     "minmax_-1_1": lambda: MinMaxScaler(feature_range=(-1, 1)),
    # },
    learning_rate=(1e-4, 1e-2),
)

# %% [markdown]
# ## 5. Model Definition

# %%
def build_model(
    trial: optuna.Trial,
    kparams: dict,
    *,
    show_summary: bool = True,
    **kwargs: Any,
) -> tf.keras.Model:

    train_seed = kwargs.get("train_seed")

    # ———————————————————————————————————————————————————————————————————————————— #
    #                              Model Construction                              #
    # ———————————————————————————————————————————————————————————————————————————— #
    initializer = tf.keras.initializers.GlorotUniform(
        seed=train_seed,
    )

    # ———————————————————————————————— LiDAR Input ——————————————————————————————— #
    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")

    # Serializable replacement for the previous Lambda LiDAR transform.
    one_hot_lidar = LidarToOneHot(name="lidar_transform_to_one_hot")(x_lidar_input)
    # -> (batch, 20, 200, 4)

    # Flatten the 20×200 grid into a 4000-length sequence with the 4 channels
    x_lidar_flat: layers.Layer = layers.Reshape((20 * 200, 4), name="lidar_flatten_4_channels")(one_hot_lidar)

    # ———————————————————————————————— GPS Input ———————————————————————————————— #
    # Input for coordinate data (e.g., shape: (2,))
    x_coord_input = layers.Input(shape=(2,), name="coord_input")

    # Serializable replacement for the previous Lambda coordinate tiling.
    x_coord: layers.Layer = CoordTileFlat(20 * 200, name="coord_tile_flat")(x_coord_input)

    # ————————————————————————————— Combine Branches ————————————————————————————— #
    # Fuse channels:  (batch,4000,4) + (batch,4000,2) → (batch,4000,6)
    combined = layers.Concatenate(axis=-1, name="combine_lidar_coord")([x_lidar_flat, x_coord])

    # ———————————————————————————————————— CNN ——————————————————————————————————— #
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 2)

    for i in range(num_conv_layers):
        x = build_cnn1d(
            trial=trial,
            kparams=kparams,
            x=combined if i == 0 else x,  # Use combined only for the first layer
            name_prefix=f"conv1d_{i}",
            # Filters
            filters_range=trial.suggest_categorical(f"conv1d_{i}_filters", [64, 128, 256, 512, 1024]),
            # filters_step=40,
            # Kernel size
            kernel_size_range=(1, 9),
            kernel_size_step=1,
            # Other parameters
            # strides=trial.suggest_int(f"conv1d_{i}_strides", 1, 2),
            kernel_initializer=initializer,
        )
        #! pool size = 1 means no downsampling
        pool_size = trial.suggest_int(f"pool_size_{i}", 1, 4, step=1)
        x = layers.MaxPooling1D(pool_size=pool_size, name=f"max_pool_{i}")(x)

    pooling_type = trial.suggest_categorical("pooling_type", ["flatten", "max", "average"])
    if pooling_type == "flatten":
        x = layers.Flatten(name="flatten_cnn_output")(x)
    elif pooling_type == "max":
        x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    else:
        x = layers.GlobalAveragePooling1D(name="global_avg_pooling")(x)

    num_dense_layers = trial.suggest_int("num_dense_layers", 0, 1)
    for i in range(num_dense_layers):
        x = build_dnn(
            trial=trial,
            kparams=kparams,
            x=x,
            name_prefix=f"dense_{i}",
            units_range=(25, 500),
            units_step=25,
            dropout_rate_range=(0.0, 0.5),
            dropout_rate_step=0.1,
            kernel_initializer=initializer,
        )

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(
        num_non_null_beams,
        activation="softmax",
        name="output",
        kernel_initializer=initializer,
    )(x)

    # —————————————————————————— Set Inputs and Outputs —————————————————————————— #
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))

    # ———————————————————————————————— Compilation ——————————————————————————————— #
    model.summary() if show_summary else None

    model.compile(
        optimizer=kparams.get_optimizer(trial),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        jit_compile=USE_JIT_COMPILE,  # For XLA speedup, does not support determinism
        steps_per_execution=STEPS_PER_EXECUTION,
    )

    return model

# %% [markdown]
# ## 6. Objective Function

# %%
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
    DATA_SEED = 0
    TRAIN_SEED = 0

    # Set Python, NumPy, Keras and TensorFlow seeds
    set_random_seed(TRAIN_SEED)

    # ——————————————————————————————————— Setup —————————————————————————————————— #
    global s009_coord_input, s009_lidar_input, s009_y
    global s008_coord_input, s008_lidar_input, s008_y_train

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
        random_state=DATA_SEED,
        shuffle=True,
    )

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
        coord_scaler = StandardScaler()

        coord_scaler.fit(x_s008_coord_train)
        x_s008_coord_train = coord_scaler.transform(x_s008_coord_train)
        x_s008_coord_val = coord_scaler.transform(x_s008_coord_val)
        s009_coord_input = coord_scaler.transform(s009_coord_input)
        s008_coord_input = coord_scaler.transform(s008_coord_input)

        scaler_path = os.path.join(scaler_dir, f"trial_{trial.number}.pkl")
        with open(scaler_path, "wb") as scaler_file:
            pickle.dump(coord_scaler, scaler_file)

        # ———————————————————————————————————————————————————————————————————————————— #
        #                        Model Construction and Training                       #
        # ———————————————————————————————————————————————————————————————————————————— #
        model = build_model(
            trial=trial,
            kparams=kparams,
            show_summary=False,
            train_seed=TRAIN_SEED,
        )
        BATCH_SIZE = 64

        prune_model_by_config(
            trial=trial,
            model=model,
            thresholds={
                "model_size": 350,  # Maximum model size in MB
                "memory_mb": 9000,  # Maximum memory training usage in MB
                "param": 2e8,  # Maximum number of parameters
                "flops": 1.2e10,  # Maximum number of FLOPs
            },
            bytes_per_param=BYTES_PER_PARAM,
            batch_size=BATCH_SIZE,
        )

        history = model.fit(
            x=[x_s008_lidar_train, x_s008_coord_train],
            y=y_s008_train,
            validation_data=([x_s008_lidar_val, x_s008_coord_val], y_s008_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks_study(
                trial=trial,
                monitor="val_loss",
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
            batch_size=BATCH_SIZE,
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

        # Evaluate on full s009
        s009_loss, s009_acc = model.evaluate(
            [s009_lidar_input, s009_coord_input], s009_y, batch_size=BATCH_SIZE, verbose=2
        )

        # Evaluate on s008
        s008_loss, s008_acc = model.evaluate(
            [s008_lidar_input, s008_coord_input], s008_y_train, batch_size=BATCH_SIZE, verbose=2
        )

        # ———————————————————————————————————————————————————————————————————————————— #
        #                               Extra Attributes                               #
        # ———————————————————————————————————————————————————————————————————————————— #
        # # Choose best epoch based on validation loss
        # if "minimize" in DIRECTION:
        #     # The best epoch is the one with the lowest validation loss
        #     best_idx = int(np.argmin(history.history["val_loss"]))
        # else:
        #     # The best epoch is the one with the highest validation loss
        #     best_idx = int(np.argmax(history.history["val_loss"]))
            
        # Choose best epoch based on validation accuracy
        best_idx = int(np.argmax(history.history["val_accuracy"]))

        best_train_loss = float(history.history["loss"][best_idx])
        best_val_loss = float(history.history["val_loss"][best_idx])
        best_train_acc = float(history.history["accuracy"][best_idx])
        best_val_acc = float(history.history["val_accuracy"][best_idx])

        trial.set_user_attr("best_epoch", best_idx + 1)
        trial.set_user_attr("best_train_loss", best_train_loss)
        trial.set_user_attr("best_val_loss", best_val_loss)
        trial.set_user_attr("s008_loss", float(s008_loss))
        trial.set_user_attr("s009_loss", float(s009_loss))

        trial.set_user_attr("best_train_accuracy", best_train_acc)
        trial.set_user_attr("best_val_accuracy", best_val_acc)
        trial.set_user_attr("s008_accuracy", float(s008_acc))
        trial.set_user_attr("s009_accuracy", float(s009_acc))

        # ——————————————————————————————— Save history ——————————————————————————————— #
        history_path = os.path.join(history_dir, f"trial_{trial.number}.csv")

        # Create a DataFrame with all history data
        history_data = {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "train_accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
        }

        # Convert to DataFrame and save as CSV
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_path, index=False)

        # ————————————————————————— Finish the current trial ————————————————————————— #
        if len(history.history["val_loss"]) > 1:  # Termination Judgement Report
            report_cross_validation_scores(trial, scores=history.history["val_loss"])

        return best_val_loss  # Value to minimize or maximize
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
            force_crash_oom=None,  # Crash after X occurrences of OOM
        )

# %% [markdown]
# ## Main

# %%
# # Search space:
# base_path = f"{RUN_DIR}/search_space/"
# (
#     x_s008_lidar_train,
#     x_s008_lidar_val,
#     x_s008_coord_train,
#     x_s008_coord_val,
#     y_s008_train,
#     y_s008_val,
# ) = train_test_split(
#     s008_lidar_input,
#     s008_coord_input,
#     s008_y_train,
#     test_size=0.2,
#     random_state=0,
#     shuffle=True,
# )


# plot_model_param_distribution(
#     lambda trial: build_model(
#         trial=trial,
#         kparams=kparams,
#         show_summary=False,
#         train_seed=0,
#     ),
#     benchmark_training=False,
#     fit_x=(x_s008_lidar_train, x_s008_coord_train),
#     fit_y=y_s008_train,
#     fit_validation_data=((x_s008_lidar_val, x_s008_coord_val), y_s008_val),
#     bytes_per_param=tf.dtypes.as_dtype(POLICY.variable_dtype).size,
#     batch_size=64,
#     n_trials=NUM_TRIALS,
#     fig_save_path=f"{base_path}model_param_distribution.png",
#     csv_path=f"{base_path}model_param_distribution.csv",
#     logs_dir=f"{base_path}logs/",
#     corr_csv_path=f"{base_path}model_param_distribution_corr.csv",
#     # plot_model_dir=f"{base_path}plots/",
#     figsize=(18, 6),
# )

# %%
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
        "s008_loss",
        "s009_loss",
        "best_train_accuracy",
        "best_val_accuracy",
        "s008_accuracy",
        "s009_accuracy",
    ],
    variance_threshold=1e-10,
    prune_threshold=50,
    patience=100,
)


