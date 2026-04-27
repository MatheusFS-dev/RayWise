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
from _load_dataset import convert_to_sparse_labels

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


def _load_baseline_split_arrays(
    *,
    split_name: str,
    beam_path: str,
    coord_path: str,
    lidar_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one baseline split and keep only samples with non-null beam tensors.

    A sample is considered null when every beam score in its ``(8, 32)`` tensor
    is exactly zero. This helper removes those rows before sparse-label
    conversion so training/evaluation uses only meaningful beam targets.

    Args:
        split_name (str): Human-readable split identifier used in log messages,
            such as ``"s008_train"`` or ``"s009"``.
        beam_path (str): Path to the ``.npz`` file containing
            ``output_classification`` beam tensors.
        coord_path (str): Path to the ``.npz`` file containing ``coordinates``.
        lidar_path (str): Path to the ``.npz`` file containing LiDAR ``input``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered
        ``(lidar, coordinates, sparse_labels)`` arrays for the split.

    Raises:
        FileNotFoundError: If one of the split files is missing.
        KeyError: If expected keys are not found in the ``.npz`` archives.
        ValueError: If arrays have inconsistent lengths or no non-null samples
            remain after filtering.
    """

    # Load raw arrays from baseline files and cast to the same dtypes used in training.
    beam_scores = np.load(beam_path)["output_classification"].astype(np.float32)
    coordinates = np.load(coord_path)["coordinates"].astype(np.float32)
    lidar_tensor = np.load(lidar_path)["input"].astype(np.float32)

    # Ensure sample counts are aligned before applying any filtering.
    if not (beam_scores.shape[0] == coordinates.shape[0] == lidar_tensor.shape[0]):
        raise ValueError(
            f"{split_name} arrays are misaligned: "
            f"beams={beam_scores.shape[0]}, coords={coordinates.shape[0]}, lidar={lidar_tensor.shape[0]}"
        )

    # Keep only rows with at least one non-zero beam score (non-null beams).
    non_null_mask = np.any(beam_scores != 0.0, axis=(1, 2))
    kept_samples = int(np.count_nonzero(non_null_mask))
    removed_samples = int(non_null_mask.size - kept_samples)
    print(f"{split_name}: removed {removed_samples} null-beam samples; kept {kept_samples} samples.")

    if kept_samples == 0:
        raise ValueError(f"{split_name} has no non-null beam samples after filtering.")

    filtered_beams = beam_scores[non_null_mask]
    filtered_coords = coordinates[non_null_mask]
    filtered_lidar = lidar_tensor[non_null_mask]

    # Convert filtered beam tensors to sparse labels expected by the classifier.
    sparse_labels = convert_to_sparse_labels(filtered_beams)
    return filtered_lidar, filtered_coords, sparse_labels


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
    int,
]:
    """Load baseline splits and keep only beam classes with real examples.

    Args:
        s008_path (str): Path to the baseline S008 dataset folder.
        s009_path (str): Path to the baseline S009 dataset folder.
        data_seed (int): Random seed used for the train/validation split to
            ensure deterministic shuffling and reproducibility. When this value
            changes, the S008 train/validation split changes accordingly, while
            null-beam filtering logic remains unchanged.

    Returns:
        tuple: A tuple containing:
            - Training split ``(x_lidar_train, x_coord_train, y_train)``.
            - Validation split ``(x_lidar_val, x_coord_val, y_val)``.
            - The fitted ``StandardScaler`` for coordinates.
                        - Full S009 baseline data ``(x_lidar_s009, x_coord_s009, y_s009)``
                            after scaling and label remapping.
                        - Number of non-null beam classes used by the model output layer.

    Raises:
        ValueError: If the loaded arrays do not align in length, indicating a
        corrupted dataset or mismatched inputs.
    """

    # Load baseline S008 train and validation splits independently, and remove
    # null-beam samples in each split before combining them for the random split.
    s008_train_lidar, s008_train_coord, s008_train_y = _load_baseline_split_arrays(
        split_name="s008_train",
        beam_path=os.path.join(s008_path, "beam_output", "beam_output_train.npz"),
        coord_path=os.path.join(s008_path, "coord_input", "coord_train.npz"),
        lidar_path=os.path.join(s008_path, "lidar_input", "lidar_train.npz"),
    )
    s008_val_lidar, s008_val_coord, s008_val_y = _load_baseline_split_arrays(
        split_name="s008_val",
        beam_path=os.path.join(s008_path, "beam_output", "beam_output_val.npz"),
        coord_path=os.path.join(s008_path, "coord_input", "coord_val.npz"),
        lidar_path=os.path.join(s008_path, "lidar_input", "lidar_val.npz"),
    )

    # Keep S008 and S009 separated; only merge S008 train+val to reproduce the
    # previous local train/validation split behavior.
    s008_lidar_input = np.concatenate((s008_train_lidar, s008_val_lidar), axis=0)
    s008_coord_input = np.concatenate((s008_train_coord, s008_val_coord), axis=0)
    s008_y_train = np.concatenate((s008_train_y, s008_val_y), axis=0)

    # Load S009 independently and filter null-beam samples without mixing with S008.
    s009_lidar_input, s009_coord_input, s009_y = _load_baseline_split_arrays(
        split_name="s009",
        beam_path=os.path.join(s009_path, "beam_output", "beam_output.npz"),
        coord_path=os.path.join(s009_path, "coord_input", "coord_input.npz"),
        lidar_path=os.path.join(s009_path, "lidar_input", "lidar_input.npz"),
    )

    print("================ Beam-class filtering stage ================")
    print(
        "Before remap -> "
        f"S008 samples: {s008_y_train.shape[0]}, unique beams: {np.unique(s008_y_train).size} | "
        f"S009 samples: {s009_y.shape[0]}, unique beams: {np.unique(s009_y).size}"
    )

    # Build a compact beam-class space using only S008 beams that have examples.
    # This removes null/empty beam IDs from the classifier target space.
    present_beams_s008 = np.unique(s008_y_train.astype(np.int64))
    if present_beams_s008.size == 0:
        raise ValueError("No valid S008 beam classes were found after preprocessing.")

    print(
        "S008 non-null beam ids with examples: "
        f"{present_beams_s008.size} classes (min={int(present_beams_s008.min())}, "
        f"max={int(present_beams_s008.max())})"
    )

    label_map = np.full(int(present_beams_s008.max()) + 1, -1, dtype=np.int64)
    label_map[present_beams_s008] = np.arange(present_beams_s008.size, dtype=np.int64)

    # Remap S008 labels to contiguous ids [0, num_non_null_beams-1].
    s008_y_train = label_map[s008_y_train.astype(np.int64)]
    print(
        "S008 remap complete -> "
        f"new label range: [0, {int(s008_y_train.max())}], "
        f"unique beams: {np.unique(s008_y_train).size}"
    )

    # Keep only S009 samples whose beam labels exist in the S008 non-null beam set.
    valid_s009_mask = s009_y.astype(np.int64) < label_map.shape[0]
    valid_s009_mask &= label_map[s009_y.astype(np.int64)] >= 0
    s009_kept = int(np.count_nonzero(valid_s009_mask))
    s009_removed = int(valid_s009_mask.size - np.count_nonzero(valid_s009_mask))
    if s009_removed > 0:
        print(
            "s009: removed {removed} samples with beams unseen in S008 non-null set.".format(
                removed=s009_removed,
            )
        )
    print(f"s009: kept {s009_kept} samples after filtering by S008 non-null beam set.")

    s009_lidar_input = s009_lidar_input[valid_s009_mask]
    s009_coord_input = s009_coord_input[valid_s009_mask]
    s009_y = label_map[s009_y[valid_s009_mask].astype(np.int64)]

    num_non_null_beams = int(present_beams_s008.size)
    print(
        "Using {num_classes} non-null beam classes (from S008 examples).".format(
            num_classes=num_non_null_beams,
        )
    )
    print(
        "After remap/filter -> "
        f"S008 unique beams: {np.unique(s008_y_train).size}, "
        f"S009 unique beams: {np.unique(s009_y).size}"
    )
    print("============================================================")

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
        num_non_null_beams,
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
def build_model(num_classes: int, show_summary: bool = True) -> Model:
    """Build the fixed CNN1D model that matches nas_cnn1d_v7.0.

    Args:
        num_classes (int): Number of output beam classes after removing null
            classes. Must be positive. A lower value reduces output parameters
            and limits predictions to beams observed in S008. A higher value
            increases output size and only makes sense if those classes are
            present in the remapped labels.
        show_summary (bool): If ``True``, prints the Keras model summary to
            stdout, which can add overhead during repeated runs. If ``False``,
            skips printing to keep training logs minimal.

    Returns:
        Model: Compiled Keras model that mirrors the nas_cnn1d_v7.0 top-1 trial
        architecture and hyperparameters. The model reports top-k accuracy based
        on the global ``TOP_K`` value.

    Raises:
        ValueError: If ``num_classes`` is not a positive integer.
    """

    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

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
        num_classes,
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
        num_non_null_beams,
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

    model = build_model(num_classes=num_non_null_beams, show_summary=True)

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
    train_topk_key = f"top_{TOP_K}_accuracy"
    val_topk_key = f"val_top_{TOP_K}_accuracy"
    history_data = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_topk_accuracy": history.history.get(train_topk_key, [np.nan] * len(history.history["loss"])),
        "val_topk_accuracy": history.history.get(val_topk_key, [np.nan] * len(history.history["loss"])),
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
            f"final_train_top_{TOP_K}_accuracy": float(history.history.get(train_topk_key, [np.nan])[-1]),
            f"final_val_top_{TOP_K}_accuracy": float(history.history.get(val_topk_key, [np.nan])[-1]),
            "s009_loss": s009_loss,
            f"s009_top_{TOP_K}_accuracy": s009_topk,
            "num_non_null_beams": num_non_null_beams,
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
