# %% [markdown]
# # NAS - Optuna
#
# - **Authored by:** Matheus Ferreira Silva
# - **GitHub:**: https://github.com/MatheusFS-dev

# %% [markdown]
# ## 1. Setup and Configuration

# %% [markdown]
# ### 1.1. Environment Variables

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

# %% [markdown]
# ### 1.3. GPU Management

# %%
get_gpu_info()

# %% [markdown]
# ## 2. Run Parameters

# %%
NUM_TRIALS = 2000
EPOCHS = 50

SAMPLER_SEED = 0

STEPS_PER_EXECUTION = 32

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = True

# %%
# Number of top trials to save
TOP_K = 3

# Order to rank trials by:
# "ascending" -> the lowest value is the best
# "descending" -> the highest value is the best
ORDER = "ascending"

# Key to rank trials by:
# "value" -> objective trial value
# other e.g., "test_accuracy" -> user params
RANK_KEY = "value"

# Direction of optimization:
# "minimize" -> the lowest value is the best
# "maximize" -> the highest value is the best
DIRECTION = "minimize"

# %%
POLICY = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# %%
# Set to an existing dir to resume training
RUN_DIR = f"runs/{get_caller_stem()}"  # (e.g. "runs/nas_1")

# %% [markdown]
# ## 3. Data Loading and Preprocessing

# %%
(
    s008_coord_input,
    s008_lidar_input,
    s008_y_train,
    s009_coord_input,
    s009_lidar_input,
    s009_y,
) = load_dataset_sparse_labels(s008_path="./data/s008", s009_path="./data/s009")

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
    """Build a grid-preserving 2D CNN NAS model for beam classification.

    Args:
        trial (optuna.Trial): Optuna trial used to sample the 2D CNN search
            space. The sampled architecture preserves the LiDAR grid as
            ``(20, 200)`` and varies block count, filter count, kernel shape,
            normalization, activation, pooling, global pooling, dense width,
            and dropout.
        kparams (dict): Hyperparameter helper used by the existing NAS scripts
            to construct the optimizer. This function only uses it for
            ``kparams.get_optimizer(trial)`` so optimizer choices and learning
            rate sampling stay aligned with the other architecture families.
        show_summary (bool): If ``True``, prints the Keras model summary before
            compilation, which adds console output but does not change the
            model. If ``False``, skips summary printing for quieter NAS trials.
        **kwargs (Any): Additional build options. ``train_seed`` may be
            provided to seed the Glorot initializer; if omitted, TensorFlow uses
            the initializer's default randomness.

    Returns:
        tf.keras.Model: Compiled two-input Keras model. The inputs are raw LiDAR
        tensors shaped ``(20, 200, 10)`` and GPS coordinates shaped ``(2,)``;
        the output is a 256-class softmax.

    Raises:
        ValueError: Propagated from Keras if a sampled architecture creates an
            invalid tensor shape, such as over-pooling a spatial dimension.
    """

    train_seed = kwargs.get("train_seed")

    # ———————————————————————————————————————————————————————————————————————————— #
    #                              Model Construction                              #
    # ———————————————————————————————————————————————————————————————————————————— #
    initializer = tf.keras.initializers.GlorotUniform(
        seed=train_seed,
    )

    # ———————————————————————————————— LiDAR Input ——————————————————————————————— #
    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")

    # Keep the same semantic LiDAR representation used by the 1D NAS so this
    # experiment tests grid preservation, not a different input encoding.
    one_hot_lidar = layers.Lambda(
        lambda x: tf.concat(
            [
                # “Is there a BS anywhere in the 10 channels?” → 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, -2), axis=-1, keepdims=True), tf.float32),
                # “Vehicle?” → 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, -1), axis=-1, keepdims=True), tf.float32),
                # “Obstacle?” → 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, 1), axis=-1, keepdims=True), tf.float32),
                # “Free?” → 1 channel (all channels zero)
                tf.cast(tf.reduce_all(tf.equal(x, 0), axis=-1, keepdims=True), tf.float32),
            ],
            axis=-1,
        ),
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=(20, 200, 4),
        name="lidar_transform_to_one_hot",
    )(x_lidar_input)
    # -> (batch, 20, 200, 4)

    # ———————————————————————————————— GPS Input ———————————————————————————————— #
    # Input for coordinate data (e.g., shape: (2,))
    x_coord_input = layers.Input(shape=(2,), name="coord_input")

    # Broadcast GPS over the spatial grid so the 2D CNN sees the same two GPS
    # channels at every LiDAR cell without destroying the original layout.
    x_coord: layers.Layer = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, axis=1), axis=1), [1, 20, 200, 1]),
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=(20, 200, 2),
        name="coord_broadcast_grid",
    )(x_coord_input)

    # ————————————————————————————— Combine Branches ————————————————————————————— #
    # Fuse channels:  (batch,20,200,4) + (batch,20,200,2) → (batch,20,200,6)
    x = layers.Concatenate(axis=-1, name="combine_lidar_coord_grid")([one_hot_lidar, x_coord])

    # ————————————————————————————————— 2D CNN NAS ——————————————————————————————— #
    kernel_choices = {
        "3x3": (3, 3),
        "3x5": (3, 5),
        "5x5": (5, 5),
        "1x5": (1, 5),
        "1x7": (1, 7),
        "5x1": (5, 1),
    }
    pool_size_choices = {
        "2x2": (2, 2),
        "1x2": (1, 2),
        "2x4": (2, 4),
    }
    activation_choices = {
        "relu": tf.keras.activations.relu,
        "gelu": tf.keras.activations.gelu,
        "swish": tf.keras.activations.swish,
    }

    num_conv_blocks = trial.suggest_int("num_conv_blocks", 1, 4)
    for block_idx in range(num_conv_blocks):
        filters = trial.suggest_categorical(f"conv2d_block_{block_idx}_filters", [16, 32, 64, 128, 256])
        kernel_size = kernel_choices[
            trial.suggest_categorical(f"conv2d_block_{block_idx}_kernel_size", list(kernel_choices.keys()))
        ]
        conv_layers_per_block = trial.suggest_int(f"conv2d_block_{block_idx}_layers", 1, 2)
        normalization = trial.suggest_categorical(f"conv2d_block_{block_idx}_normalization", ["none", "batch_norm"])
        activation = activation_choices[
            trial.suggest_categorical(f"conv2d_block_{block_idx}_activation", list(activation_choices.keys()))
        ]

        for conv_idx in range(conv_layers_per_block):
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation=None,
                kernel_initializer=initializer,
                name=f"conv2d_block_{block_idx}_conv_{conv_idx}",
            )(x)
            if normalization == "batch_norm":
                x = layers.BatchNormalization(name=f"conv2d_block_{block_idx}_batch_norm_{conv_idx}")(x)
            x = layers.Activation(activation, name=f"conv2d_block_{block_idx}_activation_{conv_idx}")(x)

        # Asymmetric pooling is part of the search because width 200 is much
        # larger than height 20, and early square pooling can collapse height too
        # quickly for this grid-control baseline.
        pooling_type = trial.suggest_categorical(
            f"conv2d_block_{block_idx}_pooling",
            ["none", "max_pool_2d", "average_pool_2d"],
        )
        if pooling_type != "none":
            pool_size = pool_size_choices[
                trial.suggest_categorical(f"conv2d_block_{block_idx}_pool_size", list(pool_size_choices.keys()))
            ]
            if pooling_type == "max_pool_2d":
                x = layers.MaxPooling2D(pool_size=pool_size, name=f"conv2d_block_{block_idx}_max_pool")(x)
            else:
                x = layers.AveragePooling2D(pool_size=pool_size, name=f"conv2d_block_{block_idx}_avg_pool")(x)

    global_pooling = trial.suggest_categorical("global_pooling", ["average", "max", "concat_avg_max"])
    if global_pooling == "average":
        x = layers.GlobalAveragePooling2D(name="global_avg_pooling_2d")(x)
    elif global_pooling == "max":
        x = layers.GlobalMaxPooling2D(name="global_max_pooling_2d")(x)
    else:
        avg_pool = layers.GlobalAveragePooling2D(name="global_avg_pooling_2d")(x)
        max_pool = layers.GlobalMaxPooling2D(name="global_max_pooling_2d")(x)
        x = layers.Concatenate(axis=-1, name="global_pooling_concat")([avg_pool, max_pool])

    num_dense_layers = trial.suggest_int("num_dense_layers", 0, 1)
    for i in range(num_dense_layers):
        dense_units = trial.suggest_categorical(f"dense_{i}_units", [64, 128, 256, 512])
        dropout_rate = trial.suggest_float(f"dense_{i}_dropout", 0.0, 0.5, step=0.1)
        x = layers.Dense(
            dense_units,
            activation="relu",
            kernel_initializer=initializer,
            name=f"dense_{i}",
        )(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, seed=train_seed, name=f"dense_{i}_dropout")(x)

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(
        256,
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
    """Train one 2D CNN NAS trial and return validation cross-entropy.

    Args:
        trial (optuna.Trial): Current Optuna trial for model, optimizer, and
            learning-rate suggestions. The trial receives validation metrics,
            S008/S009 evaluation metrics, and model statistics as user
            attributes, but the returned objective value is validation loss.
        **kwargs (Any): Study directory paths supplied by ``run_study``. The
            required keys are ``backup``, ``model``, ``fig``, ``tensorboard``,
            ``logs``, ``history``, and ``scaler``. These paths control where
            trial artifacts are written; missing keys raise ``KeyError``.

    Returns:
        float: Best validation sparse categorical cross-entropy selected from
        the epoch with the best validation accuracy. Lower values are better
        because the study direction is ``"minimize"``.

    Raises:
        KeyError: If a required study artifact directory is missing from
            ``kwargs``.
        optuna.TrialPruned: If the architecture has invalid pooling geometry or
            exceeds configured resource limits.
        Exception: Propagates non-pruned runtime failures after logging through
            the existing trial error handler.
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
                # "model_size": 350,  # Maximum model size in MB
                "memory_mb": 9000,  # Maximum memory training usage in MB
                # "param": 1.0e6,  # Maximum number of parameters
                # "flops": 5.0e8,  # Maximum number of FLOPs
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

        # Optimize validation loss only; S009 metrics are stored for reporting
        # after training and are not used to rank trials.
        return best_val_loss
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
            force_crash_oom=2,  # Crash after X occurrences of OOM
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
    variance_threshold=None,
    prune_threshold=None,
    patience=None,
)
