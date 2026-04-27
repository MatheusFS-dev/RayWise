"""
NAS Optuna

- Authored by Matheus Ferreira Silva
- GitHub https://github.com/MatheusFS-dev
"""

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
NUM_TRIALS = 10
EPOCHS = 50

SAMPLER_SEED = 0

# steps_per_execution can break validation when a loop has too few batches,
# leading to missing val_* logs or empty evaluate() returns.
# Ensure steps_per_execution <= min_batches across train, val, and eval loops.
# num_batches = ceil(num_samples / batch_size)
# Example: 1473 samples, batch_size=64 -> ceil(1473/64)=24 batches,
# so steps_per_execution must be <= 24.
STEPS_PER_EXECUTION = 32

# Enable or disable XLA compilation
# Note: some layers don't support determinism with XLA
USE_JIT_COMPILE = True

# Number of top trials to save
TOP_K = 3

# Order to rank trials by:
# "ascending" -> the lowest value is the best
# "descending" -> the highest value is the best
ORDER = "descending"

# Key to rank trials by:
# "value" -> objective trial value
# other e.g., "test_accuracy" -> user params
RANK_KEY = "s009_f1"

# Direction of optimization:
# "minimize" -> the lowest value is the best
# "maximize" -> the highest value is the best
DIRECTION = "minimize"

POLICY = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# Set to an existing dir to resume training
RUN_DIR = f"runs/{get_caller_stem()}"  # (e.g. "runs/nas_1")

# %%—————————————————————————————————————————————————————————————————————————— #
#                        Data Loading and Preprocessing                        #
# ———————————————————————————————————————————————————————————————————————————— #
# Binary labels: LOS=0, NLOS=1
# We dont need the original labels for this task, just the condition
LABEL_LOS = 0
LABEL_NLOS = 1


def load_data(path: Path, seed: int):
    DATA_ROOT = path

    S008_COORD_CSV = DATA_ROOT / "Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv"
    S008_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s008/processed_raw_data/lidar_data_s008"
    S008_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s008/baseline_data/beam_output/beams_output_s008.npz"

    S009_COORD_CSV = DATA_ROOT / "Raymobtime_s009/raw_data/CoordVehiclesRxPerScene_s009.csv"
    S009_LIDAR_FOLDER = DATA_ROOT / "Raymobtime_s009/processed_raw_data/lidar_data_s009"
    S009_BEAM_OUTPUT = DATA_ROOT / "Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz"

    (
        (x_lidar_s008_los, x_coord_s008_los, _),
        (x_lidar_s009_los, x_coord_s009_los, _),
        (x_lidar_s008_nlos, x_coord_s008_nlos, _),
        (x_lidar_s009_nlos, x_coord_s009_nlos, _),
    ) = load_dataset_raw_sparse_labels_by_condition(
        s008_coord_csv=str(S008_COORD_CSV),
        s008_lidar_folder=str(S008_LIDAR_FOLDER),
        s008_beam_output_path=str(S008_BEAM_OUTPUT),
        s009_coord_csv=str(S009_COORD_CSV),
        s009_lidar_folder=str(S009_LIDAR_FOLDER),
        s009_beam_output_path=str(S009_BEAM_OUTPUT),
        data_seed=seed,
        report_label_coverage=True,
    )

    s008_lidar_input = np.concatenate([x_lidar_s008_los, x_lidar_s008_nlos], axis=0)
    s008_coord_input = np.concatenate([x_coord_s008_los, x_coord_s008_nlos], axis=0)
    s008_condition_labels = np.concatenate(
        [
            np.full(shape=(x_coord_s008_los.shape[0],), fill_value=LABEL_LOS, dtype=np.int32),
            np.full(shape=(x_coord_s008_nlos.shape[0],), fill_value=LABEL_NLOS, dtype=np.int32),
        ],
        axis=0,
    )

    s009_lidar_input = np.concatenate([x_lidar_s009_los, x_lidar_s009_nlos], axis=0)
    s009_coord_input = np.concatenate([x_coord_s009_los, x_coord_s009_nlos], axis=0)
    s009_condition_labels = np.concatenate(
        [
            np.full(shape=(x_coord_s009_los.shape[0],), fill_value=LABEL_LOS, dtype=np.int32),
            np.full(shape=(x_coord_s009_nlos.shape[0],), fill_value=LABEL_NLOS, dtype=np.int32),
        ],
        axis=0,
    )

    print("s008 ->", s008_lidar_input.shape, s008_coord_input.shape, s008_condition_labels.shape)
    print("s009 ->", s009_lidar_input.shape, s009_coord_input.shape, s009_condition_labels.shape)

    return (
        (s008_lidar_input, s008_coord_input, s008_condition_labels),
        (s009_lidar_input, s009_coord_input, s009_condition_labels),
    )


# Load the data
(s008_lidar_input, s008_coord_input, s008_condition_labels), (
    s009_lidar_input,
    s009_coord_input,
    s009_condition_labels,
) = load_data(
    path=Path("/media/matheus/SSD-2/matheus/datasets/RayWise"),
    seed=0,
)
print("\n\n\n\n\n")


# %%—————————————————————————————————————————————————————————————————————————— #
#                                Hyperparameters                               #
# ———————————————————————————————————————————————————————————————————————————— #
kparams = KParams.default()

# Or

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
        "sgd": tf.keras.optimizers.SGD(momentum=0.9),
        "adam": tf.keras.optimizers.Adam(),
        "adamw": tf.keras.optimizers.AdamW(weight_decay=1e-4),
        "lion": tf.keras.optimizers.Lion(beta_1=0.9, beta_2=0.99),
        "rmsprop": tf.keras.optimizers.RMSprop(),
    },
    scaler_choices={
        # "standard": StandardScaler,
        # "minmax_0_1": lambda: MinMaxScaler(feature_range=(0, 1)),
        # "minmax_-1_1": lambda: MinMaxScaler(feature_range=(-1, 1)),
    },
    learning_rate=(1e-4, 1e-2),
)


# %%—————————————————————————————————————————————————————————————————————————— #
#                               Model Definition                               #
# ———————————————————————————————————————————————————————————————————————————— #
@tf.keras.utils.register_keras_serializable(package="RayWise")
class LidarChannelOneHot(layers.Layer):
    def __init__(self, offset: int = 2, depth: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.offset = offset
        self.depth = depth

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        indices = tf.cast(inputs + self.offset, tf.int32)
        return tf.one_hot(indices, depth=self.depth, dtype=tf.float32)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape(tuple(input_shape) + (self.depth,))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"offset": self.offset, "depth": self.depth})
        return config


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

    # Per-channel one-hot encoding: map values {-2,-1,0,1} -> depth-4, keep 10 slices
    lidar_one_hot = LidarChannelOneHot(name="lidar_channel_one_hot")(x_lidar_input)
    # -> (batch, 20, 200, 10, 4)

    # Collapse per-slice one-hot into channels for 2D convolutions
    lidar_channels = layers.Reshape((20, 200, 40), name="lidar_channels")(lidar_one_hot)
    # -> (batch, 20, 200, 40)

    # —————————————————————————————————— 2D CNN —————————————————————————————————— #
    x = lidar_channels

    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
    for i in range(num_conv_layers):
        x = build_cnn2d(
            trial=trial,
            kparams=kparams,
            x=x,
            filters_range=trial.suggest_categorical(f"conv2d_{i}_filters", [32, 64, 96, 128, 256]),
            kernel_size_range=(3, 7),
            kernel_size_step=1,
            name_prefix=f"conv2d_{i}",
            kernel_initializer=initializer,
        )

        #! pool size = 1 means no downsampling
        pool_size_width = trial.suggest_int(f"pool2d_width_{i}", 1, 4)
        pool_size_height = trial.suggest_int(f"pool2d_height_{i}", 1, 4)

        pool_size = (pool_size_height, pool_size_width)
        x = layers.MaxPooling2D(pool_size=pool_size, name=f"max_pool2d_{i}")(x)

    lidar_features = layers.GlobalAveragePooling2D(name="lidar_global_avg_pool")(x)

    # ———————————————————————————————— GPS Input ———————————————————————————————— #
    x_coord_input = layers.Input(shape=(2,), name="coord_input")

    coord = x_coord_input
    coord_layers = trial.suggest_int("coord_num_layers", 1, 2)
    for i in range(coord_layers):
        coord = build_dnn(
            trial=trial,
            kparams=kparams,
            x=coord,
            name_prefix=f"coord_dense_{i}",
            units_range=(16, 128),
            units_step=16,
            dropout_rate_range=(0.0, 0.5),
            dropout_rate_step=0.1,
            kernel_initializer=initializer,
        )

    # ———————————————————————————————— Fusion ———————————————————————————————— #
    fused = layers.Concatenate(name="fusion_concat")([lidar_features, coord])

    fusion_layers = trial.suggest_int("fusion_layers", 1, 2)
    x = fused
    for i in range(fusion_layers):
        x = build_dnn(
            trial=trial,
            kparams=kparams,
            x=x,
            name_prefix=f"fusion_dense_{i}",
            units_range=(64, 256),
            units_step=32,
            dropout_rate_range=(0.0, 0.5),
            dropout_rate_step=0.1,
            kernel_initializer=initializer,
        )

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(
        2,
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
    DATA_SEED = 0
    TRAIN_SEED = 0

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

        global s008_lidar_input, s008_coord_input, s008_condition_labels
        global s009_lidar_input, s009_coord_input, s009_condition_labels

        (
            x_lidar_train,
            x_lidar_val,
            x_coord_train,
            x_coord_val,
            y_train,
            y_val,
        ) = train_test_split(
            s008_lidar_input,
            s008_coord_input,
            s008_condition_labels,
            test_size=0.2,
            random_state=DATA_SEED,
            shuffle=True,
            stratify=s008_condition_labels,
        )

        coord_scaler = StandardScaler()

        coord_scaler.fit(x_coord_train)
        x_coord_train = coord_scaler.transform(x_coord_train)
        x_coord_val = coord_scaler.transform(x_coord_val)
        s009_coord_input_scaled = coord_scaler.transform(s009_coord_input)
        # s008_coord_input_scaled = coord_scaler.transform(s008_coord_input)

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
        batch_size = 64

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

        # Check steps_per_execution
        validate_steps_per_execution(
            name=[
                "X_lidar_train",
                "X_coord_train",
                "X_lidar_val",
                "X_coord_val",
                "s009_lidar_input",
                "s009_coord_input",
            ],
            data_size=[
                x_lidar_train.shape[0],
                x_coord_train.shape[0],
                x_lidar_val.shape[0],
                x_coord_val.shape[0],
                s009_lidar_input.shape[0],
                s009_coord_input.shape[0],
            ],
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
                early_stopping_patience=5,
                reduce_lr_patience=2,
                pruning_interval=4,
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

        # ———————————————————————————————————————————————————————————————————————————— #
        #                               Extra Attributes                               #
        # ———————————————————————————————————————————————————————————————————————————— #
        if "minimize" in DIRECTION:
            # The best epoch is the one with the lowest validation loss
            best_idx = np.argmin(history.history["val_loss"])
        else:
            # The best epoch is the one with the highest validation loss
            best_idx = np.argmax(history.history["val_loss"])

        best_train_loss = history.history["loss"][best_idx]
        best_val_loss = history.history["val_loss"][best_idx]

        best_train_accuracy = history.history["accuracy"][best_idx]
        best_val_accuracy = history.history["val_accuracy"][best_idx]

        trial.set_user_attr("best_epoch", int(best_idx) + 1)
        trial.set_user_attr("best_train_loss", float(best_train_loss))
        trial.set_user_attr("best_val_loss", float(best_val_loss))
        trial.set_user_attr("best_train_accuracy", float(best_train_accuracy))
        trial.set_user_attr("best_val_accuracy", float(best_val_accuracy))

        # Evaluate on s009 test set
        s009_loss, s009_accuracy = model.evaluate(
            x=[s009_lidar_input, s009_coord_input_scaled],
            y=s009_condition_labels,
            batch_size=batch_size,
            verbose=2,
        )
        trial.set_user_attr("s009_loss", float(s009_loss))
        trial.set_user_attr("s009_accuracy", float(s009_accuracy))

        # ————————————————————————— Create confusion matrices ———————————————————————— #
        def plot_confusion_matrix(y_true, y_pred, title, save_path):
            cm = confusion_matrix(y_true, y_pred, labels=[LABEL_LOS, LABEL_NLOS])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LOS", "NLOS"])
            disp.plot(cmap="Blues")
            plt.title(title)
            plt.savefig(save_path)
            plt.close()

        # Confusion matrix for training set
        y_pred_train = np.argmax(
            model.predict(
                x=[x_lidar_train, x_coord_train],
                batch_size=batch_size,
                verbose=0,
            ),
            axis=1,
        )
        plot_confusion_matrix(
            y_true=y_train,
            y_pred=y_pred_train,
            title=f"Trial {trial.number} - Training Set Confusion Matrix",
            save_path=os.path.join(fig_dir, f"trial_{trial.number}_confusion_matrix_train.png"),
        )

        # Confusion matrix for validation set
        y_pred_val = np.argmax(
            model.predict(
                x=[x_lidar_val, x_coord_val],
                batch_size=batch_size,
                verbose=0,
            ),
            axis=1,
        )
        plot_confusion_matrix(
            y_true=y_val,
            y_pred=y_pred_val,
            title=f"Trial {trial.number} - Validation Set Confusion Matrix",
            save_path=os.path.join(fig_dir, f"trial_{trial.number}_confusion_matrix_val.png"),
        )

        # Confusion matrix for s009 test set
        y_pred_s009 = np.argmax(
            model.predict(
                x=[s009_lidar_input, s009_coord_input_scaled],
                batch_size=batch_size,
                verbose=0,
            ),
            axis=1,
        )
        plot_confusion_matrix(
            y_true=s009_condition_labels,
            y_pred=y_pred_s009,
            title=f"Trial {trial.number} - s009 Test Set Confusion Matrix",
            save_path=os.path.join(fig_dir, f"trial_{trial.number}_confusion_matrix_s009.png"),
        )

        # Also save the confusion matrices as user attributes with label names
        label_ids = [LABEL_LOS, LABEL_NLOS]
        label_names = ["LOS", "NLOS"]

        def compute_prf_metrics(y_true, y_pred):
            precision = precision_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=LABEL_NLOS,
                zero_division=0,
            )
            recall = recall_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=LABEL_NLOS,
                zero_division=0,
            )
            f1 = f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=LABEL_NLOS,
                zero_division=0,
            )
            return float(precision), float(recall), float(f1)

        def format_confusion_matrix(cm, labels):
            # Pretty-print a square matrix with headers
            label_width = max(len(lbl) for lbl in labels)
            cell_width = max(len(str(val)) for row in cm for val in row)
            col_width = max(label_width, cell_width)

            header = " " * (label_width + 1) + " ".join(f"{lbl:>{col_width}}" for lbl in labels)
            rows = [
                f"{lbl:>{label_width}} " + " ".join(f"{val:>{col_width}}" for val in row)
                for lbl, row in zip(labels, cm)
            ]
            return "\n".join([header, *rows])

        cm_train = confusion_matrix(y_train, y_pred_train, labels=label_ids)
        cm_val = confusion_matrix(y_val, y_pred_val, labels=label_ids)
        cm_s009 = confusion_matrix(s009_condition_labels, y_pred_s009, labels=label_ids)

        trial.set_user_attr("confusion_matrix_train", f"\n{format_confusion_matrix(cm_train, label_names)}")
        trial.set_user_attr("confusion_matrix_val", f"\n{format_confusion_matrix(cm_val, label_names)}")
        trial.set_user_attr("confusion_matrix_s009", f"\n{format_confusion_matrix(cm_s009, label_names)}")

        train_precision, train_recall, train_f1 = compute_prf_metrics(y_train, y_pred_train)
        val_precision, val_recall, val_f1 = compute_prf_metrics(y_val, y_pred_val)
        s009_precision, s009_recall, s009_f1 = compute_prf_metrics(
            s009_condition_labels,
            y_pred_s009,
        )

        trial.set_user_attr("train_precision", train_precision)
        trial.set_user_attr("train_recall", train_recall)
        trial.set_user_attr("train_f1", train_f1)
        trial.set_user_attr("val_precision", val_precision)
        trial.set_user_attr("val_recall", val_recall)
        trial.set_user_attr("val_f1", val_f1)
        trial.set_user_attr("s009_precision", s009_precision)
        trial.set_user_attr("s009_recall", s009_recall)
        trial.set_user_attr("s009_f1", s009_f1)

        # ——————————————————————————————— Save history ——————————————————————————————— #
        history_path = os.path.join(history_dir, f"trial_{trial.number}.csv")
        history_data = {
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "train_accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
        }

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
            force_crash_oom=5,  # Crash after X occurrences of OOM
        )


# %%—————————————————————————————————————————————————————————————————————————— #
#                                 Search space                                 #
# ———————————————————————————————————————————————————————————————————————————— #
# base_path = f"{RUN_DIR}/search_space/"
# plot_model_param_distribution(
#     lambda trial: build_model(
#         trial=trial,
#         kparams=kparams,
#         show_summary=False,
#         train_seed=0,
#     ),
#     benchmark_training=True,
#     fit_x=(s008_lidar_input, s008_coord_input),
#     fit_y=s008_condition_labels,
#     fit_validation_data=((s009_lidar_input, s009_coord_input), s009_condition_labels),
#     bytes_per_param=tf.dtypes.as_dtype(POLICY.variable_dtype).size,
#     # ? Use 1 for per-sample benchmarking in terms of FLOPs
#     # ? Or use equal to prune_model_by_config for correct pruning
#     batch_size=64,
#     n_trials=NUM_TRIALS,
#     fig_save_path=f"{base_path}model_param_distribution.png",
#     csv_path=f"{base_path}model_param_distribution.csv",
#     logs_dir=f"{base_path}logs/",
#     corr_csv_path=f"{base_path}model_param_distribution_corr.csv",
#     # plot_model_dir=f"{base_path}plots/",
#     figsize=(18, 6),
# )

# %%—————————————————————————————————————————————————————————————————————————— #
#                                     Study                                    #
# ———————————————————————————————————————————————————————————————————————————— #
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
        "best_train_accuracy",
        "best_val_accuracy",
        "train_precision",
        "train_recall",
        "train_f1",
        "val_precision",
        "val_recall",
        "val_f1",
        "s009_loss",
        "s009_accuracy",
        "s009_precision",
        "s009_recall",
        "s009_f1",
        "confusion_matrix_train",
        "confusion_matrix_val",
        "confusion_matrix_s009",
    ],
    variance_threshold=None,  # 1e-10
    prune_threshold=None,  # 50
    patience=None,  # 100
)
