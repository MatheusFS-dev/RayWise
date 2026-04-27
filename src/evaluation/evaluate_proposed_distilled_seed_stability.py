"""Seed-stability evaluation aligned to the original Proposed-Distilled (MOTPE) pipeline."""

import csv
import gc
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


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
OUTPUT_DIR = SRC_ROOT / "evaluation" / "outputs" / "seed_stability_proposed_distilled"
KD_ARGS_PATH = (
    REPO_ROOT
    / "weights"
    / "nas_multiobjective_best_knowledge_distillation"
    / "optuna_study"
    / "args"
    / "top_1_trial.txt"
)
KD_MODEL_PATH = (
    REPO_ROOT
    / "weights"
    / "nas_multiobjective_best_knowledge_distillation"
    / "optuna_study"
    / "model"
    / "top_1_trial_721.keras"
)
TEACHER_MODEL_DIR = (
    REPO_ROOT
    / "weights"
    / "nas_best_architectures_cnn1d"
    / "optuna_study"
    / "model"
)
RANKED_MODEL_RE = re.compile(r"top_(\d+)_trial_\d+\.keras$")

DEFAULT_SEEDS = tuple(range(10))
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64


def log_step(message: str) -> None:
    """Print one progress message for the seed-stability evaluation.

    Args:
        message (str): Human-readable status text describing the current
            evaluation step. The caller should include concrete details such as
            seed, paths, shapes, or metric values when they help diagnose long
            runs.

    Returns:
        None: Writes the message to stdout.

    Raises:
        None: The function only formats and prints a string.
    """

    print(f"[proposed_distilled_seed_stability] {message}", flush=True)


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


def load_data_baseline() -> dict[str, np.ndarray]:
    """Load baseline S008/S009 arrays.

    Args:
        None: Paths are taken from module constants.

    Returns:
        dict[str, np.ndarray]: Raw arrays required by the KD pipeline.

    Raises:
        FileNotFoundError: If required baseline files are missing.
    """

    log_step(f"Loading baseline arrays from {S008_ROOT} and {S009_ROOT}.")
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

    data = {
        "s008_lidar_full": np.concatenate([s008_lidar_train, s008_lidar_val], axis=0).astype(np.float32),
        "s008_coord_full": np.concatenate([s008_coord_train, s008_coord_val], axis=0).astype(np.float32),
        "s008_y_full": np.concatenate([s008_y_train, s008_y_val], axis=0).astype(np.int64),
        "s009_lidar": s009_lidar,
        "s009_coord": s009_coord,
        "s009_y": s009_y,
    }
    log_step(
        "Loaded data: "
        f"S008 lidar={data['s008_lidar_full'].shape}, "
        f"S008 coord={data['s008_coord_full'].shape}, "
        f"S008 labels={data['s008_y_full'].shape}, "
        f"S009 lidar={data['s009_lidar'].shape}, "
        f"S009 coord={data['s009_coord'].shape}, "
        f"S009 labels={data['s009_y'].shape}."
    )
    return data


def parse_trial_hparam_value(raw_value: str) -> Any:
    """Parse one hyperparameter value from the saved Optuna args report.

    Args:
        raw_value (str): Text after the ``key:`` separator in
            ``top_1_trial.txt``. ``"True"`` and ``"False"`` become booleans,
            integer-looking values become ``int``, decimal/scientific values
            become ``float``, and all other non-empty values remain ``str``.

    Returns:
        Any: Parsed scalar value with the narrowest type needed by the saved
        hyperparameter report.

    Raises:
        ValueError: If ``raw_value`` is empty after whitespace trimming.

    Examples:
        ``parse_trial_hparam_value("3.5")`` returns ``3.5``.
    """

    value = raw_value.strip()
    if not value:
        raise ValueError("Hyperparameter value cannot be empty.")
    if value == "True":
        return True
    if value == "False":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_best_kd_hparams() -> dict[str, Any]:
    """Load top-1 KD hyperparameters from the saved args report.

    Args:
        None: Reads ``KD_ARGS_PATH``, the repository artifact generated beside
            the top-1 distilled model. The epoch-history CSV is intentionally
            not used because it contains training metrics, not trial
            hyperparameters.

    Returns:
        dict[str, Any]: Hyperparameters needed for seed-stability retraining:
        teacher count, optimizer name, learning rate, distillation alpha, and
        distillation temperature.

    Raises:
        FileNotFoundError: If ``KD_ARGS_PATH`` is missing.
        ValueError: If the args report has no ``Trial hyperparameters`` section,
            if a required hyperparameter is missing, or if a required value has
            the wrong type.

    Examples:
        ``hparams = load_best_kd_hparams()`` loads the saved top-1 KD trial
        settings used by ``run_one_seed``.
    """

    log_step(f"Loading KD hyperparameters from {KD_ARGS_PATH}.")
    if not KD_ARGS_PATH.is_file():
        raise FileNotFoundError(f"KD args report not found: {KD_ARGS_PATH}.")

    hparams: dict[str, Any] = {}
    in_hparam_section = False
    for line in KD_ARGS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() == "Trial hyperparameters:":
            in_hparam_section = True
            continue
        if not in_hparam_section:
            continue
        if not line.strip():
            continue
        if not line.startswith("  "):
            break

        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"Invalid KD hyperparameter line: {line!r}.")
        hparams[key] = parse_trial_hparam_value(value)

    if not hparams:
        raise ValueError(f"No trial hyperparameters found in {KD_ARGS_PATH}.")

    required = [
        "num_teachers",
        "optimizer",
        "learning_rate",
        "distill_alpha",
        "distill_temperature",
    ]
    missing = [name for name in required if name not in hparams]
    if missing:
        raise ValueError(f"KD args report missing required hyperparameters: {missing}.")

    num_teachers = hparams["num_teachers"]
    if isinstance(num_teachers, bool) or not isinstance(num_teachers, int):
        raise ValueError(f"num_teachers must be an integer, but received {num_teachers!r}.")

    optimizer = hparams["optimizer"]
    if not isinstance(optimizer, str):
        raise ValueError(f"optimizer must be a string, but received {optimizer!r}.")

    loaded_hparams = {
        "num_teachers": num_teachers,
        "optimizer": optimizer,
        "learning_rate": float(hparams["learning_rate"]),
        "distill_alpha": float(hparams["distill_alpha"]),
        "distill_temperature": float(hparams["distill_temperature"]),
    }
    log_step(f"Loaded KD hyperparameters: {loaded_hparams}.")
    return loaded_hparams


def resolve_teacher_paths(num_teachers: int) -> list[Path]:
    """Resolve the ranked teacher checkpoints used by the KD trial.

    Args:
        num_teachers (int): Number of top-ranked teacher checkpoints to use.
            Larger values increase teacher-inference cost linearly because each
            teacher runs one prediction pass over the training split. The value
            must be positive and no larger than the number of ranked checkpoint
            files in ``TEACHER_MODEL_DIR``.

    Returns:
        list[Path]: Existing teacher checkpoint paths ordered by rank.

    Raises:
        FileNotFoundError: If ``TEACHER_MODEL_DIR`` is missing.
        ValueError: If ``num_teachers`` is not positive, if a ranked filename is
            malformed, or if there are not enough teacher checkpoints.

    Examples:
        ``resolve_teacher_paths(2)`` returns the rank-1 and rank-2 teacher
        checkpoint paths.
    """

    log_step(f"Resolving top {num_teachers} teacher checkpoint(s) from {TEACHER_MODEL_DIR}.")
    if num_teachers <= 0:
        raise ValueError(f"num_teachers must be positive, but received {num_teachers}.")
    if not TEACHER_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Teacher model directory not found: {TEACHER_MODEL_DIR}.")

    ranked_paths: list[tuple[int, Path]] = []
    for path in TEACHER_MODEL_DIR.glob("top_*_trial_*.keras"):
        match = RANKED_MODEL_RE.match(path.name)
        if match is None:
            raise ValueError(f"Teacher checkpoint has unexpected ranked filename: {path}.")
        ranked_paths.append((int(match.group(1)), path))

    teacher_paths = [path for _, path in sorted(ranked_paths, key=lambda item: item[0])]
    if len(teacher_paths) < num_teachers:
        raise ValueError(
            f"Requested {num_teachers} teacher checkpoint(s), but only found {len(teacher_paths)} "
            f"in {TEACHER_MODEL_DIR}."
        )
    selected_paths = teacher_paths[:num_teachers]
    log_step(f"Resolved teacher checkpoints: {[path.name for path in selected_paths]}.")
    return selected_paths


def reseed_initializer_config(config: Any, seed: int) -> Any:
    """Recursively overwrite initializer seeds inside a Keras config object.

    Args:
        config (Any): Nested Keras model configuration containing dictionaries,
            lists, and scalar values. Dictionaries with initializer class names
            and a ``seed`` field are copied with that seed replaced.
        seed (int): Seed assigned to initializer config entries. Different
            seeds change model initialization while preserving architecture.

    Returns:
        Any: Copied configuration with seeded initializer entries.

    Raises:
        None: The function only transforms Python containers and scalar values.

    Examples:
        ``reseed_initializer_config(model.get_config(), 7)`` returns a copied
        config whose initializer seed fields are set to ``7``.
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


def build_student(tf_module: Any, custom_objects: dict[str, Any], seed: int) -> Any:
    """Clone the saved top-1 KD student architecture with fresh seeded weights.

    Args:
        tf_module (Any): Imported TensorFlow module returned by
            ``_load_tensorflow_components``.
        custom_objects (dict[str, Any]): Custom Keras objects needed to
            deserialize RayWise layers such as ``LidarToOneHot`` and
            ``CoordTileFlat``.
        seed (int): Initializer seed for the cloned architecture. Changing the
            seed changes initial weights but leaves layer topology and
            hyperparameters from the saved top-1 KD model unchanged.

    Returns:
        Any: Uncompiled Keras student model matching ``KD_MODEL_PATH``.

    Raises:
        FileNotFoundError: If the saved top-1 KD model is missing.
        ValueError: If Keras cannot rebuild the model from its configuration.

    Examples:
        ``student = build_student(tf_module=tf, custom_objects=objects, seed=0)``
        returns a fresh model with the saved KD architecture.
    """

    log_step(f"Building fresh student from {KD_MODEL_PATH.name} with seed={seed}.")
    if not KD_MODEL_PATH.is_file():
        raise FileNotFoundError(f"KD model checkpoint not found: {KD_MODEL_PATH}.")

    template_model = tf_module.keras.models.load_model(
        str(KD_MODEL_PATH),
        safe_mode=False,
        compile=False,
        custom_objects=custom_objects,
    )
    patch_lambda_globals(model=template_model, tf_module=tf_module)

    # The saved checkpoint is the source of truth for architecture; only
    # initializer seeds are changed so each seed run starts from fresh weights.
    seeded_config = reseed_initializer_config(template_model.get_config(), seed)
    student = tf_module.keras.Model.from_config(seeded_config, custom_objects=custom_objects)
    patch_lambda_globals(model=student, tf_module=tf_module)
    log_step(f"Built student model with {student.count_params()} parameters.")
    return student


def build_optimizer(tf_module: Any, optimizer_name: str, learning_rate: float) -> Any:
    """Build the optimizer selected by the saved KD trial.

    Args:
        tf_module (Any): Imported TensorFlow module whose Keras optimizers are
            used. The active TensorFlow version determines optimizer behavior
            and device placement.
        optimizer_name (str): Optimizer mode from the Optuna trial. ``"adam"``
            uses Adam, ``"adamw"`` uses AdamW with decoupled weight decay
            support from Keras defaults, ``"sgd"`` uses SGD with momentum
            ``0.9``, and ``"rmsprop"`` uses RMSprop. Unsupported values raise
            instead of falling back to a different optimizer.
        learning_rate (float): Positive optimizer learning rate.

    Returns:
        Any: Keras optimizer instance.

    Raises:
        ValueError: If ``learning_rate`` is not positive or if
            ``optimizer_name`` is not one of the supported trial values.

    Examples:
        ``optimizer = build_optimizer(tf_module=tf, optimizer_name="adamw", learning_rate=1e-3)``
        builds the AdamW optimizer selected by a KD trial.
    """

    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, but received {learning_rate}.")

    optimizers = tf_module.keras.optimizers
    if optimizer_name == "adam":
        return optimizers.Adam(learning_rate=learning_rate)
    if optimizer_name == "adamw":
        return optimizers.AdamW(learning_rate=learning_rate)
    if optimizer_name == "sgd":
        return optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if optimizer_name == "rmsprop":
        return optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Unsupported KD optimizer: {optimizer_name!r}.")


class Distiller:  # pylint: disable=too-few-public-methods
    """Knowledge distillation trainer mirroring nas_mo_kd_cnn1d behavior.

    Args:
        student (Any): Trainable student model.
        temperature (float): Distillation temperature. Higher values soften
            teacher and student probabilities before KL divergence; lower
            values keep distributions closer to their original sharpness.
        alpha (float): Hard/soft loss mixing coefficient. ``1.0`` makes the
            total loss purely hard-label cross-entropy, while ``0.0`` makes it
            purely teacher-imitation KL loss.

    Returns:
        None: Initializes the wrapper model.

    Raises:
        ValueError: If ``temperature`` is not positive or ``alpha`` is outside
            ``[0.0, 1.0]``.
    """

    def __init__(self, student: Any, temperature: float, alpha: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, but received {temperature}.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], but received {alpha}.")

        self.student = student
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.tf_module, _ = _load_tensorflow_components()
        self.keras = self.tf_module.keras
        self.model = student
        self.optimizer = None
        self.hard_loss = None
        self.metric_top1 = self.keras.metrics.SparseCategoricalAccuracy(name="top_1_accuracy")

    def compile(self, optimizer: Any, loss: Any) -> None:
        """Store optimizer/loss for custom training loop.

        Args:
            optimizer (Any): Optimizer instance.
            loss (Any): Sparse hard-label loss.

        Returns:
            None: Mutates trainer state.
        """

        self.optimizer = optimizer
        self.hard_loss = loss

    def fit(
        self,
        x_train: list[np.ndarray],
        y_train_labels: np.ndarray,
        y_train_soft: np.ndarray,
        x_val: list[np.ndarray],
        y_val_labels: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
    ) -> None:
        """Run distillation training over epochs.

        Args:
            x_train (list[np.ndarray]): Student inputs for training.
            y_train_labels (np.ndarray): Hard labels.
            y_train_soft (np.ndarray): Teacher soft targets.
            x_val (list[np.ndarray]): Validation inputs.
            y_val_labels (np.ndarray): Validation hard labels.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.

        Returns:
            None: Updates student weights.

        Raises:
            ValueError: If the optimizer or hard-label loss was not configured
                before training.
        """

        if self.optimizer is None or self.hard_loss is None:
            raise ValueError("Distiller must be compiled before fit.")

        tf_module = self.tf_module
        train_ds = tf_module.data.Dataset.from_tensor_slices(((x_train[0], x_train[1]), (y_train_labels, y_train_soft)))
        train_ds = train_ds.batch(batch_size)

        for epoch_index in range(epochs):
            log_step(f"Training epoch {epoch_index + 1}/{epochs}.")
            self.metric_top1.reset_state()
            for (lidar_batch, coord_batch), (hard_batch, soft_batch) in train_ds:
                with tf_module.GradientTape() as tape:
                    student_probs = self.student([lidar_batch, coord_batch], training=True)
                    hard = self.hard_loss(hard_batch, student_probs)

                    # The original KD trainer stores teacher outputs as
                    # probabilities, so both distributions are softened in
                    # log-space instead of assuming logits are available.
                    eps = tf_module.keras.backend.epsilon()
                    student_soft = tf_module.nn.softmax(
                        tf_module.math.log(tf_module.clip_by_value(student_probs, eps, 1.0)) / self.temperature,
                        axis=-1,
                    )
                    teacher_soft = tf_module.nn.softmax(
                        tf_module.math.log(tf_module.clip_by_value(soft_batch, eps, 1.0)) / self.temperature,
                        axis=-1,
                    )
                    soft = tf_module.keras.losses.KLDivergence()(teacher_soft, student_soft) * (self.temperature ** 2)
                    loss = self.alpha * hard + (1.0 - self.alpha) * soft

                grads = tape.gradient(loss, self.student.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
                self.metric_top1.update_state(hard_batch, student_probs)

            _ = self.student.predict(x_val, batch_size=batch_size, verbose=2)
            _ = y_val_labels
            log_step(f"Finished epoch {epoch_index + 1}/{epochs}.")


def precompute_teacher_soft_targets(
    teacher_paths: Sequence[Path],
    x_train: list[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Compute ensemble soft targets from fixed teacher checkpoints.

    Args:
        teacher_paths (Sequence[Path]): Paths to teacher models.
        x_train (list[np.ndarray]): Training inputs for teachers.
        batch_size (int): Inference batch size.

    Returns:
        np.ndarray: Averaged soft targets with shape ``(N, C)``.

    Raises:
        FileNotFoundError: If any teacher checkpoint is missing.
    """

    log_step(f"Computing teacher soft targets with {len(teacher_paths)} teacher checkpoint(s).")
    tf_module, custom_objects = _load_tensorflow_components()
    teacher_probs = []
    for teacher_index, path in enumerate(teacher_paths, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}.")
        log_step(f"Loading teacher {teacher_index}/{len(teacher_paths)}: {path.name}.")
        teacher_model = tf_module.keras.models.load_model(
            str(path),
            safe_mode=False,
            compile=False,
            custom_objects=custom_objects,
        )
        patch_lambda_globals(model=teacher_model, tf_module=tf_module)
        log_step(f"Predicting soft targets with teacher {teacher_index}/{len(teacher_paths)}.")
        probs = np.asarray(teacher_model.predict(x_train, batch_size=batch_size, verbose=2), dtype=np.float32)
        teacher_probs.append(probs)
        tf_module.keras.backend.clear_session()
        gc.collect()

    soft_targets = np.mean(np.stack(teacher_probs, axis=0), axis=0).astype(np.float32)
    log_step(f"Computed teacher soft targets with shape {soft_targets.shape}.")
    return soft_targets


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

    log_step(f"Writing CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one_seed(
    data: dict[str, np.ndarray],
    hparams: dict[str, Any],
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run one KD seed using fixed split and teacher soft targets.

    Args:
        data (dict[str, np.ndarray]): Baseline dataset arrays.
        hparams (dict[str, Any]): Top-1 KD hyperparameters.
        seed (int): Run seed and split seed.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        dict[str, Any]: Row containing ``seed``, ``top1``, and ``top10``.
    """

    log_step(f"Starting seed {seed}: epochs={epochs}, batch_size={batch_size}.")
    set_all_random_seeds(seed)
    tf_module, custom_objects = _load_tensorflow_components()

    x_lidar_train, x_lidar_val, x_coord_train, x_coord_val, y_train, y_val = train_test_split(
        data["s008_lidar_full"],
        data["s008_coord_full"],
        data["s008_y_full"],
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    log_step(
        f"Seed {seed}: split S008 into train={x_lidar_train.shape[0]} "
        f"and validation={x_lidar_val.shape[0]} samples."
    )

    scaler_mean = np.mean(x_coord_train, axis=0, dtype=np.float64)
    scaler_std = np.std(x_coord_train, axis=0, dtype=np.float64)
    scaler_std = np.where(scaler_std == 0.0, 1.0, scaler_std)
    log_step(f"Seed {seed}: fitted coordinate scaler mean={scaler_mean.tolist()}, std={scaler_std.tolist()}.")

    x_coord_train = ((x_coord_train - scaler_mean) / scaler_std).astype(np.float32)
    x_coord_val = ((x_coord_val - scaler_mean) / scaler_std).astype(np.float32)
    s009_coord = ((data["s009_coord"] - scaler_mean) / scaler_std).astype(np.float32)

    teacher_paths = resolve_teacher_paths(num_teachers=hparams["num_teachers"])
    teacher_soft = precompute_teacher_soft_targets(
        teacher_paths,
        [x_lidar_train.astype(np.float32), x_coord_train.astype(np.float32)],
        batch_size=batch_size,
    )

    student = build_student(tf_module=tf_module, custom_objects=custom_objects, seed=seed)

    distiller = Distiller(
        student=student,
        temperature=hparams["distill_temperature"],
        alpha=hparams["distill_alpha"],
    )
    distiller.compile(
        optimizer=build_optimizer(
            tf_module=tf_module,
            optimizer_name=hparams["optimizer"],
            learning_rate=hparams["learning_rate"],
        ),
        loss=tf_module.keras.losses.SparseCategoricalCrossentropy(),
    )
    log_step(f"Seed {seed}: starting distillation training.")
    distiller.fit(
        x_train=[x_lidar_train.astype(np.float32), x_coord_train.astype(np.float32)],
        y_train_labels=y_train.astype(np.int64),
        y_train_soft=teacher_soft,
        x_val=[x_lidar_val.astype(np.float32), x_coord_val.astype(np.float32)],
        y_val_labels=y_val.astype(np.int64),
        epochs=epochs,
        batch_size=batch_size,
    )

    log_step(f"Seed {seed}: scoring S009 test set.")
    test_inputs = prepare_model_inputs(model=student, lidar=data["s009_lidar"], coordinates=s009_coord)
    scores = np.asarray(student.predict(test_inputs, batch_size=batch_size, verbose=2), dtype=np.float32)

    top1 = compute_topk_accuracy(data["s009_y"], scores, k=1)
    top10 = compute_topk_accuracy(data["s009_y"], scores, k=10)

    tf_module.keras.backend.clear_session()
    gc.collect()

    log_step(f"Finished seed {seed}: top1={top1:.6f}, top10={top10:.6f}.")
    return {"seed": int(seed), "top1": float(top1), "top10": float(top10)}


def evaluate_proposed_distilled_seed_stability(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float]:
    """Run 10-seed stability evaluation aligned to KD original training.

    Args:
        seeds (Sequence[int]): Distinct run seeds.
        epochs (int): Training epochs per seed.
        batch_size (int): Batch size for fit and predict.

    Returns:
        dict[str, float]: Summary containing top1/top10 mean and std.

    Raises:
        ValueError: If ``seeds`` is empty or has duplicates.
    """

    log_step("Starting Proposed-Distilled seed-stability evaluation.")
    validated_seeds = tuple(int(seed) for seed in seeds)
    if not validated_seeds:
        raise ValueError("seeds must contain at least one seed.")
    if len(set(validated_seeds)) != len(validated_seeds):
        raise ValueError("seeds must be unique.")

    data = load_data_baseline()
    hparams = load_best_kd_hparams()

    rows = []
    for seed_index, seed in enumerate(validated_seeds, start=1):
        log_step(f"Running seed {seed_index}/{len(validated_seeds)}: {seed}.")
        rows.append(run_one_seed(data, hparams, seed, epochs, batch_size))

    top1_values = np.asarray([row["top1"] for row in rows], dtype=np.float64)
    top10_values = np.asarray([row["top10"] for row in rows], dtype=np.float64)
    summary = {
        "top1_mean": float(np.mean(top1_values)),
        "top1_std": float(np.std(top1_values, ddof=0)),
        "top10_mean": float(np.mean(top10_values)),
        "top10_std": float(np.std(top10_values, ddof=0)),
    }

    write_csv(
        OUTPUT_DIR / "proposed_distilled_seed_runs.csv",
        [{"model_name": "Proposed-Distilled (MOTPE)", **row} for row in rows],
        ["model_name", "seed", "top1", "top10"],
    )
    write_csv(
        OUTPUT_DIR / "proposed_distilled_seed_summary.csv",
        [{"model_name": "Proposed-Distilled (MOTPE)", "num_seeds": len(validated_seeds), **summary}],
        ["model_name", "num_seeds", "top1_mean", "top1_std", "top10_mean", "top10_std"],
    )
    log_step(f"Finished evaluation summary: {summary}.")
    return summary


if __name__ == "__main__":
    metrics = evaluate_proposed_distilled_seed_stability()
    print("Proposed-Distilled (MOTPE)")
    print(f"  Top-1  mean={metrics['top1_mean']:.6f} std={metrics['top1_std']:.6f}")
    print(f"  Top-10 mean={metrics['top10_mean']:.6f} std={metrics['top10_std']:.6f}")
