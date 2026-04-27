"""Run Optuna NAS over stacking ensemble model count and dataset seed.

This script reuses the training/evaluation pipeline from
`ensemble_cnn1d_stacking.py` and optimizes two knobs:
1) number of top-ranked base models used for stacking,
2) dataset split/random seed used by the data pipeline.
"""

import json
import re
import shutil
from pathlib import Path

import optuna

import ensemble_cnn1d_stacking_v2 as stacking

# ———————————————————————————————————————————————————————————————————————————— #
#                              Global Configuration                             #
# ———————————————————————————————————————————————————————————————————————————— #
# Optuna search budget.
NUM_TRIALS = 80

# Search range for DATA_SEED used by the base pipeline.
DATA_SEED_MIN = 0
DATA_SEED_MAX = 4

# Global upper bound for top-X search. The actual upper bound is clipped by
# the number of ranked models found in the selected model directory.
TOP_X_MODELS_MIN = 3
TOP_X_MODELS_MAX = 10

# NAS-specific AutoGluon profile.
# These overrides are applied to the reused stacking pipeline so NAS trials can
# run faster while preserving a diverse learner set.
NAS_AG_PRESETS = "medium_quality_faster_train"
NAS_AG_HYPERPARAMETERS = {
    "GBM": [{}],
    "CAT": [{}],
    "XGB": [{}],
    "FASTAI": [{}],
}
NAS_AG_TIME_LIMIT_SECONDS = 900
NAS_AG_NUM_STACK_LEVELS = 0
NAS_AG_NUM_BAG_FOLDS = 3
NAS_AG_NUM_BAG_SETS = 1
NAS_AG_USE_BAG_HOLDOUT = True
NAS_AG_FIT_STRATEGY = "sequential"
NAS_AG_FOLD_FITTING_STRATEGY = "sequential_local"
NAS_AG_NUM_FOLDS_PARALLEL = 1
NAS_AG_NUM_CPUS = 12
NAS_AG_VERBOSITY = 2

# Persist NAS artifacts in a dedicated run folder.
RUN_DIR = Path("runs") / Path(__file__).stem
STUDY_DB_PATH = RUN_DIR / "optuna_study.db"
STUDY_NAME = "stacking_topx_seed"
BEST_TRIAL_DIR = RUN_DIR / "best_trial"


def apply_nas_training_profile() -> None:
    """Apply NAS-specific AutoGluon settings to the shared stacking module.

    The stacking script is reused as an imported module during NAS trials.
    This helper overrides the module-level training globals so NAS uses its
    own lightweight training profile, independent from standalone ensemble
    runs.

    Returns:
        None: Stacking module globals are mutated in-place for the current
        Python process.
    """
    stacking.AG_PRESETS = NAS_AG_PRESETS
    stacking.AG_HYPERPARAMETERS = NAS_AG_HYPERPARAMETERS
    stacking.AG_TIME_LIMIT_SECONDS = NAS_AG_TIME_LIMIT_SECONDS
    stacking.AG_NUM_STACK_LEVELS = NAS_AG_NUM_STACK_LEVELS
    stacking.AG_NUM_BAG_FOLDS = NAS_AG_NUM_BAG_FOLDS
    stacking.AG_NUM_BAG_SETS = NAS_AG_NUM_BAG_SETS
    stacking.AG_USE_BAG_HOLDOUT = NAS_AG_USE_BAG_HOLDOUT
    stacking.AG_FIT_STRATEGY = NAS_AG_FIT_STRATEGY
    stacking.AG_FOLD_FITTING_STRATEGY = NAS_AG_FOLD_FITTING_STRATEGY
    stacking.AG_NUM_FOLDS_PARALLEL = NAS_AG_NUM_FOLDS_PARALLEL
    stacking.AG_NUM_CPUS = NAS_AG_NUM_CPUS
    stacking.AG_VERBOSITY = NAS_AG_VERBOSITY


def count_ranked_models(model_dir: Path) -> int:
    """Count ranked model files available for top-X selection.

    Ranked model files are expected to follow the naming convention
    `top_<rank>_... .keras`.

    Args:
        model_dir (Path): Directory that stores exported `.keras` checkpoints.

    Returns:
        int: Number of files in `model_dir` that match the ranked naming
        convention.

    Raises:
        FileNotFoundError: If `model_dir` does not exist.
        NotADirectoryError: If `model_dir` exists but is not a directory.

    Examples:
        >>> count_ranked_models(Path("/tmp/models"))
        10
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory path is not a directory: {model_dir}")

    pattern = re.compile(r"^top_(\d+)_.*\.keras$")
    ranked_count = 0

    # Only ranked files are counted so Optuna cannot suggest an invalid top-X.
    for path in model_dir.glob("*.keras"):
        if pattern.match(path.name):
            ranked_count += 1

    return ranked_count


def load_validation_top1(results_path: Path) -> float:
    """Load validation Top-1 accuracy produced by the stacking pipeline.

    Args:
        results_path (Path): Path to the JSON report written by
            `ensemble_cnn1d_stacking.main`.

    Returns:
        float: Validation Top-1 accuracy in `[0, 1]`.

    Raises:
        FileNotFoundError: If `results_path` is missing.
        KeyError: If expected validation keys are absent in the JSON payload.
        ValueError: If the loaded value cannot be interpreted as `float`.

    Examples:
        >>> load_validation_top1(Path("runs/ensemble_cnn1d_stacking/results.json"))
        0.8125
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    payload = json.loads(results_path.read_text())
    val_top1 = payload["meta_val_topk"]["top_1"]
    return float(val_top1)


def objective(trial: optuna.Trial, model_dir: Path, max_top_x: int, results_path: Path) -> float:
    """Execute one Optuna trial and return validation Top-1 accuracy.

    Args:
        trial (optuna.Trial): Active Optuna trial object used to sample
            hyperparameters and record metadata.
        model_dir (Path): Directory containing ranked base model checkpoints.
        max_top_x (int): Maximum allowed value for `top_x_models` in this
            study. This value is clipped by discovered ranked model count.
        results_path (Path): File path where the base stacking pipeline writes
            evaluation metrics for each run.

    Returns:
        float: Validation Top-1 accuracy to maximize.

    Raises:
        RuntimeError: If Optuna proposes an invalid search range.
        FileNotFoundError: If trial execution does not generate `results_path`.
        KeyError: If validation metrics are missing from result JSON.

    Examples:
        This function is intended to be called by `study.optimize`.
    """
    if max_top_x < TOP_X_MODELS_MIN:
        raise RuntimeError(
            f"Invalid top-X search range: min={TOP_X_MODELS_MIN}, max={max_top_x}."
        )

    top_x_models = trial.suggest_int("top_x_models", TOP_X_MODELS_MIN, max_top_x)
    data_seed = trial.suggest_int("data_seed", DATA_SEED_MIN, DATA_SEED_MAX)

    # Inject the sampled dataset seed into the reused base pipeline so each
    # trial evaluates a distinct split/randomization regime.
    apply_nas_training_profile()
    stacking.DATA_SEED = int(data_seed)

    # Run the end-to-end stacking workflow for the sampled configuration.
    stacking.main(model_dir=model_dir, top_x_models=int(top_x_models))

    val_top1 = load_validation_top1(results_path)
    trial.set_user_attr("val_top1", val_top1)
    trial.set_user_attr("model_dir", str(model_dir))
    trial.set_user_attr("results_path", str(results_path))
    return val_top1


def _load_results(results_path: Path) -> dict:
    """Load trial report JSON generated by the stacking pipeline.

    Args:
        results_path (Path): Path to `results.json` produced by
            `ensemble_cnn1d_stacking.main`.

    Returns:
        dict: Parsed run metadata and metrics payload.

    Raises:
        FileNotFoundError: If `results_path` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    return json.loads(results_path.read_text())


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    """Copy a file when present, leaving destination unchanged otherwise.

    Args:
        src (Path): Source file path to copy.
        dst (Path): Destination file path.

    Returns:
        None: File is copied only when `src` exists.
    """
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def export_best_trial_artifacts(best_params: dict[str, int], model_dir: Path, results_path: Path) -> Path:
    """Re-run best config and persist a complete best-trial artifact bundle.

    This function executes the base stacking pipeline one more time using the
    best Optuna parameters, then copies all key artifacts to a stable folder.

    Args:
        best_params (dict[str, int]): Best trial parameters containing
            `top_x_models` and `data_seed`.
        model_dir (Path): Directory containing ranked base model checkpoints.
        results_path (Path): Path to base pipeline `results.json` output.

    Returns:
        Path: Destination directory containing exported best-trial artifacts.

    Raises:
        KeyError: If required parameter keys are absent from `best_params`.
        FileNotFoundError: If expected run artifacts are not generated.
        NotADirectoryError: If predictor artifact path is not a directory.
    """
    # Force a deterministic re-materialization of the best configuration so the
    # exported bundle always matches the recorded best parameters.
    apply_nas_training_profile()
    stacking.DATA_SEED = int(best_params["data_seed"])
    stacking.main(model_dir=model_dir, top_x_models=int(best_params["top_x_models"]))

    run_results = _load_results(results_path)
    artifacts = run_results.get("artifacts", {})

    predictor_path = Path(artifacts["autogluon_predictor"])
    if not predictor_path.exists():
        raise FileNotFoundError(f"AutoGluon predictor directory not found: {predictor_path}")
    if not predictor_path.is_dir():
        raise NotADirectoryError(f"AutoGluon predictor path is not a directory: {predictor_path}")

    if BEST_TRIAL_DIR.exists():
        shutil.rmtree(BEST_TRIAL_DIR)
    BEST_TRIAL_DIR.mkdir(parents=True, exist_ok=True)

    predictor_dst = BEST_TRIAL_DIR / "autogluon_predictor"
    shutil.copytree(predictor_path, predictor_dst)

    # Copy run-level metadata files required for reproducible inference.
    _copy_file_if_exists(results_path, BEST_TRIAL_DIR / "results.json")
    _copy_file_if_exists(stacking.RESULTS_DIR / "accuracies.txt", BEST_TRIAL_DIR / "accuracies.txt")

    coord_scaler_path = Path(artifacts.get("coord_scaler", "")) if artifacts.get("coord_scaler") else None
    classes_path = Path(artifacts.get("classes", "")) if artifacts.get("classes") else None
    class_to_pos_path = Path(artifacts.get("class_to_pos", "")) if artifacts.get("class_to_pos") else None

    if coord_scaler_path is not None:
        _copy_file_if_exists(coord_scaler_path, BEST_TRIAL_DIR / "coord_scaler.pkl")
    if classes_path is not None:
        _copy_file_if_exists(classes_path, BEST_TRIAL_DIR / "classes.npy")
    if class_to_pos_path is not None:
        _copy_file_if_exists(class_to_pos_path, BEST_TRIAL_DIR / "class_to_pos.json")

    # Export selected base model files and exact paths to preserve full model
    # stack (base ensemble + meta-learner) for future deployment.
    selected_model_paths = [Path(path_str) for path_str in run_results["config"]["model_paths"]]
    base_models_dir = BEST_TRIAL_DIR / "base_models"
    base_models_dir.mkdir(parents=True, exist_ok=True)
    for model_path in selected_model_paths:
        _copy_file_if_exists(model_path, base_models_dir / model_path.name)

    export_summary = {
        "model_dir": str(model_dir),
        "best_params": {k: int(v) for k, v in best_params.items()},
        "exported_predictor": str(predictor_dst),
        "exported_base_models": [str(base_models_dir / path.name) for path in selected_model_paths],
        "source_results": str(results_path),
    }
    (BEST_TRIAL_DIR / "export_summary.json").write_text(json.dumps(export_summary, indent=2))
    return BEST_TRIAL_DIR


def run_study() -> None:
    """Create and execute the Optuna study for stacking NAS.

    This function resolves the model directory from the base script defaults,
    computes a safe top-X search upper bound, runs the Optuna study, and saves
    best-trial metadata to disk.

    Returns:
        None: Study artifacts are persisted under `RUN_DIR`.

    Raises:
        FileNotFoundError: If the model directory cannot be resolved.
        NotADirectoryError: If the resolved model path is not a directory.
        RuntimeError: If no ranked `.keras` models are available.

    Examples:
        >>> run_study()
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve model source using the same precedence as the base script
    # (explicit args -> environment -> defaults), keeping both scripts aligned.
    model_dir, _ = stacking.resolve_model_selection()

    ranked_count = count_ranked_models(model_dir)
    if ranked_count == 0:
        raise RuntimeError(
            "No ranked models were found for NAS. Expected files like "
            f"'top_<rank>_... .keras' in {model_dir}."
        )

    max_top_x = min(TOP_X_MODELS_MAX, ranked_count)
    results_path = stacking.RESULTS_DIR / "results.json"

    # Use fixed storage and fixed study name so reruns always continue the
    # same study instead of creating timestamped branches.
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=True,
    )

    print("Starting Optuna study for stacking NAS")
    print(f"  model_dir: {model_dir}")
    print(f"  top_x range: [{TOP_X_MODELS_MIN}, {max_top_x}]")
    print(f"  data_seed range: [{DATA_SEED_MIN}, {DATA_SEED_MAX}]")
    print(f"  trials: {NUM_TRIALS}")

    study.optimize(
        lambda trial: objective(trial, model_dir=model_dir, max_top_x=max_top_x, results_path=results_path),
        n_trials=NUM_TRIALS,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best_summary = {
        "study_name": study.study_name,
        "storage": str(STUDY_DB_PATH),
        "direction": study.direction.name,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": {k: int(v) for k, v in study.best_params.items()},
        "best_user_attrs": {
            key: (float(value) if isinstance(value, (int, float)) else str(value))
            for key, value in study.best_trial.user_attrs.items()
        },
    }

    summary_path = RUN_DIR / "best_trial.json"
    summary_path.write_text(json.dumps(best_summary, indent=2))

    exported_dir = export_best_trial_artifacts(study.best_params, model_dir=model_dir, results_path=results_path)

    print("Optuna study finished")
    print(f"  best val Top-1: {study.best_value:.6f}")
    print(f"  best params: {study.best_params}")
    print(f"  summary: {summary_path}")
    print(f"  best artifacts: {exported_dir}")


if __name__ == "__main__":
    run_study()
