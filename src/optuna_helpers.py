"""Helper utilities for running Optuna studies."""

from __future__ import annotations

import os
import shutil
import traceback
from typing import Callable, Iterable, Sequence

import optuna

from araras.ml.optuna.callbacks import (
    ImprovementStagnation,
    StopIfKeepBeingPruned,
    StopWhenNoValueImprovement,
)
from araras.ml.optuna.utils import (
    get_remaining_trials,
    cleanup_non_top_trials,
    rename_top_k_files,
    get_top_trials,
    save_top_k_trials,
    init_study_dirs,
)
from araras.ml.optuna.analyzer import analyze_study
from araras.utils.misc import clear
# Public API --------------------------------------------------------------------

def run_study(
    objective: Callable[[optuna.Trial], float],
    run_dir: str,
    *,
    epochs: int,
    num_trials: int,
    sampler_seed: int,
    direction: str,
    top_k: int,
    rank_key: str,
    order: str,
    extra_attrs: Sequence[str] | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    patience: int = 100,
    prune_threshold: int = 50,
) -> optuna.study.Study:
    """Run an Optuna hyperparameter optimization study.

    This helper configures the directory structure, runs the optimization and
    consolidates the results. It mirrors the notebook logic while exposing
    parameters for customization.

    Notes:
        - If an unexpected exception occurs the error is logged to
          ``training_error.log`` and the process aborts so an external
          monitor can restart it.

    Warning:
        The temporary ``backup_dir`` is removed regardless of success.

    Args:
        objective: Function that trains a model and returns the metric to
            optimize.
        run_dir: Base directory for the study run.
        epochs: Number of training epochs per trial.
        num_trials: Total number of trials to evaluate.
        sampler_seed: Seed for the Optuna sampler.
        direction: Optimization direction, e.g. ``"minimize"`` or ``"maximize"``.
        top_k: Number of best trials to keep after optimization.
        rank_key: Study attribute used for ranking trials.
        order: Sorting order for ranking trials.
        extra_attrs: Additional user attributes copied when saving the top trials.
        pruner: Optional custom Optuna pruner. Defaults to ``HyperbandPruner``.
        sampler: Optional custom Optuna sampler. Defaults to ``TPESampler``.
        patience: Patience for the value improvement callback.
        prune_threshold: Threshold for the pruning callback.

    Returns:
        optuna.study.Study: The completed study object.

    Raises:
        Exception: Propagates any exception raised during optimization.
    """

    (
        study_dir,
        args_dir,
        fig_dir,
        backup_dir,
        history_dir,
        model_dir,
        logs_dir,
        tensorboard_dir,
    ) = init_study_dirs(run_dir)

    try:
        study = optuna.create_study(
            study_name=os.path.basename(study_dir),
            storage=f"sqlite:///{study_dir}/optuna_study.db",
            pruner=pruner or optuna.pruners.HyperbandPruner(),
            sampler=sampler or optuna.samplers.TPESampler(seed=sampler_seed),
            load_if_exists=True,
            direction=direction,
        )

        study.optimize(
            lambda trial: objective(
                trial,
                epochs=epochs,
                size_penalizer=None,
                **{
                    "backup_dir": backup_dir,
                    "model_dir": model_dir,
                    "fig_dir": fig_dir,
                    "logs_dir": logs_dir,
                    "tensorboard_dir": tensorboard_dir,
                    "history_dir": history_dir,
                },
            ),
            n_trials=get_remaining_trials(study, num_trials),
            callbacks=[
                ImprovementStagnation(),
                StopIfKeepBeingPruned(threshold=prune_threshold),
                StopWhenNoValueImprovement(patience=patience),
            ],
            catch=(),
            gc_after_trial=True,
        )

        top_trials = get_top_trials(
            study,
            top_k=top_k,
            rank_key=rank_key,
            order=order,
        )

        cleanup_paths = [
            (model_dir, "trial_{trial_id}.keras"),
            (fig_dir, "trial_{trial_id}.png"),
            (history_dir, "trial_{trial_id}.csv"),
            (tensorboard_dir, "trial_{trial_id}"),
        ]

        rename_paths = [
            (model_dir, ".keras"),
            (fig_dir, ".png"),
            (history_dir, ".csv"),
        ]

        save_top_k_trials(
            top_trials,
            args_dir=args_dir,
            study=study,
            extra_attrs=extra_attrs or [],
        )
        cleanup_non_top_trials(
            {t.number for t in study.trials},
            {t.number for t in top_trials},
            cleanup_paths,
        )
        rename_top_k_files(top_trials, rename_paths)

        clear(), analyze_study(study, table_dir=os.path.join(study_dir, "analysis"))
        return study

    except Exception as e:  # pragma: no cover - runtime failure path
        print(f"\n An error occurred: {e}\n")
        traceback.print_exc()

        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "training_error.log"), "a") as f:
            f.write(
                f"An error occurred during training:\n{e}\n{traceback.format_exc()}\n\n"
            )

        os.abort()
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
        if os.path.isdir(logs_dir) and not os.listdir(logs_dir):
            os.rmdir(logs_dir)
