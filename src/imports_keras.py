"""
Central imports module for the project.

Usage:
    from utils.imports_keras import *
"""

from typing import *

# ————————————————————————————— STANDARD LIBRARY ————————————————————————————— #
import os, gc, shutil, traceback
from pathlib import Path
import pickle

# —————————————————————————————— DATA PROCESSING ————————————————————————————— #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ———————————————————————— Scikit-learn preprocessing ———————————————————————— #
from sklearn.model_selection import train_test_split

# ———————————————————————————————— TENSORFLOW ———————————————————————————————— #
import tensorflow as tf

# ——————————————————————————————————— Keras —————————————————————————————————— #
from tensorflow.keras import (
    Model,
    callbacks,
    initializers,
    activations,
    layers,
    losses,
    metrics,
    mixed_precision,
    optimizers,
    regularizers,
)
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed

# —————————————————————————————————— Optuna —————————————————————————————————— #
import optuna, optunahub
from optuna.integration import KerasPruningCallback
from optuna.terminator import report_cross_validation_scores

# ——————————————————————————————— Scikit-Learn ——————————————————————————————— #
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)

# —————————————————————————————————— Araras —————————————————————————————————— #
# Model builders
from araras.ml.model.builders.dnn import *

# Callbacks
from araras.ml.optuna.callbacks import (
    NanLossPrunerOptuna,
    get_callbacks_study,
    StopIfKeepBeingPruned,
    ImprovementStagnation,
    StopWhenNoValueImprovement,
)
from araras.ml.model.callbacks import get_callbacks_model

# Araras utilities
from araras.ml.model.hyperparams import KParams
from araras.ml.model.tools import punish_model, print_tensor_mem, validate_steps_per_execution
from araras.ml.optuna.utils import (
    get_remaining_trials,
    cleanup_non_top_trials,
    rename_top_k_files,
    get_top_trials,
    save_top_k_trials,
    init_study_dirs,
    log_trial_error,
    run_study,
)
from araras.ml.optuna.analyzer import analyze_study
from araras.ml.optuna.model_tools import set_user_attr_model_stats, prune_model_by_config, plot_model_param_distribution
from araras.ml.model.stats import write_model_stats_to_file

# Plotting configuration
from araras.visualization.configs import config_plt

# Utility functions
from araras.utils.io import get_caller_stem, select_path
from araras.utils.system import get_gpu_info, log_resources
from araras.utils.misc import clear

# ————————————————————————————— PROJECT SPECIFIC ————————————————————————————— #
