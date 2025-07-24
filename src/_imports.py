"""
Central imports module for the project.

Usage:
    from _imports import *
"""

from typing import *

# ————————————————————————————— STANDARD LIBRARY ————————————————————————————— #
import gc, shutil, traceback
from pathlib import Path

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
from araras.ml.model.builders.cnn import *
from araras.ml.model.builders.gnn import *

# Callbacks
from araras.ml.optuna.callbacks import (
    NanLossPrunerOptuna,
    get_callbacks_study,
    StopIfKeepBeingPruned,
    ImprovementStagnation,
)
from araras.ml.model.callbacks import get_callbacks_model

# Araras utilities
from araras.ml.model.hyperparams import KParams
from araras.ml.model.tools import punish_model
from araras.ml.optuna.utils import (
    get_remaining_trials,
    cleanup_non_top_trials,
    rename_top_k_files,
    get_top_trials,
    save_top_k_trials,
    init_study_dirs,
    log_trial_error,
)
from araras.ml.optuna.analyzer import analyze_study
from araras.ml.optuna.model_tools import set_user_attr_model_stats

# Plotting configuration
from araras.visualization.configs import config_plt

# Utility functions
from araras.utils.io import get_caller_stem
from araras.utils.system import get_gpu_info, log_resources
from araras.utils.misc import clear

# ————————————————————————————— PROJECT SPECIFIC ————————————————————————————— #
from _load_dataset import load_dataset_sparse_labels
