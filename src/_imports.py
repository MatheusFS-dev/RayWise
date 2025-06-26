"""
Central imports module for the project.

This module organizes all necessary libraries and modules into logical groups
for better maintainability and readability.

Example usage:
    from _imports import *
"""

# ————————————————————————————— STANDARD LIBRARY ————————————————————————————— #
import gc
import math
import shutil
import signal
import subprocess
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import *

# ————————————————————————————— JUPYTER / IPYTHON ———————————————————————————— #
from IPython.display import HTML, Image, clear_output, display

# —————————————————————————————— ERROR HANDLING —————————————————————————————— #
import pretty_errors

# —————————————————————————————— DATA PROCESSING ————————————————————————————— #
import fireducks.pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

# ———————————————————————— Scikit-learn preprocessing ———————————————————————— #
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

# ———————————————————————————————— TENSORFLOW ———————————————————————————————— #
import tensorflow as tf

# Configure TensorFlow logging
tf.get_logger().setLevel("ERROR")

# Keras core modules
from tensorflow.keras import (
    Model,
    callbacks,
    initializers,
    layers,
    losses,
    metrics,
    mixed_precision,
    optimizers,
    regularizers,
)
from tensorflow.keras.backend import clear_session

# TensorFlow profiling
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# ———————————————————————— HYPERPARAMETER OPTIMIZATION ——————————————————————— #
import optuna
from optuna.trial import TrialState
from optuna.integration import KerasPruningCallback

# ——————————————————————————————— ARARAS MODULE —————————————————————————————— #
# Model builders
from araras.keras.builders.cnn import build_cnn1d, build_dense_as_conv1d
from araras.keras.builders.dnn import build_dnn

# Callbacks and utilities
from araras.keras.callbacks.nan_loss_pruner import NanLossPrunerCallback
from araras.keras.hparams import HParams

# Model analysis and optimization
from araras.keras.utils.profiler import get_flops, get_macs, get_memory_and_time
from araras.keras.utils.punish import (
    compute_flops_penalized_loss,
    compute_params_penalized_loss,
)
from araras.keras.utils.summary import capture_model_summary

# Araras Optuna utilities
from araras.optuna.analyze import analyze_study
from araras.optuna.utils import (
    cleanup_non_top_trials,
    get_remaining_trials,
    get_top_trials,
    init_study_dirs,
    rename_top_k_files,
    save_top_k_trials,
)

# Plotting configuration
from araras.plot.configs import config_plt

# Utility functions
from araras.utils.dir import create_run_directory
from araras.utils.gpu import get_gpu_info, get_user_gpu_choice
from araras.utils.logs import log_resources
from araras.utils.misc import (
    clear,
    format_number,
    format_scientific,
    format_bytes,
    format_number_commas,
)

# ————————————————————————————————— Optional ————————————————————————————————— #
try:
    import pretty_errors
except ImportError:
    print(
        "WARNING: pretty_errors not found. Install it with `pip install pretty-errors` for better error formatting."
    )

# ————————————————————————————— PROJECT SPECIFIC ————————————————————————————— #
from _load_dataset import load_dataset_sparse_labels
