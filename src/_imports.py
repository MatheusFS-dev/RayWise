"""
Central imports module for the project.

This module organizes all necessary libraries and modules into logical groups
for better maintainability and readability.

Example usage:
    from _imports import *
"""

from typing import *

# ————————————————————————————— STANDARD LIBRARY ————————————————————————————— #
import gc
import shutil
import traceback
from pathlib import Path

# —————————————————————————————— DATA PROCESSING ————————————————————————————— #
import numpy as np
import fireducks.pandas as pd
import matplotlib.pyplot as plt

# ———————————————————————— Scikit-learn preprocessing ———————————————————————— #
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

# ———————————————————————— HYPERPARAMETER OPTIMIZATION ——————————————————————— #
import optuna
import optunahub
from optuna.trial import TrialState
from optuna.integration import KerasPruningCallback
from optuna.terminator import report_cross_validation_scores

# ——————————————————————————————— ARARAS MODULE —————————————————————————————— #
# Model builders
from araras.keras.builders.cnn import *
from araras.keras.builders.dnn import *
from araras.keras.builders.se import *
from araras.keras.builders.gnn import *

# Callbacks and utilities
from araras.keras.hparams import HParams
from araras.keras.callbacks.nan_loss_pruner import NanLossPrunerCallback

# Model analysis and optimization
from araras.keras.utils.profiler import *
from araras.keras.utils.punish import *
from araras.keras.utils.summary import *
from araras.tensorflow.utils.model import *

# Araras Optuna utilities
from araras.optuna.utils import *
from araras.optuna.analysis.analyze import analyze_study
from araras.optuna.keras.model_stats import get_model_stats
from araras.optuna.callbacks.keep_being_pruned import StopIfKeepBeingPruned
from araras.optuna.callbacks.improvement_stagnation import ImprovementStagnationCallback


# Plotting configuration
from araras.plot.configs import *

# Utility functions
from araras.utils.dir import *
from araras.utils.gpu import *
from araras.utils.logs import *
from araras.utils.misc import *

# ————————————————————————————— PROJECT SPECIFIC ————————————————————————————— #
from _load_dataset import load_dataset_sparse_labels
