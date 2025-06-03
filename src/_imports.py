"""
This module imports all the necessary libraries and modules for the project.

Example usage:
    from _imports import *

"""

# ———————————————————————————— Standard Libraries ———————————————————————————— #
import gc, math, signal, shutil, traceback, subprocess
from IPython.display import clear_output, display, HTML, Image

# ———————————————————————————————— Annotations ——————————————————————————————— #
from typing import *

# ————————————————————————— Data Processing Libraries ———————————————————————— #
import numpy as np
import fireducks.pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

# ——————————————————————— TensorFlow and Keras Modules ——————————————————————— #
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras import (
    layers,
    Model,
    callbacks,
    optimizers,
    regularizers,
    metrics,
    losses,
    mixed_precision,
    initializers,
)
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# Set TensorFlow logger level to ERROR
tf.get_logger().setLevel("ERROR")

# ——————————————————————————— Scikit-learn Modules ——————————————————————————— #
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# —————————————————————————————————— Optuna —————————————————————————————————— #
import optuna
from optuna.integration import KerasPruningCallback

# import optunahub  # For the AutoSampler

# —————————————————————————————————— Araras —————————————————————————————————— #
# Keras-related
from araras.keras.builders.dnn import build_dnn
from araras.keras.builders.cnn import build_cnn1d, build_dense_as_conv1d
from araras.keras.callbacks.nan_loss_pruner import NanLossPrunerCallback
from araras.keras.hparams import HParams
from araras.keras.utils.punish import (
    compute_flops_penalized_loss,
    compute_params_penalized_loss,
)
from araras.keras.utils.profiler import get_flops, get_macs

# Optuna
from araras.optuna.analyze import analyze_study

# Email notifications
from araras.email.utils import send_email, notify_training_success

# Monitoring
from araras.kernel.monitoring import start_monitor, stop_monitor

# Plotting
from araras.plot.configs import config_plt

# Utility functions
from araras.utils.dir import create_run_directory
from araras.utils.files import save_trial_params_to_file
from araras.utils.gpu import get_gpu_info
from araras.utils.logs import log_resources, log_exception_to_file

# —————————————————————————————————— Dataset ————————————————————————————————— #
from _load_dataset import load_dataset_sparse_labels
