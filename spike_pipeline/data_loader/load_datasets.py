# ==============================================================================
# DATASET LOADING & NORMALIZATION UTILITIES
# ==============================================================================
# Handles the loading of MATLAB (.mat) recording files and applies standard
# preprocessing to ensure consistent input for the models.
#
# 1. FILE HANDLING
#    - Supports both labelled datasets (D1 with Index/Class vectors) and 
#      unlabelled datasets (D2-D6 with raw signal only).
#    - Standardizes all integer labels to 0-based indexing for Python compatibility.
#
# 2. NORMALIZATION
#    - Applies Global Z-Score Normalization: (x - mean) / std.
#    - This ensures the neural networks receive data with unit variance, 
#      which is critical for convergence and stable weights.
#
# 3. SIGNAL FORMAT
#    - All signals are cast to float32 to optimize memory usage during training.
# ==============================================================================

import numpy as np
import scipy.io as spio
from spike_pipeline.utils.signal_tools import zscore as global_normalize


def load_mat(path):
    """
    Loads a .mat file and extracts raw signal and labels (if present).
    Returns None for Index/Class if they don't exist (D2-D6).
    """
    mat = spio.loadmat(path, squeeze_me=True)

    # Ensure float32 for memory efficiency
    d = mat["d"].astype(np.float32)

    Index = mat.get("Index", None)
    Class = mat.get("Class", None)

    # Standardize labels to 1D arrays and 0-based indexing
    if Index is not None:
        Index = np.array(Index, dtype=np.int64).reshape(-1)
        Index -= 1
    if Class is not None:
        Class = np.array(Class, dtype=np.int64).reshape(-1)

    return d, Index, Class


def load_D1(path, fs=25000):
    """
    Loads the training dataset (D1) and applies global normalization.
    Returns: d_norm, Index, Class
    """
    d, Index, Class = load_mat(path)
    d_norm = global_normalize(d)
    return d_norm, Index, Class


def load_unlabelled(path, fs=25000):
    """
    Loads evaluation datasets (D2-D6) which only contain raw signals.
    Returns: d_norm
    """
    d, _, _ = load_mat(path)
    d_norm = global_normalize(d)
    return d_norm
