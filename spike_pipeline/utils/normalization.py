import numpy as np


def zscore(x):
    """
    Global z-score normalization.
    """
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma


def normalize_window(w):
    """
    Normalize a single extracted window.
    """
    mu = np.mean(w)
    sigma = np.std(w) + 1e-8
    return (w - mu) / sigma
