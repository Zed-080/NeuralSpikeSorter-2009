import numpy as np


def zscore(x):
    """
    Global z-score normalization.
    """
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma


def normalize_window(w, eps=1e-8):
    """
    Per-window z-score normalisation:
    (w - mean) / std  for a single 1D window.
    Adds epsilon to std to avoid divide-by-zero.
    """
    mu = np.mean(w)
    sigma = np.std(w)
    return (w - mu) / (sigma + eps)
