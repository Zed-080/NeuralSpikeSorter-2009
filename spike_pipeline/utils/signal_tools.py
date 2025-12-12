import numpy as np
from scipy import signal


def mad(x):
    """
    Median absolute deviation for robust thresholding.
    """
    return np.median(np.abs(x - np.median(x))) / 0.6745


def bandpass_filter(x, fs=25000, low=7, high=3000, order=4):
    """
    Standard neural band-pass filter.
    """
    b, a = signal.butter(
        order,
        [low / (fs / 2), high / (fs / 2)],
        btype="band"
    )
    return signal.filtfilt(b, a, x)


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
