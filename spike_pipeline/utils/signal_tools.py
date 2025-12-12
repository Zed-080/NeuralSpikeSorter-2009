# ==============================================================================
# SIGNAL PROCESSING UTILITIES
# ==============================================================================
# A collection of generic mathematical tools for signal conditioning.
#
# 1. FILTERING
#    - Bandpass Filter: 4th order Butterworth (7Hz - 3000Hz).
#      Removes low-frequency LFP drift and high-frequency thermal noise, isolating
#      the neural spike frequency band.
#
# 2. STATISTICS
#    - MAD (Median Absolute Deviation): A robust measure of variance that is
#      resilient to outliers (like large spikes). Used for thresholding.
#
# 3. NORMALIZATION
#    - Z-Score: Standardizes data to 0 mean and 1 variance.
# ==============================================================================

import numpy as np
from scipy import signal


def mad(x):
    """
    Calculates Median Absolute Deviation (MAD) for robust noise estimation.
    Scale factor 0.6745 makes it consistent with standard deviation for Gaussian noise.
    """
    return np.median(np.abs(x - np.median(x))) / 0.6745


def bandpass_filter(x, fs=25000, low=7, high=3000, order=4):
    """
    Applies a standard neural band-pass filter (Butterworth).

    Args:
        x: Input signal (1D array).
        fs: Sampling frequency (Hz). Default 25kHz (Coursework spec).
        low: Low cut-off freq (Hz).
        high: High cut-off freq (Hz).
        order: Filter order.
    """
    b, a = signal.butter(
        order,
        [low / (fs / 2), high / (fs / 2)],
        btype="band"
    )
    return signal.filtfilt(b, a, x)


def zscore(x):
    """
    Applies Global Z-Score normalization: (x - mean) / std.
    Adds epsilon to prevent division by zero.
    """
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma


def normalize_window(w, eps=1e-8):
    """
    Applies Per-Window Z-Score normalization.
    Useful for normalizing short clips individually.
    """
    mu = np.mean(w)
    sigma = np.std(w)
    return (w - mu) / (sigma + eps)
