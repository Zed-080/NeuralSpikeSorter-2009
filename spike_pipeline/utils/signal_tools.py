import numpy as np
from scipy import signal


def mad(x):
    """
    Median absolute deviation for robust thresholding.
    """
    return np.median(np.abs(x - np.median(x))) / 0.6745


def bandpass_filter(x, fs=25000, low=300, high=3000, order=4):
    """
    Standard neural band-pass filter.
    """
    b, a = signal.butter(
        order,
        [low / (fs / 2), high / (fs / 2)],
        btype="band"
    )
    return signal.filtfilt(b, a, x)
