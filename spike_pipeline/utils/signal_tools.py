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


def build_average_mother(d1_signal, d1_idx):
    """Build mother wavelet from D1 training spikes"""
    capture_width = 80
    capture_weight = 0.8
    neg_off = int(capture_width * (1 - capture_weight))
    pos_off = capture_width - neg_off

    captures_list_all = []
    for idx in d1_idx:
        start = idx - neg_off
        end = idx + pos_off
        if start >= 0 and end < len(d1_signal):
            captures_list_all.append(np.array(d1_signal[start:end]))

    if len(captures_list_all) == 0:
        return None, None, None

    windows = np.vstack(captures_list_all)
    avg = np.mean(windows, axis=0)
    psi = avg - np.mean(avg)
    psi = psi / np.sqrt(np.sum(psi ** 2) + 1e-8)

    return psi, avg, windows


def matched_filter_denoise(signal, psi):
    """Apply matched filter for denoising"""
    psi_rev = psi[::-1]
    filtered = np.convolve(signal, psi_rev, mode='same')
    return filtered
