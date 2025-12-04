import numpy as np
from pathlib import Path
from typing import Tuple

from spike_pipeline.data_loader.load_datasets import load_D1


def build_average_spike_template(
    d1_signal: np.ndarray,
    d1_idx: np.ndarray,
    capture_width: int = 80,
    capture_weight: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an average spike template (mother wavelet) from D1.

    Parameters
    ----------
    d1_signal : array
        Clean D1 signal (normalized)
    d1_idx : array
        Spike indices from D1 (0-based)
    capture_width : int
        Total window size around each spike
    capture_weight : float
        Fraction of window after spike peak (0.8 = 16 before, 64 after)

    Returns
    -------
    psi : array
        Normalized template for matched filtering (unit energy)
    avg : array
        Raw average spike waveform
    windows : array
        All individual spike windows stacked (for inspection)
    """
    captures_list = []
    neg_off = int(capture_width * (1 - capture_weight))
    pos_off = capture_width - neg_off

    for idx in d1_idx:
        start = idx - neg_off
        end = idx + pos_off

        # Skip spikes too close to edges
        if start < 0 or end > len(d1_signal):
            continue

        window = d1_signal[start:end]
        if len(window) == capture_width:
            captures_list.append(window)

    if len(captures_list) == 0:
        raise ValueError("No valid spike windows extracted!")

    windows = np.vstack(captures_list)  # (num_spikes, capture_width)
    avg = np.mean(windows, axis=0)

    # Normalize: remove DC offset and scale to unit energy
    psi = avg - np.mean(avg)
    energy = np.sqrt(np.sum(psi ** 2))
    if energy > 1e-10:
        psi = psi / energy

    return psi.astype(np.float32), avg.astype(np.float32), windows


def matched_filter_enhance(signal: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Apply matched filter to enhance spike-like patterns in noisy signal.

    Parameters
    ----------
    signal : array
        Input signal (can be raw or lightly preprocessed)
    psi : array
        Spike template from build_average_spike_template

    Returns
    -------
    filtered : array
        Matched-filtered signal (same length as input)
    """
    signal = np.asarray(signal, dtype=np.float32)
    psi = np.asarray(psi, dtype=np.float32)

    # Reverse template for convolution (matched filter = correlation)
    psi_rev = psi[::-1]

    # Convolve and maintain same length
    filtered = np.convolve(signal, psi_rev, mode='same')

    return filtered.astype(np.float32)
