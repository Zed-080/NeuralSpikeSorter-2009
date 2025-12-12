# ==============================================================================
# MATCHED FILTERING UTILITIES
# ==============================================================================
# Implements "Template Matching" to enhance spike visibility in noisy data.
#
# 1. THEORY
#    - A "Matched Filter" is the optimal linear filter for maximizing the Signal-to-Noise
#      Ratio (SNR) in the presence of additive stochastic noise.
#    - It works by cross-correlating the noisy signal with a known "template" of
#      the signal we are looking for (the spike).
#
# 2. STRATEGY
#    - We derive a "Mother Template" by averaging thousands of clean, labeled
#      spikes from the high-SNR D1 dataset.
#    - This template represents the "ideal" spike shape for this specific electrode.
#    - We then convolve this template with the noisy datasets (D2-D6). This amplifies
#      shapes that look like spikes and suppresses random noise.
# ==============================================================================

import numpy as np
from typing import Tuple


def build_average_spike_template(
    d1_signal: np.ndarray,
    d1_idx: np.ndarray,
    capture_width: int = 80,
    capture_weight: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the 'Mother Template' by averaging clean spikes from D1.

    Args:
        d1_signal: Normalized, clean D1 recording.
        d1_idx: Ground truth indices of spikes.
        capture_width: Window size to extract around each spike (e.g., 80).
        capture_weight: Positioning weight (0.8 means 80% of window is post-peak).

    Returns:
        psi (np.array): Normalized, zero-mean template (unit energy).
        avg (np.array): Raw average waveform (useful for visualization).
        windows (np.array): Stack of all extracted windows.
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

    windows = np.vstack(captures_list)
    avg = np.mean(windows, axis=0)

    # Normalize: remove DC offset and scale to unit energy
    psi = avg - np.mean(avg)
    energy = np.sqrt(np.sum(psi ** 2))

    if energy > 1e-10:
        psi = psi / energy

    return psi.astype(np.float32), avg.astype(np.float32), windows


def matched_filter_enhance(signal: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Convolves the signal with the spike template to maximize SNR.

    Args:
        signal: The raw or noisy input signal (1D array).
        psi: The normalized spike template from build_average_spike_template.

    Returns:
        filtered: The enhanced signal where peaks correspond to likely spike locations.
    """
    signal = np.asarray(signal, dtype=np.float32)
    psi = np.asarray(psi, dtype=np.float32)

    # Reverse template for convolution (Convolution with reversed kernel = Cross-Correlation)
    psi_rev = psi[::-1]

    # 'same' mode ensures the output length matches the input length
    filtered = np.convolve(signal, psi_rev, mode='same')

    return filtered.astype(np.float32)
