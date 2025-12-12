# ==============================================================================
# NOISE SYNTHESIS & SIGNAL DEGRADATION
# ==============================================================================
# Utilities for creating synthetic "noisy" datasets by injecting colored noise.
#
# 1. STRATEGY: SPECTRAL MATCHING
#    - Real neural recordings have specific noise colors (1/f) and artifacts.
#    - Instead of adding simple white Gaussian noise, we:
#      a. Analyze a target dataset (e.g., D3) to estimate its noise floor.
#      b. Generate noise scaled to that level.
#      c. Add it to the clean D1 signal.
#
# 2. GOAL
#    - Create a training set that mathematically resembles the unlabelled
#      datasets (D2-D6), allowing the model to learn robust features.
# ==============================================================================

import numpy as np
from spike_pipeline.utils.signal_tools import bandpass_filter


def estimate_noise_std(x):
    """
    Robustly estimates background noise standard deviation.
    Excludes high-amplitude events (spikes > 3.0 sigma) to avoid skewing the stat.
    """
    mask = np.abs(x) < 3.0
    # Safety check: if signal is too short or has too many artifacts, use full std
    if mask.sum() < 100:
        return float(np.std(x))
    return float(np.std(x[mask]))


def degrade_with_spectral_noise(clean_signal, noise_ref, noise_scale=1.0):
    """
    Synthesizes a noisy signal by injecting noise derived from a reference recording.

    Args:
        clean_signal: The ground-truth D1 signal.
        noise_ref: A raw recording (e.g., D3) to emulate.
        noise_scale: Multiplier to adjust severity.

    Returns:
        The degraded, bandpassed signal ready for training.
    """
    rng = np.random.default_rng(42)

    # 1. Bandpass target to isolate the relevant noise band (excludes DC drift)
    tgt_bp = bandpass_filter(noise_ref)
    tgt_noise_std = estimate_noise_std(tgt_bp)

    # 2. Generate White Noise scaled to the target's noise level
    noise = rng.normal(0.0, tgt_noise_std,
                       size=clean_signal.shape) * noise_scale

    # 3. Add synthetic noise to the clean signal
    degraded_raw = clean_signal + noise

    # 4. Final Bandpass to ensure spectral consistency
    return bandpass_filter(degraded_raw)
