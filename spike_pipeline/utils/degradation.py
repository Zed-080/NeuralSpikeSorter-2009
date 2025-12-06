import numpy as np
from scipy import signal

# --- Constants matching src/utils/noise_matcher.py ---
FS = 25000
BP_LOW = 7
BP_HIGH = 3000


def bandpass_filter(x, fs=FS, low=BP_LOW, high=BP_HIGH, order=4):
    """Standard neural band-pass filter."""
    nyq = fs * 0.5
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    return signal.filtfilt(b, a, x).astype(np.float32)


def estimate_noise_std(x):
    """
    Robust estimation of noise standard deviation from src.
    Excludes high-amplitude spikes (threshold of 3.0).
    """
    mask = np.abs(x) < 3.0
    if mask.sum() < 100:
        return float(np.std(x))
    return float(np.std(x[mask]))


def degrade_with_spectral_noise(clean_signal, noise_ref, noise_scale=1.0):
    """
    MATCHES SRC: 'noise_match_d1'.

    Takes RAW D1 and RAW Target.
    1. Estimates noise floor from Bandpassed Target.
    2. Generates matching Gaussian noise.
    3. Adds noise to RAW D1.
    4. Bandpasses the result.
    """
    rng = np.random.default_rng(42)

    # 1. Bandpass target to get true noise stats (excludes DC drift)
    tgt_bp = bandpass_filter(noise_ref)
    tgt_noise_std = estimate_noise_std(tgt_bp)

    # 2. Generate White Noise scaled to target level
    #    (src generates unit gaussian * scaled ratio. Math simplifies to this)
    noise = rng.normal(0.0, tgt_noise_std,
                       size=clean_signal.shape) * noise_scale

    # 3. Add to RAW D1
    degraded_raw = clean_signal + noise

    # 4. Final Bandpass (Crucial: src always bandpasses the noisy output)
    return bandpass_filter(degraded_raw)


def add_noise_to_target_snr(x, target_snr_db):
    """Legacy helper for simple Gaussian augmentation on windows."""
    sig_power = np.mean(x ** 2)
    if sig_power <= 1e-12:
        return x
    snr_lin = 10.0 ** (target_snr_db / 10.0)
    noise_std = np.sqrt(sig_power / snr_lin)
    return (x + np.random.normal(0.0, noise_std, size=x.shape)).astype(np.float32)
