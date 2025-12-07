import numpy as np
from scipy import signal

# --- Constants matching src/utils/noise_matcher.py ---
FS = 24000
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

# import numpy as np
# from scipy import signal

# # --- Constants matching src/utils/noise_matcher.py ---
# FS = 24000
# BP_LOW = 7
# BP_HIGH = 3000
# RNG_SEED = 42


# def bandpass_filter(x, fs=FS, low=BP_LOW, high=BP_HIGH, order=4):
#     """
#     Standard neural band-pass filter.
#     Matches 'bandpass' in src/utils/noise_matcher.py
#     """
#     nyq = fs * 0.5
#     b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
#     # Cast to float32 to ensure bit-exact match with src
#     return signal.filtfilt(b, a, x).astype(np.float32)


# def estimate_noise_std(x):
#     """
#     Robust estimation of noise standard deviation.
#     Matches 'estimate_noise_std' in src/utils/noise_matcher.py
#     """
#     mask = np.abs(x) < 3.0
#     if mask.sum() < 100:
#         return float(np.std(x))
#     return float(np.std(x[mask]))


# def gaus_snr(x, snr_db, rng):
#     """
#     Adds Gaussian noise to achieve specific SNR dB relative to signal power.
#     Matches 'gaus_snr' in src/utils/noise_matcher.py
#     """
#     sig_power = np.mean(x**2)
#     if sig_power < 1e-12:
#         return x.copy()

#     snr_lin = 10 ** (snr_db / 10.0)
#     noise_std = np.sqrt(sig_power / snr_lin)

#     return x + rng.normal(0, noise_std, size=x.shape)


# def degrade_with_spectral_noise(clean_signal, noise_ref, noise_scale=1.0):
#     """
#     Replicates 'noise_match_d1' from src exactly.

#     Arguments:
#         clean_signal: Raw D1 array (un-normalized)
#         noise_ref:    Raw Target array (un-normalized)
#         noise_scale:  Scaling factor (default 1.0)

#     Algorithm:
#     1. Bandpass target to get noise floor.
#     2. Add 'Base Noise' (5dB) to Clean D1.
#     3. Subtract Clean D1 from (2) to isolate the synthetic noise component.
#     4. Scale that component to match target noise floor.
#     5. Add to Clean D1.
#     6. Bandpass result.
#     """
#     # CRITICAL: Re-initialize RNG every time to match src's behavior
#     # src/utils/noise_matcher.py initializes rng inside the function.
#     rng = np.random.default_rng(RNG_SEED)

#     # 1. Bandpass target (tgt_bp)
#     tgt_bp = bandpass_filter(noise_ref)

#     # 2. Create "Base" Noisy Signal (5dB SNR)
#     #    src does this to establish a noise shape relative to the signal amplitude
#     d1_base = gaus_snr(clean_signal, 5, rng)

#     # 3. Get Target Noise Std from bandpassed target
#     tgt_std = estimate_noise_std(tgt_bp)

#     # 4. Isolate the "Added" Noise
#     added_noise = d1_base - clean_signal

#     # 5. Calculate std of added noise
#     added_std = np.std(added_noise)
#     if added_std < 1e-8:
#         added_std = 1e-8

#     # 6. Scale the "Added" noise to match Target, then add to Raw D1
#     #    Formula: D1 + (BaseNoise * (TargetStd / BaseStd))
#     scaled_signal = clean_signal + added_noise * \
#         (tgt_std / added_std) * noise_scale

#     # 7. Final Bandpass (src always bandpasses the final output)
#     return bandpass_filter(scaled_signal)
