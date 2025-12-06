# spike_pipeline/noise/noise_matcher.py

#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.signal import butter, filtfilt
# << YOUR LOADERS
from spike_pipeline.data_loader.load_datasets import load_D1, load_mat
#fmt: on

FS = 24000
BP_LOW = 7
BP_HIGH = 3000
RNG_SEED = 42


# ------------------ Filtering ------------------

def bandpass(x):
    nyq = FS * 0.5
    b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


# ------------------ Noise Utilities ------------------

def estimate_noise_std(x):
    """Estimate background noise by ignoring spike peaks."""
    mask = np.abs(x) < 3.0
    if mask.sum() < 100:
        return float(np.std(x))
    return float(np.std(x[mask]))


def gaus_snr(x, snr_db, rng):
    """Add Gaussian noise to achieve the target SNR."""
    sig = np.mean(x ** 2)
    if sig < 1e-12:
        return x.copy()

    snr_lin = 10 ** (snr_db / 10)
    noise_std = np.sqrt(sig / snr_lin)

    return x + rng.normal(0, noise_std, size=x.shape)


# ------------------ Main Function ------------------

def noise_match_d1(target_name: str):
    """
    Reproduce EXACT behaviour of the original noise_match_d1()
    but using spike_pipeline loaders and paths.
    Returns a *bandpassed, noise-matched* version of D1_raw → target_name.
    """

    rng = np.random.default_rng(RNG_SEED)

    # 1) Load raw D1 + target from your pipeline's data loader
    d1_raw = load_mat("D1")          # raw signal (not normalised)
    tgt_raw = load_mat(target_name)

    # 2) Bandpass target only (same as original)
    tgt_bp = bandpass(tgt_raw)

    # 3) Inject base Gaussian noise to RAW D1
    d1_base = gaus_snr(d1_raw, 5, rng)

    # 4) Extract real noise floor from target
    tgt_std = estimate_noise_std(tgt_bp)

    # 5) Extract synthetic noise component from D1
    added = d1_base - d1_raw
    added_std = np.std(added)
    if added_std < 1e-8:
        added_std = 1e-8

    # 6) Scale noise to target noise level
    scaled = d1_raw + added * (tgt_std / added_std)

    # 7) Final bandpass → matches original EXACTLY
    matched = bandpass(scaled)

    return matched.astype(np.float32)
