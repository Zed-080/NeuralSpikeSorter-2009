"""
Wavelet + FFT-based denoising for D2–D6 datasets.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import scipy.io as spio
import pywt
from scipy.signal import butter, filtfilt
from spike_pipeline.denoise.matched_filter import build_average_spike_template, matched_filter_enhance
from spike_pipeline.data_loader.load_datasets import load_D1

# ----------------- Constants & paths -----------------

FS_DEFAULT = 24000  # was 25000.0  # Hz

# Robustly find root (2 levels up from denoise)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Point explicitly to the 'data' folder
DATA_RAW = PROJECT_ROOT / "data"
DATA_DENOISED = PROJECT_ROOT / "outputs" / "denoised"
DATA_DENOISED.mkdir(parents=True, exist_ok=True)

# Per-dataset denoising config
DENOISE_CFG = {
    "D2": dict(threshold_scale=0.6,  num_hf_levels=2, hp_cutoff=3.0, fft_low=7.0,  fft_high=3000.0),
    "D3": dict(threshold_scale=0.7,  num_hf_levels=2, hp_cutoff=3.0, fft_low=7.0,  fft_high=3000.0),
    "D4": dict(threshold_scale=0.85, num_hf_levels=2, hp_cutoff=5.0, fft_low=7.0,  fft_high=3000.0),
    "D5": dict(threshold_scale=1.8,  num_hf_levels=3, hp_cutoff=10.0, fft_low=7.0,  fft_high=3000.0),
    "D6": dict(threshold_scale=2.2,  num_hf_levels=3, hp_cutoff=10.0, fft_low=7.0,  fft_high=3000.0),
}

PSI_TEMPLATE = None

# ----------------- Core utilities -----------------


def load_raw_trace(name: str) -> np.ndarray:
    """Load raw 'd' trace from D?.mat."""
    # Try looking in DATA_RAW first
    mat_path = DATA_RAW / f"{name}.mat"

    # If not found, try looking in the current working directory as a fallback
    if not mat_path.exists():
        mat_path = Path(f"{name}.mat")

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Could not find {name}.mat in {DATA_RAW} or current folder.")

    mat = spio.loadmat(str(mat_path), squeeze_me=True)
    d = mat["d"].astype(np.float32)
    return d


def save_denoised_trace(name: str, d_denoised: np.ndarray) -> None:
    """Save denoised trace (Useful for debugging/visualization)."""
    out_path = DATA_DENOISED / f"{name}_denoised.mat"
    spio.savemat(out_path, {"d": d_denoised.astype(np.float32)})

# ----------------- Signal Processing -----------------


def fft_bandlimit(x: np.ndarray, fs: float, f_low: Optional[float] = None, f_high: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[0]
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    mask = np.ones_like(freqs, dtype=bool)
    if f_low is not None:
        mask &= freqs >= f_low
    if f_high is not None:
        mask &= freqs <= f_high

    X_filtered = np.zeros_like(X)
    X_filtered[mask] = X[mask]
    return np.fft.irfft(X_filtered, n=N).astype(np.float32)


def wavelet_denoise(x, fs, highpass_cutoff, wavelet, level, threshold_scale, num_hf_levels, fft_low=None, fft_high=None):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.median(x)  # Robust centering

    # High-pass
    nyq = 0.5 * fs
    b, a = butter(3, highpass_cutoff / nyq, btype="high")
    x_hp = filtfilt(b, a, x)

    # Wavelet
    coeffs = pywt.wavedec(x_hp, wavelet, level=level, mode="periodization")
    n_details = len(coeffs) - 1

    # Noise est
    finest = coeffs[-1]
    sigma = np.median(np.abs(finest)) / \
        0.6745 if np.any(finest) else np.std(finest)
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(x)))

    # Thresholding
    new_coeffs = [coeffs[0]]
    for i, d in enumerate(coeffs[1:], start=1):
        if (n_details - i + 1) <= num_hf_levels:
            new_coeffs.append(pywt.threshold(d, value=uthresh, mode="soft"))
        else:
            new_coeffs.append(d)

    x_rec = pywt.waverec(new_coeffs, wavelet, mode="periodization")

    # Length matching
    if len(x_rec) > len(x):
        x_rec = x_rec[:len(x)]
    elif len(x_rec) < len(x):
        x_rec = np.pad(x_rec, (0, len(x) - len(x_rec)), mode="edge")

    # FFT
    if fft_low or fft_high:
        x_rec = fft_bandlimit(x_rec, fs, fft_low, fft_high)

    return (x_rec - np.mean(x_rec)).astype(np.float32)


def get_or_build_template():
    global PSI_TEMPLATE
    if PSI_TEMPLATE is None:
        print("Building matched filter template from D1...")
        # FIX: Use absolute path to D1
        d1_norm, spike_idx, _ = load_D1(DATA_RAW / "D1.mat")
        psi, _, _ = build_average_spike_template(d1_norm, spike_idx)
        PSI_TEMPLATE = psi
    return PSI_TEMPLATE

# ----------------- High-level Entry Point -----------------


def denoise_dataset(name: str, wavelet: str = "db4", level: int = 5, save: bool = True, use_matched_filter: bool = True):
    print(f"\nDenoising {name}...")
    d_raw = load_raw_trace(name)

    cfg = DENOISE_CFG.get(name, {})

    # 1. Matched Filter (Optional)
    if use_matched_filter:
        psi = get_or_build_template()
        d_matched = matched_filter_enhance(d_raw, psi)
    else:
        d_matched = d_raw

    # 2. Wavelet Denoising
    d_denoised = wavelet_denoise(
        d_raw,
        fs=FS_DEFAULT,
        highpass_cutoff=cfg.get("hp_cutoff", 5.0),
        wavelet=wavelet,
        level=level,
        threshold_scale=cfg.get("threshold_scale", 0.7),
        num_hf_levels=cfg.get("num_hf_levels", 2),
        fft_low=cfg.get("fft_low", None),
        fft_high=cfg.get("fft_high", None),
    )

    # 3. Save ONLY if requested (for debug)
    if save:
        save_denoised_trace(name, d_denoised)

    # 4. ALWAYS return the array (for the pipeline)
    return d_raw, d_denoised
