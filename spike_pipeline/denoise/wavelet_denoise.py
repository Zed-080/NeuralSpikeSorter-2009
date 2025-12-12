# ==============================================================================
# WAVELET DENOISING PIPELINE
# ==============================================================================
# A sophisticated denoising chain combining Frequency Filtering, Wavelet Thresholding,
# and Matched Filtering.
#
# 1. PER-DATASET TUNING (DENOISE_CFG)
#    - Noise characteristics differ across datasets (D2 vs D6).
#    - We define specific parameters (threshold scales, cutoff freqs) for each
#      dataset key ('D2', 'D3', etc.) to optimize performance.
#
# 2. PROCESSING STAGES
#    a. High-Pass Filter: Removes low-frequency drift (Local Field Potentials).
#    b. Wavelet Decomposition: Breaks signal into detailed frequency bands using
#       the 'sym4' or 'db4' wavelet basis.
#    c. Thresholding: Zeros out small coefficients (assumed to be noise) while
#       preserving large coefficients (assumed to be spike features).
#    d. FFT Bandlimit: Hard frequency clip to remove high-frequency hiss.
#
# ==============================================================================

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import scipy.io as spio
import pywt
from scipy.signal import butter, filtfilt
from spike_pipeline.denoise.matched_filter import build_average_spike_template, matched_filter_enhance
from spike_pipeline.data_loader.load_datasets import load_D1

# ----------------- Constants & paths -----------------

FS_DEFAULT = 24000  # Sampling Frequency (Hz)

# Robustly find root (2 levels up from denoise)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data"
DATA_DENOISED = PROJECT_ROOT / "outputs" / "denoised"
DATA_DENOISED.mkdir(parents=True, exist_ok=True)

# Tuned parameters for each noise level
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
    """
    Loads raw 'd' vector from a .mat file in the data directory.
    """
    # Try looking in DATA_RAW first
    mat_path = DATA_RAW / f"{name}.mat"

    # Fallback to current directory
    if not mat_path.exists():
        mat_path = Path(f"{name}.mat")

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Could not find {name}.mat in {DATA_RAW} or current folder.")

    mat = spio.loadmat(str(mat_path), squeeze_me=True)
    return mat["d"].astype(np.float32)


def save_denoised_trace(name: str, d_denoised: np.ndarray) -> None:
    """
    Saves the processed signal to 'outputs/denoised/' for debugging/visualization.
    """
    out_path = DATA_DENOISED / f"{name}_denoised.mat"
    spio.savemat(out_path, {"d": d_denoised.astype(np.float32)})

# ----------------- Signal Processing -----------------


def fft_bandlimit(x: np.ndarray, fs: float, f_low: Optional[float] = None, f_high: Optional[float] = None) -> np.ndarray:
    """
    Applies a hard brick-wall filter in the frequency domain using FFT.
    Zeroes out all frequency components outside [f_low, f_high].
    """
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
    """
    Performs Stationary Wavelet Transform (SWT) based denoising.

    Strategy:
    1. High-pass filter to remove LFP drift.
    2. Decompose signal into wavelet coefficients.
    3. Estimate noise sigma from the finest detail coefficients (Median Absolute Deviation).
    4. Soft-threshold detail coefficients to suppress noise.
    5. Reconstruct signal.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - np.median(x)  # Robust centering

    # 1. High-pass Filter
    nyq = 0.5 * fs
    b, a = butter(3, highpass_cutoff / nyq, btype="high")
    x_hp = filtfilt(b, a, x)

    # 2. Wavelet Decomposition
    coeffs = pywt.wavedec(x_hp, wavelet, level=level, mode="periodization")
    n_details = len(coeffs) - 1

    # 3. Noise Estimation (MAD)
    finest = coeffs[-1]
    if np.any(finest):
        sigma = np.median(np.abs(finest)) / 0.6745
    else:
        sigma = np.std(finest)

    # Universal Threshold with scaling factor
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(x)))

    # 4. Thresholding
    new_coeffs = [coeffs[0]]  # Keep approximation coefficients
    for i, d in enumerate(coeffs[1:], start=1):
        # Apply threshold only to high-frequency details
        if (n_details - i + 1) <= num_hf_levels:
            new_coeffs.append(pywt.threshold(d, value=uthresh, mode="soft"))
        else:
            new_coeffs.append(d)

    # 5. Reconstruction
    x_rec = pywt.waverec(new_coeffs, wavelet, mode="periodization")

    # Length matching (padding/trimming edges)
    if len(x_rec) > len(x):
        x_rec = x_rec[:len(x)]
    elif len(x_rec) < len(x):
        x_rec = np.pad(x_rec, (0, len(x) - len(x_rec)), mode="edge")

    # 6. Optional FFT Cleanup
    if fft_low or fft_high:
        x_rec = fft_bandlimit(x_rec, fs, fft_low, fft_high)

    return (x_rec - np.mean(x_rec)).astype(np.float32)


def get_or_build_template():
    """
    Singleton accessor for the D1 spike template.
    Loads D1 and builds the template if it hasn't been built yet.
    """
    global PSI_TEMPLATE
    if PSI_TEMPLATE is None:
        print("Building matched filter template from D1...")
        d1_norm, spike_idx, _ = load_D1(DATA_RAW / "D1.mat")
        psi, _, _ = build_average_spike_template(d1_norm, spike_idx)
        PSI_TEMPLATE = psi
    return PSI_TEMPLATE

# ----------------- High-level Entry Point -----------------


def denoise_dataset(name: str, wavelet: str = "db4", level: int = 5, save: bool = True, use_matched_filter: bool = True):
    """
    Main entry point to process a specific dataset (e.g., 'D2').
    Applies the specific config for that dataset and returns the clean signal.
    """
    print(f"\nDenoising {name}...")
    d_raw = load_raw_trace(name)

    cfg = DENOISE_CFG.get(name, {})

    # 1. Wavelet Denoising
    d_denoised = wavelet_denoise(
        d_raw,  # Pass the matched-filter output into the wavelet denoiser
        fs=FS_DEFAULT,
        highpass_cutoff=cfg.get("hp_cutoff", 5.0),
        wavelet=wavelet,
        level=level,
        threshold_scale=cfg.get("threshold_scale", 0.7),
        num_hf_levels=cfg.get("num_hf_levels", 2),
        fft_low=cfg.get("fft_low", None),
        fft_high=cfg.get("fft_high", None),
    )

    if save:
        save_denoised_trace(name, d_denoised)

    return d_raw, d_denoised
