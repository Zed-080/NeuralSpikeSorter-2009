#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONTINUE WITH IMPORTS ---
import numpy as np
import scipy.io as spio
import pywt
import matplotlib.pyplot as plt

from spike_pipeline.utils.signal_tools import bandpass_filter   # 300–3000 Hz
from spike_pipeline.data_loader.load_datasets import load_D1    # includes zscore
# fmt: on 

# -----------------------------------------
# 1. Load raw D1
# -----------------------------------------
d_raw, Index, Class = load_D1("D1.mat")  # already z-scored
d_raw = d_raw.astype(np.float32)

# For visualization, pick a small region
start = 30000
end = start + 3000
segment = d_raw[start:end]


# -----------------------------------------
# 2. FFT-based drift removal
# -----------------------------------------
def remove_dc_fft(x):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/25000)

    # Zero-out DC and ultra-low frequencies (<1 Hz)
    X[freqs < 1] = 0
    return np.fft.irfft(X, n=N)


segment_fft = remove_dc_fft(segment)


# -----------------------------------------
# 3. Band-pass filtering (300–3000 Hz)
# -----------------------------------------
segment_band = bandpass_filter(segment, fs=25000, low=300, high=3000)


# -----------------------------------------
# 4. Wavelet denoising AFTER band-pass
# -----------------------------------------
def wavelet_denoise(x, wavelet="db4", level=3):
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    denoised = coeffs[:]
    for i in range(1, len(coeffs)):
        denoised[i] = pywt.threshold(coeffs[i], uthresh, mode="soft")

    return pywt.waverec(denoised, wavelet, mode="symmetric")


segment_wavelet = wavelet_denoise(segment_band)


# -----------------------------------------
# 5. PLOT COMPARISON
# -----------------------------------------
plt.figure(figsize=(16, 10))

plt.subplot(4, 1, 1)
plt.plot(segment)
plt.title("1. RAW (z-scored)")

plt.subplot(4, 1, 2)
plt.plot(segment_fft)
plt.title("2. FFT Drift Removed")

plt.subplot(4, 1, 3)
plt.plot(segment_band)
plt.title("3. Band-pass 300–3000 Hz")

plt.subplot(4, 1, 4)
plt.plot(segment_wavelet)
plt.title("4. Band-pass + Wavelet Denoising")

plt.tight_layout()
plt.show()
