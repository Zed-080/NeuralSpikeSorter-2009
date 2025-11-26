#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONTINUE WITH IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt

from spike_pipeline.data_loader.load_datasets import load_D1, load_unlabelled
from spike_pipeline.utils.degradation import degrade_with_spectral_noise
# fmt: on 


# ================================
# 1. Load clean and noise datasets
# ================================
d1_clean, Index, Class = load_D1("D1.mat")
d3_noise = load_unlabelled("D3.mat")

# degrade clean D1 with spectral-matched D3 noise
d1_degraded = degrade_with_spectral_noise(d1_clean, d3_noise, noise_scale=1.0)


# pick a window for visualisation
start = 200000
end = start + 2000

clean_seg = d1_clean[start:end]
degraded_seg = d1_degraded[start:end]


# ================================
# 2. Compute power spectra (FFT)
# ================================
def power_spectrum(x, fs=25000):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    power = np.abs(X)**2
    return freqs, power


freqs_clean,    power_clean = power_spectrum(clean_seg)
freqs_degraded, power_degraded = power_spectrum(degraded_seg)

# also inspect raw noise spectrum
# (the noise that was extracted from D3)
noise_piece = d3_noise[start:end]
freqs_noise, power_noise = power_spectrum(noise_piece)


# ================================
# 3. PLOTS
# ================================
plt.figure(figsize=(16, 10))

# --- Time domain ----------------
plt.subplot(2, 1, 1)
plt.plot(clean_seg, label="Clean D1")
plt.plot(degraded_seg, label="Degraded D1 (spectral noise)", alpha=0.8)
plt.title("Time Domain: Clean vs Degraded")
plt.legend()

# --- Frequency domain -----------
plt.subplot(2, 1, 2)
plt.semilogy(freqs_clean,    power_clean,    label="Clean D1")
plt.semilogy(freqs_noise,    power_noise,    label="Reference Noise (D3)")
plt.semilogy(freqs_degraded, power_degraded, label="Degraded D1")
plt.title("Power Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (log scale)")
plt.legend()

plt.tight_layout()
plt.show()
