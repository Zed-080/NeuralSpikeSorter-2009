#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as spio
from pathlib import Path

# Import your NEW degradation logic
from spike_pipeline.utils.degradation import degrade_with_spectral_noise

#fmt: on 

# Path config
DATA_RAW = Path("./")  # Ensure this points to your .mat files


def load_data(name):
    path = DATA_RAW / f"{name}.mat"
    mat = spio.loadmat(str(path), squeeze_me=True)
    return mat['d'].astype(np.float32)


def compute_psd(x, fs=25000):
    f, Pxx = signal.welch(x, fs, nperseg=1024)
    return f, Pxx


def main():
    print("Loading Data...")
    try:
        d1 = load_data("D1")
        d3 = load_data("D3")  # D3 is a good 'noisy' target
    except FileNotFoundError:
        print("Error: D1.mat or D3.mat not found in ../../")
        return

    # Apply Degradation
    print("Applying Noise Matching (D1 -> D3)...")
    d1_as_d3 = degrade_with_spectral_noise(d1, d3, noise_scale=1.0)

    # --- Plotting ---
    print("Plotting...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Time Domain (Zoom in on a snippet)
    start = 100000
    end = 105000
    t = np.arange(start, end) / 25000 * 1000  # ms

    ax0 = axes[0]
    ax0.plot(t, d1[start:end], label='Clean D1 (Original)',
             alpha=0.6, linewidth=1)
    ax0.plot(t, d3[start:end], label='Real D3 (Target)',
             alpha=0.6, linewidth=1)
    ax0.plot(t, d1_as_d3[start:end], label='D1 Matched to D3 (Result)',
             color='k', linestyle='--', alpha=0.8, linewidth=1)

    ax0.set_title("Time Domain Comparison (Snippet)")
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel("Amplitude")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # 2. Frequency Domain (PSD)
    f_d1, p_d1 = compute_psd(d1)
    f_d3, p_d3 = compute_psd(d3)
    f_sim, p_sim = compute_psd(d1_as_d3)

    ax1 = axes[1]
    ax1.semilogy(f_d1, p_d1, label='Clean D1', alpha=0.5)
    ax1.semilogy(f_d3, p_d3, label='Real D3 (Target Noise)', alpha=0.7)
    ax1.semilogy(f_sim, p_sim, label='D1 Matched to D3',
                 color='k', linestyle='--')

    ax1.set_title("Power Spectral Density (PSD)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power/Frequency (dB/Hz)")
    ax1.set_xlim(0, 5000)  # Focus on 0-5kHz range
    ax1.set_ylim(1e-6, 1e2)
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
