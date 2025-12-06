"""
compare_noise_methods.py

Compare the ORIGINAL spikepipeline synthetic noise degradation
vs the REAL noise_matcher noise matching.

This script:
  - Loads D1
  - Extracts random spike windows
  - Applies:
        (A) Original degrade_with_spectral_noise (synthetic PSD noise)
        (B) New real noise_match_d1 (dataset-accurate matching)
  - Compares:
        • Waveforms
        • Power Spectral Density (PSD)
        • Amplitude histograms
        • Noise std
        • SNR estimates

Run from project root:

    (.venv_tf) python experiments/compare_noise_methods.py
"""
#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------
# ---------------- IMPORTS ----------------

from spike_pipeline.utils.degradation import degrade_with_spectral_noise
from spike_pipeline.data_loader.load_datasets import load_D1
from spike_pipeline.utils.comp_noise_matcher import noise_match_d1
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

#fmt: on

# ----------- YOUR PIPELINE IMPORTS ----------


# 👇 change if your original function lives somewhere else

# ---------------- CONFIG ----------------

TARGET = "D3"     # which dataset to match against
WINDOW = 128      # spike window length
NUM_SAMPLES = 100
FS = 24000

# ---------------- HELPERS ----------------


def estimate_noise_std(x, thresh=3.0):
    mask = np.abs(x) < thresh
    if mask.sum() < 10:
        return np.std(x)
    return np.std(x[mask])


def compute_snr(x):
    signal = np.max(np.abs(x))
    noise = estimate_noise_std(x)
    if noise < 1e-12:
        return 0.0
    return 20 * np.log10(signal / noise)


# ---------------- MAIN ----------------

def main():

    print("Loading D1 dataset...")
    d1, spike_idx, _ = load_D1('D1.mat')

    # Load full real-matched trace once
    print(f"Generating real noise-matched D1 → {TARGET} ...")
    d1_real = noise_match_d1(TARGET)

    snrs_fake = []
    snrs_real = []

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Original Synthetic vs Real Noise Matching", fontsize=14)

    for i in range(NUM_SAMPLES):

        # random spike window from D1
        idx = np.random.choice(spike_idx)
        start = max(0, idx - WINDOW//2)
        end = start + WINDOW

        clean = d1[start:end]

        # original synthetic degradation
        fake_noisy = degrade_with_spectral_noise(clean, noise_ref=d1_real)

        # real noise matching: we just sample snippet from d1_real
        real_noisy = d1_real[start:end]

        # estimate stats
        snrs_fake.append(compute_snr(fake_noisy))
        snrs_real.append(compute_snr(real_noisy))

        # plot first 3 snippet sets
        if i < 3:
            t = np.arange(WINDOW) / FS * 1000

            axs[i, 0].plot(t, clean)
            axs[i, 0].set_title("Clean D1")
            axs[i, 1].plot(t, fake_noisy)
            axs[i, 1].set_title("Original synthetic noise")
            axs[i, 2].plot(t, real_noisy)
            axs[i, 2].set_title("Real matched noise")

    # ---------- PSD COMPARISON ----------
    f_fake, p_fake = welch(np.concatenate(snrs_fake), fs=FS)
    f_real, p_real = welch(np.concatenate(snrs_real), fs=FS)

    plt.figure(figsize=(7, 5))
    plt.semilogy(f_fake, p_fake, label="Synthetic noise")
    plt.semilogy(f_real, p_real, label="Real matched noise")
    plt.title("PSD comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()

    # ---------- HISTOGRAM ----------
    plt.figure(figsize=(7, 5))
    plt.hist(snrs_fake, bins=25, alpha=0.6, label="Synthetic noise")
    plt.hist(snrs_real, bins=25, alpha=0.6, label="Real matched noise")
    plt.title("Spike Window SNR Distributions")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Count")
    plt.legend()

    # ---------- SUMMARY ----------
    print("\n===== SNR SUMMARY =====")
    print(
        f"Synthetic noise: mean={np.mean(snrs_fake):.2f} dB   std={np.std(snrs_fake):.2f}")
    print(
        f"Real noise:      mean={np.mean(snrs_real):.2f} dB   std={np.std(snrs_real):.2f}")

    plt.show()


if __name__ == "__main__":
    main()
