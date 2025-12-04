#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------

import scipy.io as spio
from spike_pipeline.utils.signal_tools import bandpass_filter,  mad, build_average_mother, matched_filter_denoise

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from spike_pipeline.data_loader.load_datasets import load_D1, load_unlabelled, load_D1_stages, load_unlabelled_stages
from spike_pipeline.utils.signal_tools import build_average_mother, matched_filter_denoise
#fmt:on


def plot_preprocessing_stages(stages_dict, dataset_name, n_samples=5000, fs=25000):
    """
    Plot time-domain signals for all stages stacked vertically.
    """
    # Define the stages to plot (in order)
    stage_order = [
        ('raw',      '1. Raw Signal'),
        ('bandpass', '2. FFT + Bandpass (Standard Approach)'),
        ('wavelet',  '3. FFT + Matched Filter (Wavelet Approach)')
    ]

    # Filter out None stages
    valid_stages = [(key, label)
                    for key, label in stage_order if stages_dict.get(key) is not None]

    n_plots = len(valid_stages)
    fig, axes = plt.subplots(n_plots, 1, figsize=(
        15, 2.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    time = np.arange(n_samples) / fs
    colors = ['#333333', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (ax, (stage_key, stage_label)) in enumerate(zip(axes, valid_stages)):
        signal = stages_dict[stage_key]

        # Plot signal
        ax.plot(time, signal[:n_samples], linewidth=0.8,
                color=colors[idx % len(colors)])

        # Styling
        ax.set_title(f"{dataset_name}: {stage_label}",
                     fontsize=10, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.2)
        ax.set_ylabel('Amp (mV)', fontsize=8)

        # Annotation: Standard Deviation (Proxy for noise level)
        std_val = np.std(signal)
        ax.text(1.0, 1.05, f'σ = {std_val:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, fontweight='bold', color='#444')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_psd_comparison(stages_dict, dataset_name, fs=25000):
    """
    Plot Power Spectral Density (PSD) overlay.
    """
    plt.figure(figsize=(15, 6))

    # YOUR CUSTOM COMPARISON CONFIGURATION
    stage_configs = [
        # (Key in Dict,      Legend Label,              Color,     Alpha)
        ('raw',              '1. Raw Signal',           'black',   0.3),
        ('bandpass',         '2. FFT + Bandpass',       'orange',  0.8),
        ('wavelet',          '3. FFT + Matched Filter', 'blue',    0.8)
    ]

    for key, label, color, alpha in stage_configs:
        if stages_dict.get(key) is None:
            continue

        signal = stages_dict[key]
        freqs, psd = welch(signal, fs, nperseg=1024)
        plt.semilogy(freqs, psd, label=label, color=color,
                     alpha=alpha, linewidth=1.5)

    plt.title(
        f"{dataset_name} - Frequency Comparison: Bandpass vs. Matched Filter", fontsize=14)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.xlim(0, 5000)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='upper right')
    plt.show()


def main():
    print("=" * 60)
    print("PREPROCESSING & SPECTRUM VISUALIZATION")
    print("=" * 60)

    # 1. Setup D1 Template
    print("Building mother wavelet from D1...")
    d1_stages, d1_idx, _ = load_D1_stages("D1.mat", fs=25000)

    # We build the wavelet from the 'fft_only' stage to ensure
    # the template captures the 'true' spike shape before bandpassing alters it
    # psi, _, _ = build_average_mother(d1_stages['fft_only'], d1_idx)
    psi, _, _ = build_average_mother(d1_stages['bandpass'], d1_idx)

    if psi is None:
        print("Error: Could not build mother wavelet. Check D1.mat path.")
        return

    # 2. Loop through datasets
    # datasets = ["D1", "D2", "D3", "D4", "D5", "D6"]
    datasets = ["D3", "D4", "D5", "D6"]

    for name in datasets:
        print(f"\nProcessing {name}...")
        path = f"{name}.mat"

        # Load stages
        if name == "D1":
            stages, _, _ = load_D1_stages(path, fs=25000)
        else:
            stages = load_unlabelled_stages(path, psi=psi, fs=25000)

        # Plot Time Domain
        plot_preprocessing_stages(stages, name, n_samples=4000)

        # Plot Frequency Domain
        plot_psd_comparison(stages, name)


if __name__ == "__main__":
    main()


# def plot_signal_comparison(signals_dict, n_samples=5000, fs=25000, figsize=(14, 10)):
#     """
#     Plot multiple signal processing stages in separate subplots.

#     Args:
#         signals_dict: dict like {'Raw': signal1, 'Filtered': signal2, ...}
#         n_samples: number of samples to display
#         fs: sampling frequency
#         figsize: figure size tuple
#     """
#     n_signals = len(signals_dict)
#     fig, axes = plt.subplots(n_signals, 1, figsize=figsize)

#     if n_signals == 1:
#         axes = [axes]

#     time = np.arange(n_samples) / fs

#     for ax, (label, signal) in zip(axes, signals_dict.items()):
#         ax.plot(time, signal[:n_samples], linewidth=0.7)
#         ax.set_title(label, fontsize=12, fontweight='bold')
#         ax.set_ylabel('Amplitude (mV)', fontsize=10)
#         ax.grid(True, alpha=0.3)
#         ax.set_xlim(time[0], time[-1])

#     axes[-1].set_xlabel('Time (s)', fontsize=10)
#     plt.tight_layout()
#     plt.show()


# def plot_matched_filter_effect(original, filtered, dataset_name, n_samples=5000, fs=25000):
#     """
#     Specific comparison for matched filter denoising.

#     Args:
#         original: signal before matched filter
#         filtered: signal after matched filter
#         dataset_name: name for title (e.g., "D6")
#         n_samples: samples to display
#         fs: sampling frequency
#     """
#     signals = {
#         f'{dataset_name} - Original (bandpass only)': original,
#         f'{dataset_name} - Matched Filter Applied': filtered
#     }
#     plot_signal_comparison(signals, n_samples=n_samples,
#                            fs=fs, figsize=(14, 6))


# # ---------- new extra code for  -----------


# # ----------- origional code for loading datasets from .mat files -----------
# def load_mat(path):
#     """
#     Loads a .mat file and extracts:
#       - d      (raw signal)
#       - Index  (optional)
#       - Class  (optional)

#     Always returns:
#       d, Index, Class   (Index/Class = None if missing)
#     """
#     mat = spio.loadmat(path, squeeze_me=True)

#     d = mat["d"].astype(np.float32)

#     Index = mat.get("Index", None)
#     Class = mat.get("Class", None)

#     # ensure Index / Class are always 1D arrays if present
#     if Index is not None:
#         Index = np.array(Index, dtype=np.int64).reshape(-1)
#     if Class is not None:
#         Class = np.array(Class, dtype=np.int64).reshape(-1)

#     return d, Index, Class


# def global_normalize(d):
#     """
#     Global z-score normalization:
#         (d - mean) / std
#     """
#     mu = np.mean(d)
#     sigma = np.std(d) + 1e-8   # avoid divide-by-zero

#     return (d - mu) / sigma

# # Remove DC drift (very low frequencies)


# def remove_dc_fft(x, fs=25000, cutoff_hz=1):
#     N = len(x)
#     X = np.fft.rfft(x)
#     freqs = np.fft.rfftfreq(N, d=1/fs)
#     X[freqs < cutoff_hz] = 0
#     return np.fft.irfft(X, n=N)


# # def load_D1(path, fs=25000):
# #     """
# #     Loads D1 including Index and Class,
# #     and returns normalized signal + labels.
# #     """
# #     d, Index, Class = load_mat(path)

# #     # 1. optional: remove DC drift
# #     d = remove_dc_fft(d, fs=fs)

# #     # 2. band-pass filter 300–3000 Hz
# #     d = bandpass_filter(d, fs=fs, low=7, high=3000)

# #     # 3. global z-score normalization
# #     d_norm = global_normalize(d)

# #     return d_norm, Index, Class

# def load_D1(path, fs=25000, use_matched_filter=False):
#     """
#     Loads D1 including Index and Class,
#     and returns normalized signal + labels.

#     Args:
#         use_matched_filter: If True, applies matched filter after bandpass
#     """
#     d, Index, Class = load_mat(path)

#     # 1. remove DC drift
#     d = remove_dc_fft(d, fs=fs)

#     # 2. band-pass filter 300–3000 Hz
#     d = bandpass_filter(d, fs=fs, low=7, high=3000)

#     # 3. OPTIONAL: matched filter denoising (NEW)
#     if use_matched_filter:
#         psi, _, _ = build_average_mother(d, Index)
#         if psi is not None:
#             d = matched_filter_denoise(d, psi)
#             print("  Applied matched filter denoising to D1")

#     # 4. global z-score normalization
#     d_norm = global_normalize(d)

#     return d_norm, Index, Class


# # def load_unlabelled(path, fs=25000):
# #     """
# #     Loads D2-D6 datasets which have only raw signal.
# #     Returns normalized signal only.
# #     """
# #     d, _, _ = load_mat(path)
# #     d = remove_dc_fft(d, fs=fs)
# #     d = bandpass_filter(d, fs=fs, low=300, high=3000)

# #     d_norm = global_normalize(d)
# #     return d_norm

# def plot_preprocessing_stages(stages_dict, dataset_name, n_samples=5000, fs=25000, figsize=(14, 12)):
#     """
#     Plot all preprocessing stages for a dataset.

#     Args:
#         stages_dict: dict from load_D1_stages or load_unlabelled_stages
#         dataset_name: name for titles
#         n_samples: samples to display
#         fs: sampling frequency
#     """
#     # Define the stages to plot (in order)
#     stage_order = [
#         ('raw', 'Raw Signal'),
#         ('fft_only', 'Raw + FFT Drift Removal'),
#         ('bandpass', 'Raw + FFT + Bandpass (7-3000 Hz)'),
#         ('wavelet', 'Raw + FFT + Matched Filter'),
#         ('full_pipeline', 'Raw + FFT + Bandpass + Matched Filter')
#     ]

#     # Filter out None stages
#     valid_stages = [(key, label)
#                     for key, label in stage_order if stages_dict.get(key) is not None]

#     n_plots = len(valid_stages)
#     fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

#     if n_plots == 1:
#         axes = [axes]

#     time = np.arange(n_samples) / fs

#     colors = ['black', 'darkblue', 'green', 'orange', 'red']

#     for idx, (ax, (stage_key, stage_label)) in enumerate(zip(axes, valid_stages)):
#         signal = stages_dict[stage_key]

#         ax.plot(time, signal[:n_samples], linewidth=0.7,
#                 color=colors[idx % len(colors)])
#         ax.set_title(f'{dataset_name} - {stage_label}',
#                      fontsize=11, fontweight='bold')
#         ax.set_ylabel('Amplitude', fontsize=9)
#         ax.grid(True, alpha=0.3)
#         ax.set_xlim(time[0], time[-1])

#         # Add std annotation
#         std_val = np.std(signal[:n_samples])
#         ax.text(0.98, 0.95, f'σ = {std_val:.3f}',
#                 transform=ax.transAxes,
#                 ha='right', va='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
#                 fontsize=8)

#     axes[-1].set_xlabel('Time (s)', fontsize=10)
#     plt.tight_layout()
#     plt.show()


# def load_unlabelled(path, fs=25000, psi=None):
#     """
#     Loads D2-D6 datasets which have only raw signal.

#     Args:
#         path: path to .mat file
#         fs: sampling frequency
#         psi: optional mother wavelet for matched filtering

#     Returns:
#         d_norm: normalized signal
#     """
#     d, _, _ = load_mat(path)
#     d = remove_dc_fft(d, fs=fs)
#     d = bandpass_filter(d, fs=fs, low=7, high=3000)

#     # Apply matched filter if wavelet provided
#     if psi is not None:
#         d = matched_filter_denoise(d, psi)
#         print(f"  Applied matched filter to {path}")

#     d_norm = global_normalize(d)
#     return d_norm


# # def main():
# #     """Visualize matched filter effects on different datasets"""

# #     # ===========================================
# #     # 1. Load D1 and build the mother wavelet
# #     # ===========================================
# #     print("=" * 60)
# #     print("MATCHED FILTER VISUALIZATION")
# #     print("=" * 60)
# #     print("\nLoading D1 and building mother wavelet...")
# #     d1_clean, d1_idx, d1_class = load_D1("D1.mat", use_matched_filter=False)

# #     # Build the matched filter wavelet from D1
# #     psi, avg_spike, all_windows = build_average_mother(d1_clean, d1_idx)
# #     print(f"✓ Mother wavelet built from {len(all_windows)} spikes")

# #     # ===========================================
# #     # 2. Test on multiple datasets
# #     # ===========================================
# #     datasets = ["D2", "D3", "D4", "D5", "D6"]

# #     for dataset_name in datasets:
# #         print("\n" + "-" * 60)
# #         print(f"Processing {dataset_name}...")
# #         print("-" * 60)

# #         filepath = f"{dataset_name}.mat"

# #         # Load without matched filter
# #         print(f"  Loading {dataset_name} (bandpass only)...")
# #         signal_bandpass = load_unlabelled(filepath, psi=None)

# #         # Load with matched filter
# #         print(f"  Loading {dataset_name} (with matched filter)...")
# #         signal_filtered = load_unlabelled(filepath, psi=psi)

# #         # Visualize
# #         print(f"  Generating comparison plot...")
# #         plot_matched_filter_effect(
# #             original=signal_bandpass,
# #             filtered=signal_filtered,
# #             dataset_name=dataset_name,
# #             n_samples=5000
# #         )

# #     print("\n" + "=" * 60)
# #     print("✅ VISUALIZATION COMPLETE")
# #     print("=" * 60)

# def main():
#     """Visualize all preprocessing stages for each dataset"""

#     print("=" * 60)
#     print("PREPROCESSING PIPELINE VISUALIZATION")
#     print("=" * 60)

#     # ===========================================
#     # 1. Load D1 and build mother wavelet
#     # ===========================================
#     print("\nLoading D1 (all stages)...")
#     d1_stages, d1_idx, d1_class = load_D1_stages("D1.mat")

#     # Build wavelet from FFT-only stage for consistency
#     print("Building mother wavelet from D1...")
#     psi, avg_spike, all_windows = build_average_mother(
#         d1_stages['fft_only'], d1_idx)
#     print(f"✓ Mother wavelet built from {len(all_windows)} spikes\n")

#     # Plot D1 stages
#     print("Visualizing D1 preprocessing stages...")
#     plot_preprocessing_stages(d1_stages, "D1 (Training Set)", n_samples=5000)

#     # ===========================================
#     # 2. Process and visualize D2-D6
#     # ===========================================
#     datasets = ["D2", "D3", "D4", "D5", "D6"]

#     for dataset_name in datasets:
#         print("\n" + "-" * 60)
#         print(f"Processing {dataset_name}...")
#         print("-" * 60)

#         filepath = f"{dataset_name}.mat"

#         # Load all preprocessing stages
#         stages = load_unlabelled_stages(filepath, psi=psi)

#         # Visualize
#         print(f"Visualizing {dataset_name} preprocessing stages...")
#         plot_preprocessing_stages(stages, dataset_name, n_samples=5000)

#     print("\n" + "=" * 60)
#     print("✅ VISUALIZATION COMPLETE")
#     print("=" * 60)
#     print("\nYou should see 5 subplots per dataset showing:")
#     print("  1. Raw signal")
#     print("  2. Raw + FFT drift removal")
#     print("  3. Raw + FFT + Bandpass")
#     print("  4. Raw + FFT + Matched filter")
#     print("  5. Raw + FFT + Bandpass + Matched filter (FULL PIPELINE)")


# if __name__ == "__main__":
#     main()
