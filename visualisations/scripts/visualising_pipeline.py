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
import scipy.io as spio
from scipy.signal import welch
from pathlib import Path

# --- IMPORTS FROM YOUR PIPELINE ---
from spike_pipeline.data_loader.load_datasets import load_D1, load_unlabelled
from spike_pipeline.utils.degradation import degrade_with_spectral_noise
from spike_pipeline.denoise.matched_filter import build_average_spike_template, matched_filter_enhance

#fmt: on 

# Output folder
OUTPUT_DIR = Path("visualisations/generated_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(x):
    """Matches the exact normalization used in your pipeline"""
    return (x - x.mean()) / (x.std() + 1e-8)


def plot_degradation_showcase():
    print("Generating 1. Degradation Showcase...")
    d1, _, _ = load_D1("D1.mat")
    t = np.arange(0, 3000)  # First 3000 samples

    fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)

    # 1. Clean D1
    axes[0].plot(t, d1[t], color='black', linewidth=1)
    axes[0].set_title("Original Clean D1 (Ground Truth Source)",
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Amplitude")

    datasets = ["D2", "D3", "D4", "D5", "D6"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, name in enumerate(datasets):
        try:
            # Load real noise reference
            d_real = load_unlabelled(f"{name}.mat")

            # Create synthetic version (What your model trains on)
            d_syn = degrade_with_spectral_noise(d1, d_real, noise_scale=1.0)

            ax = axes[i+1]

            # Plot Synthetic version
            ax.plot(t, d_syn[t], color=colors[i], linewidth=0.8,
                    alpha=0.9, label=f'Synthetic {name} (Training Data)')

            # Optional: Plot Real D2-D6 snippet behind it for comparison?
            # (Real data might not align, so we just show the style)

            ax.set_title(
                f"D1 Degraded to match {name}", fontsize=10, fontweight='bold')
            ax.legend(loc="upper right")
            ax.set_ylabel("Amplitude")
        except Exception as e:
            print(f"Skipping {name}: {e}")

    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_degradation_showcase.png")
    plt.close()
    print("Saved 1_degradation_showcase.png")


def plot_pipeline_steps():
    print("Generating 2. Pipeline Step-by-Step...")
    d1, idx, _ = load_D1("D1.mat")

    # Use D5 noise for a dramatic example of the pipeline working
    d5 = load_unlabelled("D5.mat")
    d_noisy = degrade_with_spectral_noise(d1, d5, noise_scale=1.0)

    # 1. Build Filter
    psi, _, _ = build_average_spike_template(d1, idx)

    # 2. Apply Filter (Unnormalized)
    d_matched = matched_filter_enhance(d_noisy, psi)

    # 3. Normalize
    d_final = normalize(d_matched)

    # Pick a specific window with a known spike
    # Let's find a spike index around sample 10000-20000
    spike_loc = idx[(idx > 10000) & (idx < 20000)][0]
    window = slice(spike_loc - 200, spike_loc + 200)
    t_win = np.arange(window.start, window.stop)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Plot 1: Ground Truth
    axes[0].plot(t_win, d1[window], color='green', linewidth=1.5)
    axes[0].set_title("Step 1: Ground Truth (Clean Spike)", fontweight='bold')

    # Plot 2: Noisy Input
    axes[1].plot(t_win, d_noisy[window], color='red', linewidth=1, alpha=0.8)
    axes[1].set_title(
        "Step 2: Raw Noisy Input (Simulating D5)", fontweight='bold')

    # Plot 3: Matched Filter Output
    axes[2].plot(t_win, d_matched[window], color='blue', linewidth=1.5)
    axes[2].set_title(
        f"Step 3: Matched Filter Output (Note Amplitude: ~{d_matched[window].max():.1f})", fontweight='bold', color='blue')
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)

    # Plot 4: Normalized Input
    axes[3].plot(t_win, d_final[window], color='purple', linewidth=1.5)
    axes[3].set_title(f"Step 4: Normalized Input (Std=1) -> Goes to CNN",
                      fontweight='bold', color='purple')
    axes[3].axhline(0.7, color='orange', linestyle='--',
                    label="Detection Threshold (0.7)")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_pipeline_steps.png")
    plt.close()
    print("Saved 2_pipeline_steps.png")


def plot_psd_analysis():
    print("Generating 3. PSD Analysis...")
    d1, _, _ = load_D1("D1.mat")
    d5 = load_unlabelled("D5.mat")

    # Create noisy signal
    d_noisy = degrade_with_spectral_noise(d1, d5, noise_scale=1.0)

    # Process
    psi, _, _ = build_average_spike_template(d1, _)  # dummy index
    d_matched = matched_filter_enhance(d_noisy, psi)

    fs = 25000

    # Compute PSD
    f_raw, p_raw = welch(d_noisy, fs, nperseg=1024)
    f_filt, p_filt = welch(d_matched, fs, nperseg=1024)

    plt.figure(figsize=(10, 6))
    # [Image of power spectral density plot]
    plt.semilogy(f_raw, p_raw, label='Raw Noisy Signal (D5-like)',
                 color='red', alpha=0.6)
    plt.semilogy(f_filt, p_filt, label='Matched Filtered Output',
                 color='blue', linewidth=2)

    plt.title(
        "Frequency Analysis: How Matched Filtering suppresses noise", fontsize=14)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlim(0, 5000)  # Focus on spike frequencies

    plt.savefig(OUTPUT_DIR / "3_psd_analysis.png")
    plt.close()
    print("Saved 3_psd_analysis.png")


def plot_classified_spikes():
    print("Generating 4. Detected Spike Shapes...")

    datasets = ["D2", "D3", "D4", "D5", "D6"]

    for name in datasets:
        # Load raw signal (We visualize raw, or you can visualize filtered if you prefer)
        # Usually visualizing filtered is cleaner for the report.
        try:
            d_raw = load_unlabelled(f"{name}.mat")

            # Re-apply filter just for visualization consistency
            d1, idx, _ = load_D1("D1.mat")
            psi, _, _ = build_average_spike_template(d1, idx)
            d_filtered = matched_filter_enhance(d_raw, psi)
            d_viz = normalize(d_filtered)

            # Load predictions
            pred_path = Path(f"outputs/predictions/{name}.mat")
            if not pred_path.exists():
                continue

            mat = spio.loadmat(pred_path, squeeze_me=True)
            indices = mat['Index']
            classes = mat['Class']

            if indices.size == 0:
                print(f"No spikes in {name} to plot.")
                continue

            # Dictionary to hold waveforms
            waveforms = {1: [], 2: [], 3: [], 4: [], 5: []}

            for idx_spike, cls in zip(indices, classes):
                # Extract window around spike
                start = idx_spike - 32
                end = idx_spike + 32
                if start >= 0 and end < len(d_viz):
                    waveforms[cls].append(d_viz[start:end])

            # Plot
            fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
            fig.suptitle(
                f"Detected Spike Averages (Filtered) - {name}", fontsize=16)

            for cls in range(1, 6):
                waves = np.array(waveforms[cls])
                ax = axes[cls-1]

                if len(waves) > 0:
                    mean_wave = np.mean(waves, axis=0)
                    std_wave = np.std(waves, axis=0)
                    t = np.arange(len(mean_wave))

                    # Plot all individual waves (faint) if not too many
                    # if len(waves) < 100:
                    #     for w in waves:
                    #         ax.plot(t, w, color='gray', alpha=0.1)

                    ax.plot(t, mean_wave,
                            color=f'C{cls}', linewidth=2, label=f'Class {cls}')
                    ax.fill_between(t, mean_wave-std_wave, mean_wave +
                                    std_wave, color=f'C{cls}', alpha=0.2)
                    ax.set_title(f"Class {cls}\nCount: {len(waves)}")
                else:
                    ax.set_title(f"Class {cls}\n(No detections)")

                ax.grid(True, alpha=0.3)
                if cls == 1:
                    ax.set_ylabel("Amplitude (Std)")

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"4_shapes_{name}.png")
            plt.close()
            print(f"Saved 4_shapes_{name}.png")

        except Exception as e:
            print(f"Error plotting {name}: {e}")


if __name__ == "__main__":
    plot_degradation_showcase()
    plot_pipeline_steps()
    plot_psd_analysis()
    plot_classified_spikes()
    print(f"\nDONE! All plots saved to: {OUTPUT_DIR.resolve()}")
