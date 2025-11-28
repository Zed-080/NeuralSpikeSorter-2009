#fmt:off
import sys
import os

# --- FIX IMPORT PATHS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# from scipy.signal import butter, filtfilt

# # from spike_pipeline.data_loader.load_datasets import load_mat, load_D1
# # from spike_pipeline.utils.normalization import normalize_window
# # from spike_pipeline.utils.windowing import extract_window

# # ------------------------------
# #  IMPORT YOUR EXISTING UTILS
# # ------------------------------
# from spike_pipeline.data_loader.load_datasets import load_mat, global_normalize
# from spike_pipeline.utils.signal_tools import bandpass_filter
# from spike_pipeline.utils.windowing import extract_window

# # %%
# # FS = 25000  # Hz
# # PRE = 20
# # POST = 44
# # WIN_LEN = PRE + POST  # 64


# # def plot_time_segment(raw, proc, fs, start_sec, dur_sec, title_prefix="D1"):
# #     """Plot raw vs processed over a chosen time window."""
# #     start = int(start_sec * fs)
# #     end = int((start_sec + dur_sec) * fs)

# #     t = np.arange(start, end) / fs

# #     plt.figure(figsize=(10, 4))
# #     plt.plot(t, raw[start:end], label="Raw", alpha=0.7)
# #     plt.plot(t, proc[start:end], label="Processed", alpha=0.7)
# #     plt.xlabel("Time (s)")
# #     plt.ylabel("Amplitude")
# #     plt.title(f"{title_prefix}: Raw vs Processed (time domain)")
# #     plt.legend()
# #     plt.tight_layout()


# # def plot_zoom_around_spike(raw, proc, fs, spike_idx, window_ms=10, title_prefix="D1"):
# #     """Zoomed view around one spike index."""
# #     half = int(window_ms * 1e-3 * fs)
# #     start = max(spike_idx - half, 0)
# #     end = min(spike_idx + half, len(raw))

# #     t = (np.arange(start, end) - spike_idx) / fs * 1e3  # ms relative to spike

# #     plt.figure(figsize=(10, 4))
# #     plt.plot(t, raw[start:end], label="Raw", alpha=0.7)
# #     plt.plot(t, proc[start:end], label="Processed", alpha=0.7)
# #     plt.xlabel("Time relative to spike (ms)")
# #     plt.ylabel("Amplitude")
# #     plt.title(f"{title_prefix}: Zoom around spike")
# #     plt.legend()
# #     plt.tight_layout()


# # def plot_spectrum(raw, proc, fs, title_prefix="D1"):
# #     """FFT magnitude comparison raw vs processed."""
# #     N = len(raw)

# #     # one-sided FFT
# #     freqs = np.fft.rfftfreq(N, d=1 / fs)
# #     Raw = np.fft.rfft(raw)
# #     Proc = np.fft.rfft(proc)

# #     mag_raw = np.abs(Raw)
# #     mag_proc = np.abs(Proc)

# #     plt.figure(figsize=(10, 4))
# #     plt.semilogy(freqs, mag_raw + 1e-12, label="Raw")
# #     plt.semilogy(freqs, mag_proc + 1e-12, label="Processed")
# #     plt.xlim(0, 6000)  # focus on useful band
# #     plt.xlabel("Frequency (Hz)")
# #     plt.ylabel("Magnitude (log)")
# #     plt.title(f"{title_prefix}: Spectrum raw vs processed")
# #     plt.legend()
# #     plt.tight_layout()


# # def plot_window_comparison(win_global, win_perwin, title_prefix="D1"):
# #     """Compare a 64-sample window before & after per-window z-score."""
# #     x = np.arange(len(win_global))

# #     plt.figure(figsize=(8, 4))
# #     plt.plot(x, win_global, label="Global-normalised only", marker="o")
# #     plt.plot(x, win_perwin, label="Global + per-window norm", marker="x")
# #     plt.xlabel("Sample index in window")
# #     plt.ylabel("Amplitude (a.u.)")
# #     plt.title(f"{title_prefix}: 64-sample window comparison")
# #     plt.legend()
# #     plt.tight_layout()

# #     print("=== Window stats ===")
# #     print(
# #         f"Global-only   : mean={win_global.mean():.4f}, std={win_global.std():.4f}")
# #     print(
# #         f"Per-window z  : mean={win_perwin.mean():.4f}, std={win_perwin.std():.4f}")


# # def main():
# #     dataset_path = "D1.mat"  # change if needed

# #     # 1) Load raw + fully processed D1
# #     # raw signal (only DC removal done later)
# #     d_raw, Index, Class = load_mat(dataset_path)
# #     # remove_dc_fft + bandpass + global z-score
# #     d_proc, _, _ = load_D1(dataset_path)

# #     if Index is None or len(Index) == 0:
# #         raise RuntimeError("D1.mat should have Index; got none. Check file.")

# #     # pick a middle spike to visualise
# #     spike_idx = int(Index[len(Index) // 2])

# #     # 2) Whole-signal visualisations
# #     # wide view: e.g. from 5s to 5.2s (change as you like)
# #     plot_time_segment(d_raw, d_proc, FS, start_sec=5.0,
# #                       dur_sec=0.2, title_prefix="D1")

# #     # zoom around a single spike
# #     plot_zoom_around_spike(d_raw, d_proc, FS, spike_idx,
# #                            window_ms=10, title_prefix="D1")

# #     # spectrum
# #     plot_spectrum(d_raw, d_proc, FS, title_prefix="D1")

# #     # 3) Window-level comparison (64-sample waveform)
# #     # from globally normalised signal
# #     win_global = extract_window(d_proc, spike_idx, PRE, POST)
# #     if win_global is None or len(win_global) != WIN_LEN:
# #         raise RuntimeError(
# #             "Failed to extract 64-sample window; check PRE/POST.")

# #     win_perwin = normalize_window(win_global.copy())  # add per-window z-score

# #     plot_window_comparison(win_global, win_perwin, title_prefix="D1")

# #     # show all plots
# #     plt.show()


# # if __name__ == "__main__":
# #     main()
# # %%

# FS = 25000
# PRE = 20
# POST = 44
# WIN_LEN = PRE + POST


# # ============================================================
# #  YOUR CURRENT PIPELINE
# # ============================================================
# def remove_dc_fft(x, fs=FS, cutoff_hz=1):
#     N = len(x)
#     X = np.fft.rfft(x)
#     freqs = np.fft.rfftfreq(N, d=1/fs)
#     X[freqs < cutoff_hz] = 0
#     return np.fft.irfft(X, n=N)


# def pipeline_current(signal, fs=FS, low=300, high=3000):
#     """Your current filtering pipeline."""
#     sig_dc = remove_dc_fft(signal, fs)
#     sig_bp = bandpass_filter(sig_dc, fs=fs, low=low, high=high)
#     sig_norm = global_normalize(sig_bp)
#     return sig_norm


# # ============================================================
# #  WAVELET PIPELINE
# # ============================================================
# def pipeline_wavelet(signal, fs=FS):
#     nyq = 0.5 * fs

#     # 1. Gentle high-pass 10 Hz
#     b, a = butter(3, 10/nyq, btype='high')
#     sig_hp = filtfilt(b, a, signal)

#     # 2. Wavelet shrinkage
#     coeffs = pywt.wavedec(sig_hp, 'db4', level=5)
#     sigma = np.median(np.abs(coeffs[-1])) / 0.6745
#     uthresh = 0.7 * sigma * np.sqrt(2 * np.log(len(sig_hp)))

#     coeffs_thr = [pywt.threshold(c, value=uthresh, mode='soft')
#                   for c in coeffs]
#     sig_wav = pywt.waverec(coeffs_thr, 'db4')

#     sig_wav = sig_wav + 1e-4 * np.random.randn(len(sig_wav))
#     return global_normalize(sig_wav)


# # ============================================================
# #  PLOTS (clean line graphs only)
# # ============================================================
# def plot_compare(raw, curr, wave, title="D1"):
#     plt.figure(figsize=(12, 4))
#     t = np.arange(len(raw)) / FS
#     plt.plot(t, raw, label="Raw")
#     plt.plot(t, curr, label="Current")
#     plt.plot(t, wave, label="Wavelet")
#     plt.xlim(5.0, 5.2)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title(f"{title}: Whole Signal Comparison")
#     plt.legend()
#     plt.tight_layout()


# def plot_zoom(raw, curr, wave, spike_idx, title="D1"):
#     half = int(0.005 * FS)
#     start = spike_idx - half
#     end = spike_idx + half
#     t = (np.arange(start, end) - spike_idx) / FS * 1e3

#     plt.figure(figsize=(12, 4))
#     plt.plot(t, raw[start:end], label="Raw")
#     plt.plot(t, curr[start:end], label="Current")
#     plt.plot(t, wave[start:end], label="Wavelet")
#     plt.xlabel("Time (ms)")
#     plt.title(f"{title}: Zoom Around Spike")
#     plt.legend()
#     plt.tight_layout()


# def plot_spectrum(raw, curr, wave, title="D1"):
#     N = len(raw)
#     freqs = np.fft.rfftfreq(N, d=1/FS)
#     plt.figure(figsize=(12, 4))
#     plt.semilogy(freqs, np.abs(np.fft.rfft(raw)), label="Raw")
#     plt.semilogy(freqs, np.abs(np.fft.rfft(curr)), label="Current")
#     plt.semilogy(freqs, np.abs(np.fft.rfft(wave)), label="Wavelet")
#     plt.xlim(0, 6000)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude (log)")
#     plt.title(f"{title}: Spectrum Comparison")
#     plt.legend()
#     plt.tight_layout()


# def plot_window(curr_win, wave_win, title="D1"):
#     x = np.arange(len(curr_win))
#     plt.figure(figsize=(8, 3))
#     plt.plot(x, curr_win, label="Current Pipeline")
#     plt.plot(x, wave_win, label="Wavelet Pipeline")
#     plt.title(f"{title}: 64-Sample Window Comparison")
#     plt.legend()
#     plt.tight_layout()


# # ============================================================
# #  MAIN
# # ============================================================
# def main():
#     dataset = "D1"
#     path = dataset + ".mat"

#     raw, Index, Class = load_mat(path)
#     raw = raw.astype(np.float32)

#     spike_idx = int(Index[len(Index) // 2])

#     sig_curr = pipeline_current(raw, low=300, high=3000)
#     sig_wave = pipeline_wavelet(raw)

#     plot_compare(raw, sig_curr, sig_wave, title=dataset)
#     plot_zoom(raw, sig_curr, sig_wave, spike_idx, title=dataset)
#     plot_spectrum(raw, sig_curr, sig_wave, title=dataset)

#     curr_win = extract_window(sig_curr, spike_idx, PRE, POST)
#     wave_win = extract_window(sig_wave, spike_idx, PRE, POST)
#     plot_window(curr_win, wave_win, title=dataset)

#     plt.show()


# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt

from spike_pipeline.data_loader.load_datasets import load_mat
from spike_pipeline.utils.signal_tools import bandpass_filter
from spike_pipeline.utils.normalization import zscore

FS = 25000

#fmt: on
# --------------------------
# your original denoising
# --------------------------


def pipeline_original(x):
    # remove low frequencies
    x = remove_dc_fft(x)
    # band-pass 7–3000 Hz
    x = bandpass_filter(x, fs=FS, low=7, high=3000,  order=1)
    # global z-score
    return zscore(x)


def remove_dc_fft(x, fs=FS, cutoff_hz=1):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    X[freqs < cutoff_hz] = 0
    return np.fft.irfft(X, N)


# --------------------------
# super clean plot
# --------------------------
def simple_plot(raw, proc, start, end, title):
    t = np.arange(start, end) / FS

    plt.figure(figsize=(10, 3))
    plt.plot(t, raw[start:end], label="raw", linewidth=1)
    plt.plot(t, proc[start:end], label="original", linewidth=1)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------
# main loop for D1–D6
# --------------------------
def main():
    datasets = ["D1.mat", "D2.mat", "D3.mat", "D4.mat", "D5.mat", "D6.mat"]

    # region of signal to show
    start = 100_000
    end = start + 2000

    for ds in datasets:
        print(f"Processing {ds}...")
        raw, _, _ = load_mat(ds)

        proc = pipeline_original(raw)

        simple_plot(raw, proc, start, end, title=ds)


if __name__ == "__main__":
    main()
