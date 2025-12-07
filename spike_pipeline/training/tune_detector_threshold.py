# import numpy as np
# import os
# from pathlib import Path  # <--- Need this

# from spike_pipeline.data_loader.load_datasets import load_D1, load_unlabelled
# from spike_pipeline.utils.degradation import degrade_with_spectral_noise
# from spike_pipeline.inference.matching import match_predictions

# # --- Config (Robust Paths) ---
# PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to root
# DATA_RAW = PROJECT_ROOT / "data"                    # Points to /data
# OUTPUT_DIR = PROJECT_ROOT / "outputs"               # Points to /outputs
# CONFIG_PATH = OUTPUT_DIR / "detector_config.npz"

# WINDOW_LEN = 120  # Detector window length
# DET_LABEL_TOL = 5


# def sliding_window_probs_in_memory(model, d_norm, window_len=120, batch_size=2048):
#     d_norm = np.asarray(d_norm, dtype=np.float32)
#     N = d_norm.shape[0]
#     starts = np.arange(0, N - window_len + 1, dtype=np.int64)
#     probs = np.empty((len(starts), window_len), dtype=np.float32)

#     for i in range(0, len(starts), batch_size):
#         batch_indices = starts[i:i + batch_size]
#         batch_windows = np.stack([d_norm[s:s + window_len]
#                                  for s in batch_indices])
#         p = model.predict(batch_windows[..., np.newaxis], verbose=0)
#         probs[i:i + len(batch_indices), :] = np.squeeze(p, axis=-1)

#     return probs, starts


# def apply_refractory_fast(probs, starts, threshold, refractory, center_offset):
#     center_probs = probs[:, center_offset]
#     cand_window_indices = np.where(center_probs >= threshold)[0]

#     if len(cand_window_indices) == 0:
#         return np.array([], dtype=np.int64)

#     absolute_times = starts[cand_window_indices] + center_offset
#     kept_spikes = []
#     last_spike = -np.inf

#     for t in absolute_times:
#         if t - last_spike >= refractory:
#             kept_spikes.append(t)
#             last_spike = t

#     return np.array(kept_spikes, dtype=np.int64)


# def tune_detector_threshold(detector_model,
#                             # Remove default "D1.mat" string to avoid confusion
#                             threshold_range=np.linspace(0.5, 0.99, 10),
#                             refractory_range=range(30, 61, 15)):

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print("Preparing datasets for tuning...")

#     # --- FIX: Use DATA_RAW to find files ---
#     d1_path = DATA_RAW / "D1.mat"
#     if not d1_path.exists():
#         print(f"Error: D1.mat not found at {d1_path}")
#         return 0.75, 45  # Return defaults on failure

#     d_clean, Index_gt, _ = load_D1(d1_path)
#     d_D3 = load_unlabelled(DATA_RAW / "D3.mat")
#     d_D5 = load_unlabelled(DATA_RAW / "D5.mat")
#     # ---------------------------------------

#     # Generate noisy versions for robust tuning
#     d_D1_D3 = degrade_with_spectral_noise(d_clean, d_D3, noise_scale=1.0)
#     d_D1_D5 = degrade_with_spectral_noise(d_clean, d_D5, noise_scale=1.5)

#     print("\nPre-calculating CNN probabilities (this takes a minute)...")
#     probs_clean, starts_clean = sliding_window_probs_in_memory(
#         detector_model, d_clean, WINDOW_LEN)
#     probs_D3, starts_D3 = sliding_window_probs_in_memory(
#         detector_model, d_D1_D3, WINDOW_LEN)
#     probs_D5, starts_D5 = sliding_window_probs_in_memory(
#         detector_model, d_D1_D5, WINDOW_LEN)

#     print("Grid search (Clean=0.4 | D3=0.3 | D5=0.3)...")

#     best_score = -1
#     best_thr = 0.8
#     best_refr = 45
#     center_off = WINDOW_LEN // 2

#     for thr in threshold_range:
#         for refr in refractory_range:
#             preds_c = apply_refractory_fast(
#                 probs_clean, starts_clean, thr, refr, center_off)
#             preds_3 = apply_refractory_fast(
#                 probs_D3, starts_D3, thr, refr, center_off)
#             preds_5 = apply_refractory_fast(
#                 probs_D5, starts_D5, thr, refr, center_off)

#             f1_c = match_predictions(preds_c, Index_gt)["f1"]
#             f1_3 = match_predictions(preds_3, Index_gt)["f1"]
#             f1_5 = match_predictions(preds_5, Index_gt)["f1"]

#             w_f1 = (0.4 * f1_c + 0.3 * f1_3 + 0.3 * f1_5)

#             if w_f1 > best_score:
#                 best_score = w_f1
#                 best_thr = thr
#                 best_refr = refr

#     print(f"\n=== Best Settings Found ===")
#     print(f"Threshold: {best_thr:.3f}")
#     print(f"Refractory: {best_refr}")
#     print(f"Weighted F1: {best_score:.3f}")

#     np.savez(CONFIG_PATH,
#              decision_threshold=best_thr,
#              refractory_suppression=best_refr,
#              best_f1=best_score)
#     print(f"Saved tuning config to {CONFIG_PATH}")

#     return best_thr, best_refr


import numpy as np
import os
from pathlib import Path

from spike_pipeline.data_loader.load_datasets import load_D1
from spike_pipeline.inference.matching import match_predictions

# --- Config (Robust Paths) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to root
DATA_RAW = PROJECT_ROOT / "data"                    # Points to /data
OUTPUT_DIR = PROJECT_ROOT / "outputs"               # Points to /outputs
CONFIG_PATH = OUTPUT_DIR / "detector_config.npz"

WINDOW_LEN = 120
DET_LABEL_TOL = 5


def sliding_window_probs_in_memory(model, d_norm, window_len=120, batch_size=2048):
    d_norm = np.asarray(d_norm, dtype=np.float32)
    N = d_norm.shape[0]
    starts = np.arange(0, N - window_len + 1, dtype=np.int64)
    probs = np.empty((len(starts), window_len), dtype=np.float32)

    for i in range(0, len(starts), batch_size):
        batch_indices = starts[i:i + batch_size]
        batch_windows = np.stack([d_norm[s:s + window_len]
                                 for s in batch_indices])
        p = model.predict(batch_windows[..., np.newaxis], verbose=0)
        probs[i:i + len(batch_indices), :] = np.squeeze(p, axis=-1)

    return probs, starts


def apply_refractory_fast(probs, starts, threshold, refractory, center_offset):
    center_probs = probs[:, center_offset]
    cand_window_indices = np.where(center_probs >= threshold)[0]

    if len(cand_window_indices) == 0:
        return np.array([], dtype=np.int64)

    absolute_times = starts[cand_window_indices] + center_offset
    kept_spikes = []
    last_spike = -np.inf

    for t in absolute_times:
        if t - last_spike >= refractory:
            kept_spikes.append(t)
            last_spike = t

    return np.array(kept_spikes, dtype=np.int64)


def tune_detector_threshold(detector_model,
                            threshold_range=np.linspace(0.70, 0.95, 6),
                            refractory_range=[25, 35, 45, 55, 60]):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load ONLY Clean D1 for tuning
    print("Loading Clean D1 for tuning...")
    d1_path = DATA_RAW / "D1.mat"
    if not d1_path.exists():
        print(f"Error: D1.mat not found at {d1_path}")
        return 0.75, 45

    d_clean, Index_gt, _ = load_D1(d1_path)

    # 2. Pre-calculate probabilities ONCE
    print("Pre-calculating CNN probabilities on Clean D1...")
    probs_clean, starts_clean = sliding_window_probs_in_memory(
        detector_model, d_clean, WINDOW_LEN)

    print(
        f"Grid search over {len(threshold_range)} thresholds x {len(refractory_range)} refractories...")

    best_f1 = -1
    best_thr = 0.75
    best_refr = 45
    center_off = WINDOW_LEN // 2

    # 3. Grid Search based on Clean D1 performance only
    for thr in threshold_range:
        for refr in refractory_range:
            preds = apply_refractory_fast(
                probs_clean, starts_clean, thr, refr, center_off)

            stats = match_predictions(preds, Index_gt)
            f1 = stats["f1"]

            # print(f"Thr={thr:.2f}, Refr={refr} -> F1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_refr = refr

    print(f"\n=== Best Settings (Clean D1) ===")
    print(f"Threshold: {best_thr:.3f}")
    print(f"Refractory: {best_refr}")
    print(f"F1 Score: {best_f1:.3f}")

    # 4. Save Config
    np.savez(CONFIG_PATH,
             decision_threshold=best_thr,
             refractory_suppression=best_refr,
             best_f1=best_f1)
    print(f"Saved tuning config to {CONFIG_PATH}")

    return best_thr, best_refr
