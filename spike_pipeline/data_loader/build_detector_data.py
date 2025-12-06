# import numpy as np
# from .load_datasets import load_D1, load_unlabelled
# from spike_pipeline.utils.degradation import degrade_with_spectral_noise
# from spike_pipeline.denoise.matched_filter import build_average_spike_template, matched_filter_enhance

# # Configuration
# WINDOW = 128
# TOL = 5
# N_AUG_POS = 3
# N_AUG_NEG = 1

# SNR_RANGE_EASY = (40.0, 60.0)
# SNR_RANGE_MED = (20.0, 40.0)
# SNR_RANGE_HARD = (-5.0, 20.0)


# def widen_labels(label_bin, width=3):
#     """Expands single '1' into wider mask (000111000)."""
#     expanded = np.zeros_like(label_bin)
#     idx = np.where(label_bin == 1)[0]
#     for i in idx:
#         start = max(0, i - width)
#         end = min(len(label_bin), i + width + 1)
#         expanded[start:end] = 1
#     return expanded


# def add_noise_to_target_snr(x, target_snr_db):
#     """Local helper for augmentation noise."""
#     sig_power = np.mean(x ** 2)
#     if sig_power <= 1e-12:
#         return x
#     snr_lin = 10.0 ** (target_snr_db / 10.0)
#     noise_power = sig_power / snr_lin
#     noise_std = np.sqrt(noise_power)
#     return (x + np.random.normal(0.0, noise_std, size=x.shape)).astype(np.float32)


# def sample_snr_db():
#     u = np.random.random()
#     if u < 0.2:
#         return float(np.random.uniform(*SNR_RANGE_EASY))
#     elif u < 0.5:
#         return float(np.random.uniform(*SNR_RANGE_MED))
#     else:
#         return float(np.random.uniform(*SNR_RANGE_HARD))


# def build_detector_dataset(path_to_D1, save_prefix="outputs/"):
#     print("Loading Clean D1...")
#     d_clean, Index, _ = load_D1(path_to_D1)

#     # 1. Build Matched Filter Template from D1
#     print("Building Matched Filter Template...")
#     psi, _, _ = build_average_spike_template(d_clean, Index)

#     # 2. Create Label Mask (Widen for Sequence Labeling)
#     full_mask = np.zeros(len(d_clean), dtype=np.float32)
#     full_mask[Index] = 1.0
#     full_mask = widen_labels(full_mask, width=3)

#     # 3. Create Signal Versions
#     raw_versions = [("Clean", d_clean)]
#     target_datasets = ["D2", "D3", "D4", "D5", "D6"]

#     for ds_name in target_datasets:
#         try:
#             d_noise_ref = load_unlabelled(f"{ds_name}.mat")
#             d_cloned = degrade_with_spectral_noise(
#                 d_clean, d_noise_ref, noise_scale=1.0)
#             raw_versions.append((f"Match_{ds_name}", d_cloned))
#         except FileNotFoundError:
#             pass

#     all_X = []
#     all_y = []

#     print("\nApplying Matched Filter, Normalizing & Extracting Windows...")

#     for version_name, d_raw_noisy in raw_versions:
#         # --- KEY CHANGE: Apply Matched Filter to the full trace ---
#         d_filtered = matched_filter_enhance(d_raw_noisy, psi)

#         # ---> CRITICAL FIX: Normalize to std=1 <---
#         d_filtered = (d_filtered - d_filtered.mean()) / \
#             (d_filtered.std() + 1e-8)

#         # A. POSITIVE WINDOWS
#         for s in Index:
#             start = s - TOL
#             end = start + WINDOW
#             if start < 0 or end > len(d_filtered):
#                 continue

#             w = d_filtered[start:end]
#             m = full_mask[start:end]

#             all_X.append(w)
#             all_y.append(m)

#             # Augmentation
#             for _ in range(N_AUG_POS):
#                 w_aug = add_noise_to_target_snr(w, sample_snr_db())
#                 scale = np.random.uniform(0.9, 1.1)
#                 all_X.append(w_aug * scale)
#                 all_y.append(m)

#         # B. NEGATIVE WINDOWS
#         num_neg = len(Index) * 2
#         for _ in range(num_neg):
#             for _ in range(10):
#                 start = np.random.randint(0, len(d_filtered) - WINDOW)
#                 m = full_mask[start:start+WINDOW]
#                 if np.sum(m) == 0:
#                     w = d_filtered[start:start+WINDOW]
#                     all_X.append(w)
#                     all_y.append(m)

#                     for _ in range(N_AUG_NEG):
#                         w_aug = add_noise_to_target_snr(w, sample_snr_db())
#                         all_X.append(w_aug)
#                         all_y.append(m)
#                     break

#     # Finalize
#     X = np.array(all_X, dtype=np.float32)
#     y = np.array(all_y, dtype=np.float32)
#     if X.ndim == 2:
#         X = X[..., np.newaxis]
#     if y.ndim == 2:
#         y = y[..., np.newaxis]

#     idx = np.random.permutation(len(X))
#     X, y = X[idx], y[idx]

#     np.save(f"{save_prefix}X_detector.npy", X)
#     np.save(f"{ save_prefix}y_detector.npy", y)

#     print(f"Saved Detector Data: X={X.shape} (Normalized & Matched Filtered)")
#     return X, y

import numpy as np

# --- Configuration ---
WINDOW_LEN_DET = 120   # src uses 120 for detector
DET_LABEL_TOL = 5      # +/- 5 samples alignment
RNG_SEED = 42

# Augmentation Settings (Snippet Level)
N_AUG_POS_DET = 3
N_AUG_NEG_DET = 2
RANDOM_NEG_FACTOR = 1.0
HARD_NEG_FACTOR = 1.0

SNR_RANGE_EASY = (40.0, 60.0)
SNR_RANGE_MED = (20.0, 40.0)
SNR_RANGE_HARD = (-5.0, 20.0)
NOISE_STD = 0.02
AMP_SCALE_RANGE = (0.9, 1.1)


# --- Helpers ---

def widen_labels(label_bin, width=3):
    """Expands single '1' into wider mask (000111000)."""
    expanded = np.zeros_like(label_bin)
    idx = np.where(label_bin == 1)[0]
    for i in idx:
        start = max(0, i - width)
        end = min(len(label_bin), i + width + 1)
        expanded[start:end] = 1
    return expanded


def make_spike_mask(length, spike_idx, tol):
    mask = np.zeros(length, dtype=bool)
    for s in spike_idx:
        start = max(0, s - tol)
        end = min(length, s + tol + 1)
        mask[start:end] = True
    return mask


def add_noise_to_target_snr_window(x, target_snr_db, rng):
    x = x.astype(np.float32)
    sig_power = np.mean(x ** 2)
    if sig_power <= 1e-12:
        return x.copy()

    snr_lin = 10.0 ** (target_snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(0.0, noise_std, size=x.shape)
    return (x + noise).astype(np.float32)


def sample_snr_db(rng):
    u = rng.random()
    if u < 0.2:
        return float(rng.uniform(*SNR_RANGE_EASY))
    elif u < 0.5:
        return float(rng.uniform(*SNR_RANGE_MED))
    else:
        return float(rng.uniform(*SNR_RANGE_HARD))


def augment_window(x, rng):
    scale = rng.uniform(*AMP_SCALE_RANGE)
    noisy = x * scale + rng.normal(0.0, NOISE_STD, size=x.shape)
    return noisy.astype(np.float32)


# --- Main Builder Function ---

def build_detector_data(d_norm: np.ndarray, spike_idx: np.ndarray):
    """
    Pure builder: Takes a pre-processed (noisy/normalized) signal 
    and extracts training windows.
    Returns X, y arrays (does NOT save to disk).
    """
    rng = np.random.default_rng(RNG_SEED)
    N = len(d_norm)

    # 1. Create Binary Label Mask
    spike_binary = np.zeros(N, dtype=np.float32)
    spike_binary[spike_idx] = 1
    spike_binary = widen_labels(spike_binary, width=3)

    # 2. Extract Positive Windows
    base_pos_windows = []
    base_pos_windows_binary = []

    for s in spike_idx:
        start = s - DET_LABEL_TOL
        end = start + WINDOW_LEN_DET
        if start < 0 or end > N:
            continue

        w = d_norm[start:end]
        w_bin = spike_binary[start:end]

        base_pos_windows.append(w)
        base_pos_windows_binary.append(w_bin)

    if not base_pos_windows:
        # Return empty if no spikes found (rare safety check)
        return np.empty((0, WINDOW_LEN_DET, 1)), np.empty((0, WINDOW_LEN_DET, 1))

    base_pos_windows = np.stack(base_pos_windows)
    base_pos_windows_binary = np.stack(base_pos_windows_binary)
    n_pos = len(base_pos_windows)

    # 3. Extract Random Negatives
    spike_mask = make_spike_mask(N, spike_idx, DET_LABEL_TOL)
    n_random = int(RANDOM_NEG_FACTOR * n_pos)
    random_neg, random_neg_bin = [], []

    attempts = 0
    while len(random_neg) < n_random and attempts < n_random * 20:
        attempts += 1
        start = rng.integers(0, N - WINDOW_LEN_DET)
        end = start + WINDOW_LEN_DET
        if not spike_mask[start:end].any():
            random_neg.append(d_norm[start:end])
            random_neg_bin.append(spike_binary[start:end])

    # 4. Extract Hard Negatives (Near spikes)
    n_hard = int(HARD_NEG_FACTOR * n_pos)
    hard_neg, hard_neg_bin = [], []

    for s in spike_idx:
        if len(hard_neg) >= n_hard:
            break
        for offset in [DET_LABEL_TOL * 2, -DET_LABEL_TOL * 2]:
            start = s + offset - WINDOW_LEN_DET // 2
            end = start + WINDOW_LEN_DET
            if start < 0 or end > N:
                continue
            # Ensure we don't accidentally center on the spike
            if start <= s < end and spike_binary[s] == 1:
                continue

            hard_neg.append(d_norm[start:end])
            hard_neg_bin.append(spike_binary[start:end])
            if len(hard_neg) >= n_hard:
                break

    # 5. Apply Augmentation (Snippet Level)
    final_X, final_y = [], []

    # Positives: Clean + Augmented
    for w, wb in zip(base_pos_windows, base_pos_windows_binary):
        final_X.append(w)
        final_y.append(wb)
        # Augment
        for _ in range(N_AUG_POS_DET):
            w_aug = augment_window(w, rng)
            w_aug = add_noise_to_target_snr_window(
                w_aug, sample_snr_db(rng), rng)
            final_X.append(w_aug)
            final_y.append(wb)

    # Negatives: Clean + Augmented
    base_neg = (random_neg + hard_neg)
    base_neg_bin = (random_neg_bin + hard_neg_bin)

    for w, wb in zip(base_neg, base_neg_bin):
        final_X.append(w)
        final_y.append(wb)
        for _ in range(N_AUG_NEG_DET):
            w_aug = add_noise_to_target_snr_window(w, sample_snr_db(rng), rng)
            final_X.append(w_aug)
            final_y.append(wb)

    # 6. Format and Shuffle
    X = np.array(final_X, dtype=np.float32)
    y = np.array(final_y, dtype=np.float32)

    if len(X) > 0:
        X = X[..., np.newaxis]  # (N, 120, 1)
        y = y[..., np.newaxis]  # (N, 120, 1)
        perm = rng.permutation(len(X))
        return X[perm], y[perm]
    else:
        return np.empty((0, WINDOW_LEN_DET, 1)), np.empty((0, WINDOW_LEN_DET, 1))
