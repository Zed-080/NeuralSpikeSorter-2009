# import numpy as np
# import os
# from .load_datasets import load_D1, load_unlabelled
# from spike_pipeline.utils.degradation import degrade_with_spectral_noise
# from spike_pipeline.denoise.matched_filter import build_average_spike_template, matched_filter_enhance

# # --- HELPERS ---
# PRE = 20
# POST = 44
# WAVEFORM_LEN = 64
# N_AUG_POS_CLF = 3
# SNR_RANGE_EASY = (40.0, 60.0)
# SNR_RANGE_MED = (20.0, 40.0)
# SNR_RANGE_HARD = (-5.0, 20.0)


# def sample_snr_db():
#     u = np.random.random()
#     if u < 0.2:
#         return float(np.random.uniform(*SNR_RANGE_EASY))
#     elif u < 0.5:
#         return float(np.random.uniform(*SNR_RANGE_MED))
#     else:
#         return float(np.random.uniform(*SNR_RANGE_HARD))


# def add_noise_to_target_snr(x, target_snr_db):
#     sig_power = np.mean(x ** 2)
#     if sig_power <= 1e-12:
#         return x
#     snr_lin = 10.0 ** (target_snr_db / 10.0)
#     noise_power = sig_power / snr_lin
#     noise_std = np.sqrt(noise_power)
#     return (x + np.random.normal(0.0, noise_std, size=x.shape)).astype(np.float32)


# def build_classifier_dataset(path_to_D1, save_prefix="outputs/"):
#     print(f"Loading D1 from {path_to_D1}...")
#     d_clean, Index, Class = load_D1(path_to_D1)

#     print("Building Matched Filter Template...")
#     psi, _, _ = build_average_spike_template(d_clean, Index)

#     raw_versions = [("Clean", d_clean)]
#     for ds_name in ["D2", "D3", "D4", "D5", "D6"]:
#         try:
#             d_noise_ref = load_unlabelled(f"{ds_name}.mat")
#             d_cloned = degrade_with_spectral_noise(
#                 d_clean, d_noise_ref, noise_scale=1.0)
#             raw_versions.append((f"Match_{ds_name}", d_cloned))
#         except:
#             pass

#     all_X = []
#     all_y_raw = []

#     print(f"Extracting Classifier Waveforms (Matched Filtered & Normalized)...")

#     for name, d_raw_noisy in raw_versions:
#         # 1. Apply Matched Filter
#         d_filtered = matched_filter_enhance(d_raw_noisy, psi)

#         # 2. CRITICAL FIX: Normalize to match inference pipeline
#         d_filtered = (d_filtered - d_filtered.mean()) / \
#             (d_filtered.std() + 1e-8)

#         for i, s in enumerate(Index):
#             label_int = int(Class[i]) - 1
#             start = s - PRE
#             end = s + POST

#             if start < 0 or end > len(d_filtered):
#                 continue
#             w = d_filtered[start:end]
#             if len(w) != WAVEFORM_LEN:
#                 continue

#             # Original
#             all_X.append(w)
#             all_y_raw.append(label_int)

#             # Augmented
#             for _ in range(N_AUG_POS_CLF):
#                 snr = sample_snr_db()
#                 w_aug = add_noise_to_target_snr(w, snr)
#                 scale = np.random.uniform(0.9, 1.1)
#                 all_X.append(w_aug * scale)
#                 all_y_raw.append(label_int)

#     # Finalize
#     X = np.array(all_X, dtype=np.float32)
#     y_raw = np.array(all_y_raw, dtype=np.int64)
#     if X.ndim == 2:
#         X = X[..., np.newaxis]

#     y_onehot = np.zeros((len(y_raw), 5), dtype=np.float32)
#     y_onehot[np.arange(len(y_raw)), y_raw] = 1.0

#     idx = np.random.permutation(len(X))
#     X, y_onehot, y_raw = X[idx], y_onehot[idx], y_raw[idx]

#     os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
#     np.save(f"{save_prefix}X_classifier.npy", X)
#     np.save(f"{save_prefix}y_classifier.npy", y_onehot)
#     np.save(f"{save_prefix}y_classifier_raw.npy", y_raw)

#     print(
#         f"Saved Classifier Data: X={X.shape} (Normalized & Matched Filtered)")
#     return X, y_onehot, y_raw

import numpy as np

# --- Configuration ---
PRE = 20
POST = 44
WAVEFORM_LEN = 64
N_AUG_POS_CLF = 3
RNG_SEED = 42

SNR_RANGE_EASY = (40.0, 60.0)
SNR_RANGE_MED = (20.0, 40.0)
SNR_RANGE_HARD = (-5.0, 20.0)

# --- Helpers ---


def sample_snr_db(rng):
    u = rng.random()
    if u < 0.2:
        return float(rng.uniform(*SNR_RANGE_EASY))
    elif u < 0.5:
        return float(rng.uniform(*SNR_RANGE_MED))
    else:
        return float(rng.uniform(*SNR_RANGE_HARD))


def add_noise_to_target_snr(x, target_snr_db, rng):
    sig_power = np.mean(x ** 2)
    if sig_power <= 1e-12:
        return x.copy()

    snr_lin = 10.0 ** (target_snr_db / 10.0)
    noise_std = np.sqrt(sig_power / snr_lin)
    return (x + rng.normal(0.0, noise_std, size=x.shape)).astype(np.float32)

# --- Main Builder Function ---


def build_classifier_data(d_norm: np.ndarray,
                          spike_idx: np.ndarray,
                          spike_class: np.ndarray):
    """
    Pure builder: Takes a pre-processed (noisy/normalized) signal 
    and extracts classifier waveforms.
    """
    rng = np.random.default_rng(RNG_SEED)
    N = len(d_norm)

    waveforms = []
    labels = []

    for s, c in zip(spike_idx, spike_class):
        start = s - PRE
        end = s + POST

        if start < 0 or end > N:
            continue

        w = d_norm[start:end]
        if len(w) != WAVEFORM_LEN:
            continue

        label_0_based = int(c) - 1

        # 1. Clean (processed) waveform
        waveforms.append(w)
        labels.append(label_0_based)

        # 2. Augmented copies
        for _ in range(N_AUG_POS_CLF):
            snr = sample_snr_db(rng)
            w_aug = add_noise_to_target_snr(w, snr, rng)

            # Small amplitude jitter
            scale = rng.uniform(0.9, 1.1)
            w_aug = w_aug * scale

            waveforms.append(w_aug)
            labels.append(label_0_based)

    if not waveforms:
        return np.empty((0, 64, 1)), np.empty((0, 5)), np.empty(0)

    # Format
    X = np.array(waveforms, dtype=np.float32)[..., np.newaxis]
    y_raw = np.array(labels, dtype=np.int64)

    # One-hot
    y_onehot = np.zeros((len(y_raw), 5), dtype=np.float32)
    y_onehot[np.arange(len(y_raw)), y_raw] = 1.0

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y_onehot[perm], y_raw[perm]
