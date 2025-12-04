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
#     #    We create the noisy versions FIRST, then apply Matched Filter to ALL of them.
#     #    This mimics the inference pipeline: Raw(Noisy) -> Matched Filter -> Detect
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

#     print("\nApplying Matched Filter & Extracting Windows...")

#     for version_name, d_raw_noisy in raw_versions:
#         # --- KEY CHANGE: Apply Matched Filter to the full trace ---
#         # This aligns the training data domain with your inference pipeline.
#         d_filtered = matched_filter_enhance(d_raw_noisy, psi)

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
#     np.save(f"{save_prefix}y_detector.npy", y)

#     print(f"Saved Detector Data: X={X.shape} (Matched Filtered)")
#     return X, y


import numpy as np
from .load_datasets import load_D1, load_unlabelled
from spike_pipeline.utils.degradation import degrade_with_spectral_noise
from spike_pipeline.denoise.matched_filter import build_average_spike_template, matched_filter_enhance

# Configuration
WINDOW = 128
TOL = 5
N_AUG_POS = 3
N_AUG_NEG = 1

SNR_RANGE_EASY = (40.0, 60.0)
SNR_RANGE_MED = (20.0, 40.0)
SNR_RANGE_HARD = (-5.0, 20.0)


def widen_labels(label_bin, width=3):
    """Expands single '1' into wider mask (000111000)."""
    expanded = np.zeros_like(label_bin)
    idx = np.where(label_bin == 1)[0]
    for i in idx:
        start = max(0, i - width)
        end = min(len(label_bin), i + width + 1)
        expanded[start:end] = 1
    return expanded


def add_noise_to_target_snr(x, target_snr_db):
    """Local helper for augmentation noise."""
    sig_power = np.mean(x ** 2)
    if sig_power <= 1e-12:
        return x
    snr_lin = 10.0 ** (target_snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise_std = np.sqrt(noise_power)
    return (x + np.random.normal(0.0, noise_std, size=x.shape)).astype(np.float32)


def sample_snr_db():
    u = np.random.random()
    if u < 0.2:
        return float(np.random.uniform(*SNR_RANGE_EASY))
    elif u < 0.5:
        return float(np.random.uniform(*SNR_RANGE_MED))
    else:
        return float(np.random.uniform(*SNR_RANGE_HARD))


def build_detector_dataset(path_to_D1, save_prefix="outputs/"):
    print("Loading Clean D1...")
    d_clean, Index, _ = load_D1(path_to_D1)

    # 1. Build Matched Filter Template from D1
    print("Building Matched Filter Template...")
    psi, _, _ = build_average_spike_template(d_clean, Index)

    # 2. Create Label Mask (Widen for Sequence Labeling)
    full_mask = np.zeros(len(d_clean), dtype=np.float32)
    full_mask[Index] = 1.0
    full_mask = widen_labels(full_mask, width=3)

    # 3. Create Signal Versions
    #    We create the noisy versions FIRST, then apply Matched Filter to ALL of them.
    #    This mimics the inference pipeline: Raw(Noisy) -> Matched Filter -> Detect
    raw_versions = [("Clean", d_clean)]
    target_datasets = ["D2", "D3", "D4", "D5", "D6"]

    for ds_name in target_datasets:
        try:
            d_noise_ref = load_unlabelled(f"{ds_name}.mat")
            d_cloned = degrade_with_spectral_noise(
                d_clean, d_noise_ref, noise_scale=1.0)
            raw_versions.append((f"Match_{ds_name}", d_cloned))
        except FileNotFoundError:
            pass

    all_X = []
    all_y = []

    print("\nApplying Matched Filter & Extracting Windows...")

    for version_name, d_raw_noisy in raw_versions:
        # --- KEY CHANGE: Apply Matched Filter to the full trace ---
        # This aligns the training data domain with your inference pipeline.
        d_filtered = matched_filter_enhance(d_raw_noisy, psi)

        # A. POSITIVE WINDOWS
        for s in Index:
            start = s - TOL
            end = start + WINDOW
            if start < 0 or end > len(d_filtered):
                continue

            w = d_filtered[start:end]
            m = full_mask[start:end]

            all_X.append(w)
            all_y.append(m)

            # Augmentation
            for _ in range(N_AUG_POS):
                w_aug = add_noise_to_target_snr(w, sample_snr_db())
                scale = np.random.uniform(0.9, 1.1)
                all_X.append(w_aug * scale)
                all_y.append(m)

        # B. NEGATIVE WINDOWS
        num_neg = len(Index) * 2
        for _ in range(num_neg):
            for _ in range(10):
                start = np.random.randint(0, len(d_filtered) - WINDOW)
                m = full_mask[start:start+WINDOW]
                if np.sum(m) == 0:
                    w = d_filtered[start:start+WINDOW]
                    all_X.append(w)
                    all_y.append(m)

                    for _ in range(N_AUG_NEG):
                        w_aug = add_noise_to_target_snr(w, sample_snr_db())
                        all_X.append(w_aug)
                        all_y.append(m)
                    break

    # Finalize
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    if X.ndim == 2:
        X = X[..., np.newaxis]
    if y.ndim == 2:
        y = y[..., np.newaxis]

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    np.save(f"{save_prefix}X_detector.npy", X)
    np.save(f"{save_prefix}y_detector.npy", y)

    print(f"Saved Detector Data: X={X.shape} (Matched Filtered)")
    return X, y
