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


# --- Main Builder Function ---

def build_detector_data(d_norm: np.ndarray, spike_idx: np.ndarray):
    """
    Pure builder: Takes a pre-processed (noisy/normalized) signal 
    and extracts training windows.
    Returns X, y arrays (does NOT save to disk).
    """
    rng = np.random.default_rng(RNG_SEED)
    N = len(d_norm)

    # Create Binary Label Mask
    spike_binary = np.zeros(N, dtype=np.float32)
    spike_binary[spike_idx] = 1
    spike_binary = widen_labels(spike_binary, width=3)

    # Extract Positive Windows
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

    # Extract Random Negatives
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

    # Extract Hard Negatives (Near spikes)
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

    # Apply Augmentation (Snippet Level)
    final_X, final_y = [], []

    # Positives: Clean + Augmented
    for w, wb in zip(base_pos_windows, base_pos_windows_binary):
        final_X.append(w)
        final_y.append(wb)
        # Augment
        for _ in range(N_AUG_POS_DET):
            w_aug = add_noise_to_target_snr_window(
                w, sample_snr_db(rng), rng)
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

    # Format and Shuffle
    X = np.array(final_X, dtype=np.float32)
    y = np.array(final_y, dtype=np.float32)

    if len(X) > 0:
        X = X[..., np.newaxis]  # (N, 120, 1)
        y = y[..., np.newaxis]  # (N, 120, 1)
        perm = rng.permutation(len(X))
        return X[perm], y[perm]
    else:
        return np.empty((0, WINDOW_LEN_DET, 1)), np.empty((0, WINDOW_LEN_DET, 1))
