# ==============================================================================
# DETECTOR DATASET CONFIGURATION
# ==============================================================================
# Constructs training data for the Spike Detector (CNN).
# Generates positive (spike) and negative (noise) windows with augmentation.
#
# 1. WINDOWING & LABELING
#    - WINDOW_LEN_DET: 120 samples. Defines the CNN input size (approx 4.8ms).
#    - DET_LABEL_TOL: +/- 5 samples. detections within this range of ground truth
#      are considered correct matches during training.
#
# 2. AUGMENTATION STRATEGY (DATA BALANCE)
#    - N_AUG_POS_DET (3): For every real spike, generate 3 noisy synthetic copies.
#      This forces the model to learn spike features invariant to noise.
#    - N_AUG_NEG_DET (2): For every non-spike window, generate 2 noisy copies.
#    - RANDOM_NEG_FACTOR (1.0): Extract 1x random background noise windows per spike.
#    - HARD_NEG_FACTOR (1.0): Extract 1x "hard" negatives (windows adjacent to spikes)
#      per spike. This trains the model to center detections precisely.
#
# 3. NOISE INJECTION LEVELS (SNR in dB)
#    Synthetic Gaussian noise is added to augment training data:
#    - EASY (40-60 dB): High-quality recording simulation.
#    - MED  (20-40 dB): Typical experimental noise floor.
#    - BAD  (-5-20 dB):  Heavy noise conditions (stress testing the detector).
# ==============================================================================


import numpy as np

# --- Configuration ---
WINDOW_LEN_DET = 120
DET_LABEL_TOL = 5      # Tolerance for label alignment (+/- 5 samples)
RNG_SEED = 42

# Augmentation Parameters
N_AUG_POS_DET = 3
N_AUG_NEG_DET = 2
RANDOM_NEG_FACTOR = 1.0
HARD_NEG_FACTOR = 1.0

SNR_RANGE_EASY = (40.0, 60.0)
SNR_RANGE_MED = (20.0, 40.0)
SNR_RANGE_HARD = (-5.0, 20.0)


def widen_labels(label_bin, width=3):
    """Expands point labels (1s) into a wider binary mask (e.g., 0011100)."""
    expanded = np.zeros_like(label_bin)
    idx = np.where(label_bin == 1)[0]
    for i in idx:
        start = max(0, i - width)
        end = min(len(label_bin), i + width + 1)
        expanded[start:end] = 1
    return expanded


def make_spike_mask(length, spike_idx, tol):
    """Creates a boolean mask marking regions around known spikes."""
    mask = np.zeros(length, dtype=bool)
    for s in spike_idx:
        start = max(0, s - tol)
        end = min(length, s + tol + 1)
        mask[start:end] = True
    return mask


def add_noise_to_target_snr_window(x, target_snr_db, rng):
    """Injects Gaussian noise into a window to reach a specific SNR."""
    x = x.astype(np.float32)
    sig_power = np.mean(x ** 2)

    if sig_power <= 1e-12:
        return x.copy()

    snr_lin = 10.0 ** (target_snr_db / 10.0)
    noise_std = np.sqrt(sig_power / snr_lin)

    noise = rng.normal(0.0, noise_std, size=x.shape)
    return (x + noise).astype(np.float32)


def sample_snr_db(rng):
    """Randomly selects an SNR level (Easy, Med, or Hard)."""
    u = rng.random()
    if u < 0.2:
        return float(rng.uniform(*SNR_RANGE_EASY))
    elif u < 0.5:
        return float(rng.uniform(*SNR_RANGE_MED))
    else:
        return float(rng.uniform(*SNR_RANGE_HARD))


def build_detector_data(d_norm: np.ndarray, spike_idx: np.ndarray):
    """
    Main builder for detector training data.
    Extracts windows, manages class balance, and applies augmentation.
    """
    rng = np.random.default_rng(RNG_SEED)
    N = len(d_norm)

    # 1. Create target labels (binary mask)
    spike_binary = np.zeros(N, dtype=np.float32)
    spike_binary[spike_idx] = 1
    spike_binary = widen_labels(spike_binary, width=3)

    # 2. Extract Positive Windows (centered on spikes)
    base_pos_windows = []
    base_pos_windows_binary = []

    for s in spike_idx:
        start = s - DET_LABEL_TOL
        end = start + WINDOW_LEN_DET
        if start < 0 or end > N:
            continue

        base_pos_windows.append(d_norm[start:end])
        base_pos_windows_binary.append(spike_binary[start:end])

    if not base_pos_windows:
        return np.empty((0, WINDOW_LEN_DET, 1)), np.empty((0, WINDOW_LEN_DET, 1))

    n_pos = len(base_pos_windows)

    # 3. Extract Random Negatives (pure noise regions)
    spike_mask = make_spike_mask(N, spike_idx, DET_LABEL_TOL)
    n_random = int(RANDOM_NEG_FACTOR * n_pos)
    random_neg, random_neg_bin = [], []

    attempts = 0
    while len(random_neg) < n_random and attempts < n_random * 20:
        attempts += 1
        start = rng.integers(0, N - WINDOW_LEN_DET)
        end = start + WINDOW_LEN_DET

        # Only keep if window contains no spike data
        if not spike_mask[start:end].any():
            random_neg.append(d_norm[start:end])
            random_neg_bin.append(spike_binary[start:end])

    # 4. Extract Hard Negatives (regions close to but not centered on spikes)
    n_hard = int(HARD_NEG_FACTOR * n_pos)
    hard_neg, hard_neg_bin = [], []

    for s in spike_idx:
        if len(hard_neg) >= n_hard:
            break
        # Offset slightly left or right of the spike
        for offset in [DET_LABEL_TOL * 2, -DET_LABEL_TOL * 2]:
            start = s + offset - WINDOW_LEN_DET // 2
            end = start + WINDOW_LEN_DET

            if start < 0 or end > N:
                continue
            if start <= s < end and spike_binary[s] == 1:
                continue  # Safety check

            hard_neg.append(d_norm[start:end])
            hard_neg_bin.append(spike_binary[start:end])
            if len(hard_neg) >= n_hard:
                break

    # 5. Apply SNR Augmentation
    final_X, final_y = [], []

    # Process positives
    for w, wb in zip(base_pos_windows, base_pos_windows_binary):
        final_X.append(w)
        final_y.append(wb)
        for _ in range(N_AUG_POS_DET):
            final_X.append(add_noise_to_target_snr_window(
                w, sample_snr_db(rng), rng))
            final_y.append(wb)

    # Process negatives
    base_neg = random_neg + hard_neg
    base_neg_bin = random_neg_bin + hard_neg_bin

    for w, wb in zip(base_neg, base_neg_bin):
        final_X.append(w)
        final_y.append(wb)
        for _ in range(N_AUG_NEG_DET):
            final_X.append(add_noise_to_target_snr_window(
                w, sample_snr_db(rng), rng))
            final_y.append(wb)

    # 6. Final format and shuffle
    X = np.array(final_X, dtype=np.float32)
    y = np.array(final_y, dtype=np.float32)

    if len(X) > 0:
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
        perm = rng.permutation(len(X))
        return X[perm], y[perm]

    return np.empty((0, WINDOW_LEN_DET, 1)), np.empty((0, WINDOW_LEN_DET, 1))
