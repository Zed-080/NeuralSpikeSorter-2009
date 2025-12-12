# ==============================================================================
# CLASSIFIER DATASET CONFIGURATION
# ==============================================================================
# Constructs training data for the Spike Classifier (CNN).
# Extracts short, centered waveform snapshots for 5-class classification.
#
# 1. WINDOWING (WAVEFORM EXTRACTION)
#    - WINDOW: 64 samples (approx 2.56ms).
#    - ALIGNMENT: Asymmetric window (Pre=20, Post=44) relative to the spike peak.
#      This is critical to capture the depolarization rise and repolarization tail.
#
# 2. AUGMENTATION STRATEGY
#    - N_AUG_POS_CLF: Controls number of augmented copies per spike.
#      (Currently set to 0 as training on high-quality templates proved most effective).
#    - JITTER: Adds slight amplitude scaling (0.9x - 1.1x) to simulate impedance variations.
#
# 3. LABELING
#    - Returns both One-Hot encoded labels (for Cross-Entropy Loss) and
#      Raw integer labels (for Stratified splitting).
# ==============================================================================

import numpy as np

# --- Configuration ---
PRE = 20
POST = 44
WAVEFORM_LEN = 64
N_AUG_POS_CLF = 0  # Set to 0 based on performance tuning
RNG_SEED = 42

SNR_RANGE_EASY = (40.0, 60.0)
SNR_RANGE_MED = (20.0, 40.0)
SNR_RANGE_HARD = (-5.0, 20.0)


def sample_snr_db(rng):
    """Randomly selects an SNR level for augmentation."""
    u = rng.random()
    if u < 0.2:
        return float(rng.uniform(*SNR_RANGE_EASY))
    elif u < 0.5:
        return float(rng.uniform(*SNR_RANGE_MED))
    else:
        return float(rng.uniform(*SNR_RANGE_HARD))


def add_noise_to_target_snr(x, target_snr_db, rng):
    """Injects Gaussian noise to match target SNR."""
    sig_power = np.mean(x ** 2)
    if sig_power <= 1e-12:
        return x.copy()

    snr_lin = 10.0 ** (target_snr_db / 10.0)
    noise_std = np.sqrt(sig_power / snr_lin)

    return (x + rng.normal(0.0, noise_std, size=x.shape)).astype(np.float32)


def build_classifier_data(d_norm: np.ndarray,
                          spike_idx: np.ndarray,
                          spike_class: np.ndarray):
    """
    Extracts and labels individual spike waveforms from the signal.
    Returns: X (waveforms), y_onehot (training), y_raw (evaluation).
    """
    rng = np.random.default_rng(RNG_SEED)
    N = len(d_norm)

    waveforms = []
    labels = []

    for s, c in zip(spike_idx, spike_class):
        start = s - PRE
        end = s + POST

        # Bounds check
        if start < 0 or end > N:
            continue

        w = d_norm[start:end]
        if len(w) != WAVEFORM_LEN:
            continue

        label_0_based = int(c) - 1

        # 1. Add clean waveform
        waveforms.append(w)
        labels.append(label_0_based)

        # 2. Add augmented copies (if enabled)
        for _ in range(N_AUG_POS_CLF):
            snr = sample_snr_db(rng)
            w_aug = add_noise_to_target_snr(w, snr, rng)

            # Subtle amplitude jitter
            scale = rng.uniform(0.9, 1.1)
            w_aug = w_aug * scale

            waveforms.append(w_aug)
            labels.append(label_0_based)

    if not waveforms:
        return np.empty((0, 64, 1)), np.empty((0, 5)), np.empty(0)

    # Reshape for CNN (Batch, Time, Channels)
    X = np.array(waveforms, dtype=np.float32)[..., np.newaxis]
    y_raw = np.array(labels, dtype=np.int64)

    # Create One-hot encoding for training
    y_onehot = np.zeros((len(y_raw), 5), dtype=np.float32)
    y_onehot[np.arange(len(y_raw)), y_raw] = 1.0

    # Shuffle dataset
    perm = rng.permutation(len(X))
    return X[perm], y_onehot[perm], y_raw[perm]
