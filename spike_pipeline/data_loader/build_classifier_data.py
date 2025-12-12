import numpy as np

# --- Configuration ---
PRE = 20
POST = 44
WAVEFORM_LEN = 64
N_AUG_POS_CLF = 0  # seems to effecively be 0 in better run
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
