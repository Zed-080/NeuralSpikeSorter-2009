import numpy as np
from .load_datasets import load_D1, load_unlabelled
from spike_pipeline.utils.degradation import degrade_with_spectral_noise


WINDOW = 128           # detector window size (128 samples)
TOL = 5                # +/- tolerance around spike for window start
HARD_NEG_DIST = 50     # range around a spike to create hard negatives
TARGET_RATIO = 10      # The negative:positive ratio needd to convey propportion


def extract_positive_windows(d, Index):
    """
    Builds positive (spike-containing) windows of length 128.
    """
    X_pos = []

    N = len(d)

    for s in Index:
        start = s - TOL
        end = start + WINDOW

        if start < 0 or end > N:
            continue

        win = d[start:end]
        X_pos.append(win)

    return np.array(X_pos, dtype=np.float32)


def extract_negative_windows(d, Index, num_random=10):
    """
    Builds negative windows by sampling regions with no spikes.
    Includes:
      - random negatives far from spikes
      - hard negatives near spikes (but not overlapping)
    """
    N = len(d)
    used = set(Index)

    X_neg = []

    # ---------- Hard negatives ----------
    for s in Index:
        # Try a window slightly before or after the spike
        for offset in [-HARD_NEG_DIST, +HARD_NEG_DIST]:
            start = s + offset

            if start < 0 or start + WINDOW > N:
                continue

            # ensure the window does NOT include the spike
            if any((start <= x < start + WINDOW) for x in Index):
                continue

            X_neg.append(d[start:start + WINDOW])

    # ---------- Random negatives ----------
    # Try num_random times per spike
    max_start = N - WINDOW - 1

    for _ in range(num_random * len(Index)):
        start = np.random.randint(0, max_start)

        # skip if window contains ANY spike
        if any((start <= x < start + WINDOW) for x in Index):
            continue

        X_neg.append(d[start:start + WINDOW])

    return np.array(X_neg, dtype=np.float32)


def optional_augment(X):
    """
    Optional augmentation:
      - amplitude scaling
      - tiny Gaussian noise
    """
    X_aug = []

    for w in X:
        # amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        w_scaled = w * scale

        # gaussian noise
        noise = np.random.normal(0, 0.01, size=w.shape)
        w_noisy = w_scaled + noise

        X_aug.append(w_noisy)

    return np.array(X_aug, dtype=np.float32)


def build_detector_dataset(path_to_D1, save_prefix=""):
    """
    Main function:
      Loads D1
      Builds positive + negative windows
      Optional augmentation
      Shuffles and saves X_detector.npy and y_detector.npy
    """
    d_clean, Index, Class = load_D1(path_to_D1)

    # --- load noisy reference datasets (D3, D5 for example) ---
    d_D3 = load_unlabelled("D3.mat")   # already preprocessed like D1
    d_D5 = load_unlabelled("D5.mat")

    # --- create degraded versions of D1 ---
    d_D1_D3noise = degrade_with_spectral_noise(d_clean, d_D3, noise_scale=1.0)
    d_D1_D5noise = degrade_with_spectral_noise(d_clean, d_D5, noise_scale=1.5)

    # 1. positive windows: clean + degraded
    X_pos_clean = extract_positive_windows(d_clean, Index)
    X_pos_D3 = extract_positive_windows(d_D1_D3noise, Index)
    X_pos_D5 = extract_positive_windows(d_D1_D5noise, Index)

    X_pos = np.concatenate([X_pos_clean, X_pos_D3, X_pos_D5], axis=0)

    # 2. negatives: clean + degraded
    X_neg_clean = extract_negative_windows(d_clean, Index)
    X_neg_D3 = extract_negative_windows(d_D1_D3noise, Index)
    X_neg_D5 = extract_negative_windows(d_D1_D5noise, Index)

    X_neg = np.concatenate([X_neg_clean, X_neg_D3, X_neg_D5], axis=0)

    # 3. optional augmentation on positives
    X_aug = optional_augment(X_pos)

    # 4. oversample negatives to TARGET_RATIO * num_pos (same as you already do)
    num_pos = len(X_pos) + len(X_aug)
    desired_neg = TARGET_RATIO * num_pos

    if len(X_neg) < desired_neg:
        reps = int(np.ceil(desired_neg / len(X_neg)))
        X_neg = np.tile(X_neg, (reps, 1))[:desired_neg]

    # 5. combine + shuffle (same as before)
    X = np.concatenate([X_pos, X_aug, X_neg], axis=0)
    y = np.concatenate([
        np.ones(len(X_pos)),
        np.ones(len(X_aug)),
        np.zeros(len(X_neg))
    ])

    p = np.random.permutation(len(X))
    X = X[p][..., np.newaxis]  # add channel
    y = y[p]

    np.save(f"{save_prefix}X_detector.npy", X)
    np.save(f"{save_prefix}y_detector.npy", y)

    print("Detector dataset built:")
    print(f"  Positives (clean+spec): {len(X_pos)}")
    print(f"  Augmented positives:    {len(X_aug)}")
    print(f"  Negatives:              {len(X_neg)}")
    print(f"  Total:                  {len(X)} windows")

    return X, y
