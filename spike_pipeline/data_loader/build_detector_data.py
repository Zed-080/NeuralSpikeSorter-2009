import numpy as np
from .load_datasets import load_D1


WINDOW = 128           # detector window size (128 samples)
TOL = 5                # +/- tolerance around spike for window start
HARD_NEG_DIST = 50     # range around a spike to create hard negatives


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
    d_norm, Index, Class = load_D1(path_to_D1)

    # 1. positive windows
    X_pos = extract_positive_windows(d_norm, Index)

    # 2. negative windows
    X_neg = extract_negative_windows(d_norm, Index)

    # 3. apply augmentation to positives (optional)
    X_aug = optional_augment(X_pos)

    # 4. combine
    X = np.concatenate([X_pos, X_aug, X_neg], axis=0)
    y = np.concatenate([
        np.ones(len(X_pos)),         # real positives
        np.ones(len(X_aug)),         # augmented positives
        np.zeros(len(X_neg))         # negatives
    ])

    # 5. shuffle
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    # add channel dimension
    X = X[..., np.newaxis]

    # 6. save
    np.save(f"{save_prefix}X_detector.npy", X)
    np.save(f"{save_prefix}y_detector.npy", y)

    print(f"Detector dataset built:")
    print(f"  Positives: {len(X_pos)}")
    print(f"  Augmented positives: {len(X_aug)}")
    print(f"  Negatives: {len(X_neg)}")
    print(f"  Total: {len(X)} windows")

    return X, y
