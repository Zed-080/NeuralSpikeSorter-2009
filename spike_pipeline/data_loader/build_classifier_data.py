import numpy as np
from .load_datasets import load_D1

PRE_SAMPLES = 20
POST_SAMPLES = 44
CLASS_WINDOW = PRE_SAMPLES + POST_SAMPLES  # 64


def build_classifier_dataset(path_to_D1, save_prefix=""):
    """
    Build classifier training set from D1 and save:
      X_classifier.npy      (K, 64, 1)
      y_classifier.npy      (K, 5) one-hot
      y_classifier_raw.npy  (K,) ints in 0..4
    """
    d_norm, Index, Class = load_D1(path_to_D1)

    X_list = []
    y_raw_list = []

    N = len(d_norm)

    for s, c in zip(Index, Class):
        start = s - PRE_SAMPLES
        end = s + POST_SAMPLES

        if start < 0 or end > N:
            continue

        window = d_norm[start:end]
        if len(window) != CLASS_WINDOW:
            continue

        X_list.append(window)
        # Convert 1..5 → 0..4
        y_raw_list.append(int(c) - 1)

    X = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_raw_list, dtype=np.int64)

    # One-hot
    num_classes = 5
    y_onehot = np.eye(num_classes)[y_raw]

    # Add channel dimension
    X = X[..., np.newaxis]

    # Save
    np.save(f"{save_prefix}X_classifier.npy", X)
    np.save(f"{save_prefix}y_classifier.npy", y_onehot)
    np.save(f"{save_prefix}y_classifier_raw.npy", y_raw)

    print("Classifier dataset built:")
    print(f"  Spikes used: {len(X)}")

    return X, y_onehot, y_raw
