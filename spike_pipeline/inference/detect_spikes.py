import numpy as np


def sliding_window_predict(model, d_norm, threshold=0.8, refractory=40, window=128):
    """
    Slide window across entire signal:
        for i in range(N - window)
            predict probability
            keep if > threshold
    Then apply refractory suppression to keep only valid spikes.
    """
    N = len(d_norm)
    preds = []

    # build all windows in one go (fast)
    X = np.zeros((N - window, window, 1), dtype=np.float32)
    for i in range(N - window):
        X[i, :, 0] = d_norm[i:i+window]

    probs = model.predict(X, verbose=0).flatten()

    # collect candidate spike indices
    for i, p in enumerate(probs):
        if p >= threshold:
            window_data = d_norm[i:i+window]

            # Find the index of the minimum value relative to the start of the window (i)
            relative_min_idx = np.argmin(window_data)

            # The absolute index in the original signal d_norm
            absolute_min_idx = i + relative_min_idx

            preds.append(absolute_min_idx)

    preds = np.array(preds, dtype=np.int64)

    # apply refractory
    return apply_refractory(preds, refractory)


def apply_refractory(indices, refractory):
    """
    Remove spikes that occur too close to each other.
    Keep the first spike, drop any within 'refractory' samples.
    """
    if len(indices) == 0:
        return np.array([], dtype=np.int64)

    final = [indices[0]]

    for idx in indices[1:]:
        if idx - final[-1] >= refractory:
            final.append(idx)

    return np.array(final, dtype=np.int64)
