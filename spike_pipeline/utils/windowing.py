import numpy as np


def sliding_windows(x, window, stride=1):
    """
    Build sliding windows of length 'window' across the signal.
    Returns (num_windows, window_len).
    """
    N = len(x)
    out = np.zeros((N - window + 1, window), dtype=np.float32)

    for i in range(N - window + 1):
        out[i] = x[i:i+window]

    return out


def extract_window(x, center, pre, post):
    """
    Extract window centered at a spike.
    Returns None if out of bounds.
    """
    start = center - pre
    end = center + post

    if start < 0 or end > len(x):
        return None

    return x[start:end]
