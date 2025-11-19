import numpy as np

PRE = 20
POST = 44
WINDOW = PRE + POST   # = 64


def extract_waveform_64(d_norm, spike_index):
    """
    Extract 64-sample waveform around detected spike.
    Returns None if out of bounds.
    """
    start = spike_index - PRE
    end = spike_index + POST

    if start < 0 or end > len(d_norm):
        return None

    w = d_norm[start:end]

    # ensure shape is exactly 64
    if len(w) != WINDOW:
        return None

    return w.astype(np.float32)
