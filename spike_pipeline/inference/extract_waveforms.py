import numpy as np
from spike_pipeline.utils.normalization import normalize_window


PRE = 20
POST = 44
WINDOW = PRE + POST   # = 64


def extract_waveform_64(d_norm, spike_index):
    """
    Extract 64-sample waveform around detected spike.
    Returns None if out of bounds.
    """
    waveforms = []
    N = len(d_norm)

    start = spike_index - PRE
    end = spike_index + POST

    if start < 0 or end > len(d_norm):
        return None

    w = d_norm[start:end]

    # ensure shape is exactly 64
    if len(w) != WINDOW:
        return None

    w = normalize_window(w)

    waveforms.append(w)

    return np.array(waveforms, dtype=np.float32)
