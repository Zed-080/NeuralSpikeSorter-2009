# ==============================================================================
# WAVEFORM EXTRACTION UTILITIES
# ==============================================================================
# Tools for cutting out short signal clips around detected spike indices.
#
# 1. WINDOW GEOMETRY
#    - Total Length: 64 samples.
#    - Alignment: Asymmetric. We take 20 samples BEFORE the peak and 44 samples
#      AFTER. This captures the initial depolarization rise and the longer
#      repolarization tail.
#
# 2. OUTPUT FORMAT
#    - Returns a tensor of shape (N, 64, 1) ready for the CNN Classifier.
#    - Handles boundary conditions by dropping spikes too close to the signal edges.
# ==============================================================================

import numpy as np

PRE = 20
POST = 44
WINDOW = PRE + POST   # = 64


def extract_waveform_64(d_norm, spike_indices):
    """
    Extracts 64-sample windows centered on given spike indices.

    Args:
        d_norm (np.array): Normalized 1D signal.
        spike_indices (np.array): Array of integer indices where spikes were detected.

    Returns:
        X (np.array): (N_spikes, 64, 1) array suitable for Keras/TensorFlow input.
    """
    waveforms = []
    N = len(d_norm)

    PRE = 20
    POST = 44

    for s in spike_indices:
        start = s - PRE
        end = s + POST

        # Bounds check: Skip spikes too close to start or end
        if start < 0 or end > N:
            continue

        w = d_norm[start:end]

        if len(w) != 64:
            continue

        waveforms.append(w)

    X = np.array(waveforms, dtype=np.float32)

    # Shape correction for CNN input (Batch, Time, Channels)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]      # (N,64) -> (N,64,1)
    elif X.shape[1] == 1 and X.shape[2] == 64:
        X = np.transpose(X, (0, 2, 1))  # (N,1,64) -> (N,64,1)

    return X
