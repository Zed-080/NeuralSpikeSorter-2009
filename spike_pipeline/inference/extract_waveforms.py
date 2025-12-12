import numpy as np

PRE = 20
POST = 44
WINDOW = PRE + POST   # = 64


def extract_waveform_64(d_norm, spike_indices):
    waveforms = []
    N = len(d_norm)

    PRE = 20
    POST = 44

    for s in spike_indices:
        start = s - PRE
        end = s + POST

        if start < 0 or end > N:
            continue

        w = d_norm[start:end]

        if len(w) != 64:
            continue

        waveforms.append(w)

    X = np.array(waveforms, dtype=np.float32)

    # Shape correction
    if X.ndim == 2:
        X = X[:, :, np.newaxis]      # (N,64)→(N,64,1)
    elif X.shape[1] == 1 and X.shape[2] == 64:
        X = np.transpose(X, (0, 2, 1))  # (N,1,64)→(N,64,1)

    return X
