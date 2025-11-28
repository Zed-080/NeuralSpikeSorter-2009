from spike_pipeline.utils.signal_tools import bandpass_filter
import scipy.io as spio
import numpy as np


def load_mat(path):
    """
    Loads a .mat file and extracts:
      - d      (raw signal)
      - Index  (optional)
      - Class  (optional)

    Always returns:
      d, Index, Class   (Index/Class = None if missing)
    """
    mat = spio.loadmat(path, squeeze_me=True)

    d = mat["d"].astype(np.float32)

    Index = mat.get("Index", None)
    Class = mat.get("Class", None)

    # ensure Index / Class are always 1D arrays if present
    if Index is not None:
        Index = np.array(Index, dtype=np.int64).reshape(-1)
    if Class is not None:
        Class = np.array(Class, dtype=np.int64).reshape(-1)

    return d, Index, Class


def global_normalize(d):
    """
    Global z-score normalization:
        (d - mean) / std
    """
    mu = np.mean(d)
    sigma = np.std(d) + 1e-8   # avoid divide-by-zero

    return (d - mu) / sigma

# Remove DC drift (very low frequencies)


def remove_dc_fft(x, fs=25000, cutoff_hz=1):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    X[freqs < cutoff_hz] = 0
    return np.fft.irfft(X, n=N)


def load_D1(path, fs=25000):
    """
    Loads D1 including Index and Class,
    and returns normalized signal + labels.
    """
    d, Index, Class = load_mat(path)

    # 1. optional: remove DC drift
    d = remove_dc_fft(d, fs=fs)

    # 2. band-pass filter 300–3000 Hz
    d = bandpass_filter(d, fs=fs, low=7, high=3000)

    # 3. global z-score normalization
    d_norm = global_normalize(d)

    return d_norm, Index, Class


def load_unlabelled(path, fs=25000):
    """
    Loads D2-D6 datasets which have only raw signal.
    Returns normalized signal only.
    """
    d, _, _ = load_mat(path)
    d = remove_dc_fft(d, fs=fs)
    d = bandpass_filter(d, fs=fs, low=300, high=3000)

    d_norm = global_normalize(d)
    return d_norm
