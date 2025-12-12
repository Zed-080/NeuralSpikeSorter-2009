from spike_pipeline.utils.signal_tools import zscore as global_normalize
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
        Index -= 1  # convert to 0-based indexing
    if Class is not None:
        Class = np.array(Class, dtype=np.int64).reshape(-1)

    return d, Index, Class


def load_D1(path, fs=25000, use_matched_filter=False):
    """
    Loads D1 including Index and Class,
    and returns normalized signal + labels.

    Args:
        use_matched_filter: If True, applies matched filter after bandpass
    """
    d, Index, Class = load_mat(path)

    d_norm = global_normalize(d)

    return d_norm, Index, Class


def load_unlabelled(path, fs=25000, psi=None):
    """
    Loads D2-D6 datasets which have only raw signal.

    Args:
        path: path to .mat file
        fs: sampling frequency
        psi: optional mother wavelet for matched filtering

    Returns:
        d_norm: normalized signal
    """
    d, _, _ = load_mat(path)

    d_norm = global_normalize(d)
    return d_norm
