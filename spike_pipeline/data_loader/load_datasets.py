from spike_pipeline.utils.signal_tools import bandpass_filter,  mad, build_average_mother, matched_filter_denoise
import scipy.io as spio
import numpy as np

# ---------- new extra code for  -----------


# ----------- origional code for loading datasets from .mat files -----------
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


def mad(data):
    """
    Robust Standard Deviation estimation using Median Absolute Deviation.
    Formula: 1.4826 * median(|x - median(x)|)
    """
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def MAD_normalize(d):
    """
    Normalize using Median and MAD (Median Absolute Deviation).
    This ensures spikes stay the same size even if noise increases.
    """
    med = np.median(d)
    sigma = mad(d) + 1e-8
    return (d - med) / sigma


# Remove DC drift (very low frequencies)
def remove_dc_fft(x, fs=25000, cutoff_hz=1):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    X[freqs < cutoff_hz] = 0
    return np.fft.irfft(X, n=N)


# def load_D1(path, fs=25000): >>>>>>>>>> before wavelet matched filter addition
#     """
#     Loads D1 including Index and Class,
#     and returns normalized signal + labels.
#     """
#     d, Index, Class = load_mat(path)

#     # 1. optional: remove DC drift
#     d = remove_dc_fft(d, fs=fs)

#     # 2. band-pass filter 300–3000 Hz
#     d = bandpass_filter(d, fs=fs, low=7, high=3000)

#     # 3. global z-score normalization
#     d_norm = global_normalize(d)

#     return d_norm, Index, Class


# ==================================================================================
def load_D1(path, fs=25000, use_matched_filter=False):  # >>>>>>>>>> doing a lot
    """
    Loads D1 including Index and Class,
    and returns normalized signal + labels.

    Args:
        use_matched_filter: If True, applies matched filter after bandpass
    """
    d, Index, Class = load_mat(path)

    # 1. remove DC drift
    d = remove_dc_fft(d, fs=fs)

    # 2. band-pass filter 300–3000 Hz
    d = bandpass_filter(d, fs=fs, low=7, high=3000)

    # 3. OPTIONAL: matched filter denoising (NEW)
    if use_matched_filter:
        psi, _, _ = build_average_mother(d, Index)
        if psi is not None:
            d = matched_filter_denoise(d, psi)
            print("  Applied matched filter denoising to D1")

    # 4. global z-score normalization
    # d_norm = global_normalize(d)
    d_norm = MAD_normalize(d)

    return d_norm, Index, Class
# ==================================================================================

# def load_unlabelled(path, fs=25000): >>>>>>>>>> before wavelet matched filter addition
#     """
#     Loads D2-D6 datasets which have only raw signal.
#     Returns normalized signal only.
#     """
#     d, _, _ = load_mat(path)
#     d = remove_dc_fft(d, fs=fs)
#     d = bandpass_filter(d, fs=fs, low=300, high=3000)

#     d_norm = global_normalize(d)
#     return d_norm

# ==================================================================================


def load_unlabelled(path, fs=25000, psi=None):  # >>>>>>>>>> doing a lot
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
    d = remove_dc_fft(d, fs=fs)
    d = bandpass_filter(d, fs=fs, low=7, high=3000)

    # Apply matched filter if wavelet provided
    if psi is not None:
        d = matched_filter_denoise(d, psi)
        print(f"  Applied matched filter to {path}")

    # d_norm = global_normalize(d)
    d_norm = MAD_normalize(d)
    return d_norm
# ----------------------------------------------------------------------------------


def load_D1_stages(path, fs=25000):
    """
    Load D1 and return each preprocessing stage separately.

    Returns:
        dict with keys: 'raw', 'fft_only', 'bandpass', 'wavelet', 'full_pipeline'
        Index, Class
    """
    d_raw, Index, Class = load_mat(path)

    stages = {}

    # Stage 1: Raw signal (just loaded)
    stages['raw'] = d_raw.copy()

    # Stage 2: FFT drift removal only
    d_fft = remove_dc_fft(d_raw, fs=fs)
    stages['fft_only'] = d_fft.copy()

    # Stage 3: FFT + Bandpass
    d_bandpass = bandpass_filter(d_fft, fs=fs, low=7, high=3000)
    stages['bandpass'] = d_bandpass.copy()

    # Stage 4: FFT + Wavelet (no bandpass)
    psi, _, _ = build_average_mother(d_fft, Index)
    if psi is not None:
        d_wavelet = matched_filter_denoise(d_fft, psi)
        stages['wavelet'] = d_wavelet.copy()
    else:
        stages['wavelet'] = None

    # Stage 5: Full pipeline (FFT + Bandpass + Wavelet)
    if psi is not None:
        d_full = matched_filter_denoise(d_bandpass, psi)
        stages['full_pipeline'] = d_full.copy()
    else:
        stages['full_pipeline'] = None

    return stages, Index, Class


def load_unlabelled_stages(path, psi, fs=25000):
    """
    Load unlabelled dataset and return each preprocessing stage.

    Args:
        psi: mother wavelet from D1 (required for wavelet stages)

    Returns:
        dict with keys: 'raw', 'fft_only', 'bandpass', 'wavelet', 'full_pipeline'
    """
    d_raw, _, _ = load_mat(path)

    stages = {}

    # Stage 1: Raw
    stages['raw'] = d_raw.copy()

    # Stage 2: FFT only
    d_fft = remove_dc_fft(d_raw, fs=fs)
    stages['fft_only'] = d_fft.copy()

    # Stage 3: FFT + Bandpass
    d_bandpass = bandpass_filter(d_fft, fs=fs, low=7, high=3000)
    stages['bandpass'] = d_bandpass.copy()

    # Stage 4: FFT + Wavelet (if psi provided)
    if psi is not None:
        d_wavelet = matched_filter_denoise(d_fft, psi)
        stages['wavelet'] = d_wavelet.copy()
    else:
        stages['wavelet'] = None

    # Stage 5: Full pipeline
    if psi is not None:
        d_full = matched_filter_denoise(d_bandpass, psi)
        stages['full_pipeline'] = d_full.copy()
    else:
        stages['full_pipeline'] = None

    return stages
