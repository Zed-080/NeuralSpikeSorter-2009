import numpy as np


def make_spectral_noise_like(ref_signal, length, fs=25000, seg_len=50000):
    """
    Build coloured noise with similar spectrum to a reference signal.

    ref_signal: 1D np.array (e.g. D3 after your normal preprocessing)
    length:     length of noise to generate (len(D1))
    """
    ref_signal = ref_signal.astype(np.float32).flatten()
    N_ref = len(ref_signal)

    if seg_len > N_ref:
        seg_len = N_ref

    # pick a random segment from ref_signal as "noise-ish"
    start = np.random.randint(0, N_ref - seg_len)
    seg = ref_signal[start:start + seg_len]

    # FFT of that segment
    S = np.fft.rfft(seg)
    mag = np.abs(S) + 1e-8   # avoid divide-by-zero

    # white noise -> frequency domain
    white = np.random.randn(length).astype(np.float32)
    W = np.fft.rfft(white)

    # shape white noise spectrum to match mag
    W_coloured = W * (mag / np.abs(W[:mag.shape[0]]))

    # back to time domain
    coloured = np.fft.irfft(W_coloured, n=length).astype(np.float32)

    # normalise noise to unit std (so we can scale easily)
    coloured /= (np.std(coloured) + 1e-8)

    return coloured


def degrade_with_spectral_noise(clean_signal, noise_ref, noise_scale=1.0):
    """
    clean_signal: D1 (after your usual preprocessing)
    noise_ref:    e.g. D3 (after same preprocessing)
    noise_scale:  multiplier controlling SNR (bigger = noisier)
    """
    clean_signal = clean_signal.astype(np.float32).flatten()
    noise = make_spectral_noise_like(noise_ref, len(clean_signal))
    degraded = clean_signal + noise_scale * noise
    return degraded.astype(np.float32)
