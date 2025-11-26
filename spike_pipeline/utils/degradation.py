import numpy as np
from scipy.interpolate import interp1d


def make_spectral_noise_like(noise_ref, target_length):
    """
    Match the spectral envelope of noise_ref to generate
    coloured noise of length target_length.
    """

    # --- FFT of reference noise ---
    N_ref = len(noise_ref)
    mag_ref = np.abs(np.fft.rfft(noise_ref))
    freqs_ref = np.fft.rfftfreq(N_ref)

    # --- FFT bin frequencies for target length ---
    freqs_target = np.fft.rfftfreq(target_length)

    # --- interpolate magnitude to match target FFT resolution ---
    mag_interp = interp1d(freqs_ref, mag_ref,
                          kind='linear',
                          fill_value="extrapolate")(freqs_target)

    # --- generate white noise and colour it ---
    white = np.random.randn(target_length)
    W = np.fft.rfft(white)

    # Avoid division by zero
    W_norm = np.abs(W)
    W_norm[W_norm == 0] = 1e-12

    # Colour the spectrum
    W_coloured = W * (mag_interp / W_norm)

    # Return time-domain noise
    noise = np.fft.irfft(W_coloured, n=target_length)
    return noise


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
