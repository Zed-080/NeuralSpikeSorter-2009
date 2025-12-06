"""
    d1_match_d2d3d4.py

    Make D1 "look like" D2, D3 and D4 separately by:
    - bandpassing D1 and each target (3–3000 Hz)
    - adding Gaussian noise to D1 at a base SNR
    - scaling that noise so D1's noise std matches the target's
    - bandpassing again
    Then plot 3 pairs: (D1->D2 vs D2), (D1->D3 vs D3), (D1->D4 vs D4).

    Run from project root:

        (.venv_tf) python experiments/d1_match_d2d3d4.py
    """

    #fmt:off
    import sys
    import os

    # --- FIX IMPORT PATHS ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from pathlib import Path

    import numpy as np
    import scipy.io as spio
    from scipy.signal import butter, filtfilt
    import matplotlib.pyplot as plt

    # ----------------- CONFIG -----------------
    #fmt: on

    # PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT

    FS = 24000.0      # adjust to your sampling rate if different
    BP_LOW = 7.0
    BP_HIGH = 3000.0

    N_VIS = 5000
    RNG_SEED = 42

    BASE_SNR_DB = 10.0   # base SNR before scaling to match each target

    # ----------------- HELPERS -----------------

    def load_trace(name: str, key: str = "d") -> np.ndarray:
        path = f"{DATA_DIR}/{name}.mat"
        mat = spio.loadmat(path, squeeze_me=True)
        d = np.asarray(mat[key], dtype=np.float32)
        return d.reshape(-1)

    def butter_bandpass_filter(x: np.ndarray,
                               lowcut: float,
                               highcut: float,
                               fs: float,
                               order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, x).astype(np.float32)

    def gaus_snr(x: np.ndarray,
                 target_snr_db: float,
                 rng: np.random.Generator) -> np.ndarray:
        """Add Gaussian noise to reach a target SNR (dB)."""
        x = x.astype(np.float32)
        sig_power = np.mean(x ** 2)
        if sig_power <= 1e-12:
            return x.copy()
        snr_lin = 10.0 ** (target_snr_db / 10.0)
        noise_power = sig_power / snr_lin
        noise_std = np.sqrt(noise_power)
        noise = rng.normal(0.0, noise_std, size=x.shape)
        return (x + noise).astype(np.float32)

    def estimate_noise_std(x: np.ndarray, spike_thresh: float = 3.0) -> float:
        """
        Rough noise estimate: std of points where |x| < spike_thresh.
        Tries to ignore spike peaks.
        """
        mask = np.abs(x) < spike_thresh
        if mask.sum() < 100:
            return float(np.std(x))
        return float(np.std(x[mask]))

    # ----------------- MATCHING (EXACT PORT OF noise_matcher.py) -----------------

    BP_LOW = 7.0
    BP_HIGH = 3000.0
    BASE_SNR_DB = 5.0

    def bandpass(x):
        nyq = FS*0.5
        b, a = butter(4, [BP_LOW/nyq, BP_HIGH/nyq], btype="band")
        return filtfilt(b, a, x).astype(np.float32)

    def make_d1_look_like_target_exact(tgt_name, rng):
        """
        IDENTICAL TO noise_matcher.py behavior.

        Returns:
            d1_matched, tgt_bandpassed
        """

        # load raw datasets
        d1 = load_trace("D1")
        tgt = load_trace(tgt_name)

        # bandpass target only
        tgt_bp = bandpass(tgt)

        # base gaussian noise injection into RAW D1
        d1_base = gaus_snr(d1, BASE_SNR_DB, rng)

        # estimate noise level from bandpassed target
        tgt_std = estimate_noise_std(tgt_bp)

        # isolate synthetic noise
        added = d1_base - d1

        added_std = np.std(added)
        if added_std < 1e-8:
            added_std = 1e-8

        # scale noise to target noise level
        scaled = d1 + added * (tgt_std / added_std)

        # bandpass AFTER matching (this matches reference pipeline)
        d1_matched = bandpass(scaled)

        return d1_matched, tgt_bp

    # ----------------- MAIN -----------------

    def main():
        rng = np.random.default_rng(RNG_SEED)

        # Load & bandpass D1 once (shared source for all matches)
        print(f"Loading D1 from {DATA_DIR}/D1.mat")
        d1_raw = load_trace("D1")
        d1_bp = butter_bandpass_filter(d1_raw, BP_LOW, BP_HIGH, FS)
        print("D1 bandpass 3–3000 Hz done.")

        # Make D1 look like D2, D3, D4
        d1_to_d2, d2_bp = make_d1_look_like_target_exact("D2", rng)
        d1_to_d3, d3_bp = make_d1_look_like_target_exact("D3", rng)
        d1_to_d4, d4_bp = make_d1_look_like_target_exact("D4", rng)

        # Plot three pairs: (D1->D2 vs D2), (D1->D3 vs D3), (D1->D4 vs D4)
        n_vis = min(
            N_VIS,
            len(d1_to_d2), len(d2_bp),
            len(d1_to_d3), len(d3_bp),
            len(d1_to_d4), len(d4_bp)
        )
        t = np.arange(n_vis) / FS

        plt.figure(figsize=(12, 10))

        # Row 1: D1->D2 vs D2
        plt.subplot(3, 2, 1)
        plt.plot(t, d1_to_d2[:n_vis])
        plt.title("D1 → noise-matched to D2")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(3, 2, 2)
        plt.plot(t, d2_bp[:n_vis])
        plt.title("D2 (bandpass 3–3000 Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Row 2: D1->D3 vs D3
        plt.subplot(3, 2, 3)
        plt.plot(t, d1_to_d3[:n_vis])
        plt.title("D1 → noise-matched to D3")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(3, 2, 4)
        plt.plot(t, d3_bp[:n_vis])
        plt.title("D3 (bandpass 3–3000 Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Row 3: D1->D4 vs D4
        plt.subplot(3, 2, 5)
        plt.plot(t, d1_to_d4[:n_vis])
        plt.title("D1 → noise-matched to D4")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(3, 2, 6)
        plt.plot(t, d4_bp[:n_vis])
        plt.title("D4 (bandpass 3–3000 Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        out_path = f"{PROJECT_ROOT}/visualisations/Images/d1_match_d2d3d4.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"\nSaved comparison plot to: {out_path}")
        plt.show()

    if __name__ == "__main__":
        main()
