#fmt: off
from spike_pipeline.utils.degradation import degrade_with_spectral_noise as my_noise_match
import numpy as np
import sys
import os
import scipy.io as spio
from pathlib import Path

# --- 1. Setup Paths to Find 'src' ---
# We go up one level, then into 'Structured-CS-CW-with-good-feedback'
CURRENT_DIR = Path(__file__).resolve().parent
FRIEND_REPO = CURRENT_DIR.parent / "Structured-CS-CW-with-good-feedback"

if not FRIEND_REPO.exists():
    print(f"❌ ERROR: Could not find friend's repo at: {FRIEND_REPO}")
    print("Please check the folder name/location.")
    sys.exit(1)

sys.path.insert(0, str(FRIEND_REPO))

# Now we can import their module
try:
    from src.utils.noise_matcher import noise_match_d1 as friend_noise_match #type: ignore
    print(
        f"✅ Successfully imported 'src.utils.noise_matcher' from {FRIEND_REPO}")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print("Ensure 'Structured-CS-CW-with-good-feedback/src' exists and has an __init__.py")
    sys.exit(1)

# Import YOUR module

#fmt: on
# --- 2. Load Data ---
# We need to manually load D1 and a target (e.g., D2) to pass to your function.
# Your friend's function loads them internally, so we assume the data files are identical.
DATA_RAW = CURRENT_DIR / "data" / "raw"


def load_raw_d(name):
    path = DATA_RAW / f"{name}.mat"
    if not path.exists():
        # Fallback to friend's data path if yours is missing
        path = FRIEND_REPO / "data" / "raw" / f"{name}.mat"

    mat = spio.loadmat(str(path), squeeze_me=True)
    return mat["d"].astype(np.float32)


def main():
    print("\n--- Starting Verification ---")

    # Load raw arrays for YOUR implementation
    print("Loading raw D1 and D2...")
    try:
        d1_raw = load_raw_d("D1")
        d2_raw = load_raw_d("D2")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return

    # --- Run Friend's Code ---
    print("Running Friend's implementation (noise_match_d1('D2'))...")
    # Note: His code might look for data relative to HIS file location.
    # If it fails, ensure his repo has a data/raw folder populated.
    try:
        friend_result = friend_noise_match("D2")
    except Exception as e:
        print(f"❌ Friend's code failed to run: {e}")
        print("Note: His code loads data internally. Ensure his repo has data/raw/D1.mat and D2.mat")
        return

    # --- Run Your Code ---
    print("Running Your implementation (degrade_with_spectral_noise)...")
    # We pass the arrays explicitly
    my_result = my_noise_match(d1_raw, d2_raw, noise_scale=1.0)

    # --- Compare ---
    print("\n--- Comparison ---")
    if friend_result.shape != my_result.shape:
        print(
            f"❌ Shape Mismatch! Friend: {friend_result.shape}, Yours: {my_result.shape}")
        return

    diff = np.abs(friend_result - my_result)
    max_diff = np.max(diff)

    print(f"Max absolute difference: {max_diff:.8f}")

    # We allow a tiny tolerance for floating point associativity differences
    if np.allclose(friend_result, my_result, atol=1e-5):
        print("\n✅ SUCCESS: Arrays match identically (or within float precision)!")
    else:
        print("\n❌ FAILURE: Arrays do not match.")
        print("First 5 Friend:", friend_result[:5])
        print("First 5 Yours: ", my_result[:5])


if __name__ == "__main__":
    main()
