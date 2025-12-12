# ==============================================================================
# TRAINING DATA GENERATION SCRIPT
# ==============================================================================
# The master orchestration script that converts raw MATLAB files into
# ready-to-train NumPy arrays (X_train, y_train).
#
# 1. NOISE MATCHING (DATA SYNTHESIS)
#    - To make the model robust to the unseen D2-D6 datasets, we do not just
#      train on Clean D1.
#    - We synthesize 5 new datasets by taking Clean D1 and injecting colored
#      noise spectrally matched to the background noise of D2-D6.
#
# 2. PIPELINE EXECUTION
#    - Loads Raw D1 -> Normalizes -> Generates Noisy Versions.
#    - Extract Windows -> Detector Data (Positive/Negative windows).
#    - Extract Waveforms -> Classifier Data (Short 64-sample clips).
#
# 3. OUTPUT
#    - Saves concatenated .npy files to 'outputs/', combining data from all
#      noise levels into a single large training set.
# ==============================================================================

import numpy as np
import os
from pathlib import Path

# Pipeline Modules
from spike_pipeline.data_loader.load_datasets import load_mat, global_normalize
from spike_pipeline.utils.degradation import degrade_with_spectral_noise
from spike_pipeline.data_loader.build_detector_data import build_detector_data
from spike_pipeline.data_loader.build_classifier_data import build_classifier_data

# --- Paths Config ---
# robustly locate project root regardless of where script is run
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
D1_PATH = DATA_RAW / "D1.mat"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """
    The master orchestration script that converts raw MATLAB files into 
    ready-to-train NumPy arrays (X_train, y_train).
    """
    print(f"Loading Raw D1 from {D1_PATH}...")

    # 1. Load Ground Truth Data (D1)
    if not D1_PATH.exists():
        print(f"\n[ERROR] Could not find {D1_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please ensure the 'data' folder exists at: {DATA_RAW}\n")
        return

    # Load raw signal + ground truth labels
    d1_raw, spike_idx, spike_class = load_mat(D1_PATH)

    # 2. Initialize Training Versions
    print("Preparing Clean D1...")
    d_versions = [("Clean", global_normalize(d1_raw))]

    target_datasets = ["D2", "D3", "D4", "D5", "D6"]

    # 3. Generate Noise-Matched Versions
    # We create 5 additional synthetic datasets. Each one takes the clean spikes from D1
    # and buries them in noise that spectrally mimics a specific target dataset (D2-D6).
    for target in target_datasets:
        tgt_path = DATA_RAW / f"{target}.mat"

        if not tgt_path.exists():
            print(f"Skipping {target} (file not found at {tgt_path})")
            continue

        print(f"Generating noise-matched version: D1 -> {target}...")

        # A. Load Raw Target (to analyze its noise profile)
        d_tgt_raw, _, _ = load_mat(tgt_path)

        # B. Degrade D1
        d_noisy = degrade_with_spectral_noise(
            d1_raw, d_tgt_raw, noise_scale=1.0)

        # C. Normalize (Standardize variance after noise injection)
        d_noisy_norm = global_normalize(d_noisy)

        d_versions.append((f"Match_{target}", d_noisy_norm))

    # 4. Build Training Windows
    # Iterate through all versions (Clean + 5 Noisy) and extract training samples.
    print("\nExtracting windows for Detector and Classifier...")

    X_det_list, y_det_list = [], []
    X_clf_list, y_clf_list, y_clf_raw_list = [], [], []

    for name, d_signal in d_versions:
        print(f"Processing version: {name}...")

        # Build Detector Data (120-sample windows, Pos vs Neg)
        X_d, y_d = build_detector_data(d_signal, spike_idx)

        # Build Classifier Data (64-sample centered waveforms, 5 classes)
        # Note: We pass spike_class here to generate the correct classification labels
        X_c, y_c, y_c_raw = build_classifier_data(
            d_signal, spike_idx, spike_class)

        # Accumulate results
        X_det_list.append(X_d)
        y_det_list.append(y_d)
        X_clf_list.append(X_c)
        y_clf_list.append(y_c)
        y_clf_raw_list.append(y_c_raw)

    # 5. Concatenate and Save
    # Merge all lists into massive NumPy arrays for efficient training
    print("\nConcatenating and Saving...")

    if not X_det_list:
        print(
            "[ERROR] No data was generated. Check if D1.mat and target datasets exist.")
        return

    # Stack along batch dimension (axis 0)
    X_detector = np.concatenate(X_det_list, axis=0)
    y_detector = np.concatenate(y_det_list, axis=0)

    X_classifier = np.concatenate(X_clf_list, axis=0)
    y_classifier = np.concatenate(y_clf_list, axis=0)
    y_classifier_raw = np.concatenate(y_clf_raw_list, axis=0)

    print(f"Detector Data: {X_detector.shape}")
    print(f"Classifier Data: {X_classifier.shape}")

    # Save to disk
    np.save(OUTPUT_DIR / "X_detector.npy", X_detector)
    np.save(OUTPUT_DIR / "y_detector.npy", y_detector)
    np.save(OUTPUT_DIR / "X_classifier.npy", X_classifier)
    np.save(OUTPUT_DIR / "y_classifier.npy", y_classifier)
    np.save(OUTPUT_DIR / "y_classifier_raw.npy", y_classifier_raw)

    print(f"Done! Datasets saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
