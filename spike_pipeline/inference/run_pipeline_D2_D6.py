# ==============================================================================
# FULL INFERENCE PIPELINE (D2-D6)
# ==============================================================================
# The main orchestration script for processing unlabelled datasets.
# Generates the final submission .mat files.
#
# PIPELINE STAGES:
# 1. DENOISING: Applies Wavelet Denoising (and optional Matched Filtering)
#    to clean the raw noisy signal.
# 2. NORMALIZATION: Standardizes the signal to unit variance (Z-score).
# 3. DETECTION:
#    - Runs the Detector CNN (sliding window) to get probability maps.
#    - Applies thresholding (auto-tuned or manual override).
#    - Applies refractory suppression to remove duplicate detections.
# 4. EXTRACTION: Cuts out 64-sample waveforms around valid detections.
# 5. CLASSIFICATION: Runs the Classifier CNN to assign neuron types (1-5).
# 6. EXPORT: Saves 'Index' and 'Class' vectors to .mat files (1-based indexing).
# ==============================================================================

import numpy as np
import scipy.io as spio
from pathlib import Path

from spike_pipeline.inference.extract_waveforms import extract_waveform_64
from spike_pipeline.denoise.wavelet_denoise import denoise_dataset

# --- CONFIGURATION ---
WINDOW_LEN_DET = 120
DET_LABEL_TOL = 5

# Hardcoded waveform parameters to match extract_waveforms.py
PRE = 20
POST = 44

# Thresholds (Tuned or Manual)
# Higher noise (D5/D6) typically requires higher thresholds to reduce false positives
THRESHOLD_PER_DATASET = {
    "D2": 0.25,
    "D3": 0.40,
    "D4": 0.50,
    "D5": 0.75,
    "D6": 0.90,
}


def sliding_window_probs(d_norm, model, window_len=120, batch_size=2048):
    """
    Runs the Detector CNN over the signal in batches to generate a probability curve.
    Returns: probs (N, window), starts (N,)
    """
    d_norm = np.asarray(d_norm, dtype=np.float32)
    N = d_norm.shape[0]
    starts = np.arange(0, N - window_len + 1, dtype=np.int64)
    probs = np.empty((len(starts), window_len), dtype=np.float32)

    for i in range(0, len(starts), batch_size):
        batch_indices = starts[i:i + batch_size]
        batch_windows = np.stack([d_norm[s:s + window_len]
                                 for s in batch_indices])
        batch_X = batch_windows[..., np.newaxis]
        p = model.predict(batch_X, verbose=0)
        probs[i:i + len(batch_indices), :] = np.squeeze(p, axis=-1)

    return probs, starts


def apply_refractory(probs, starts, decision_threshold, refractory_suppression, center_offset=None):
    """
    Post-processes probabilities to find spike indices.
    Applies thresholding and refractory period to merge close detections.
    """
    if center_offset is None:
        center_offset = probs.shape[1] // 2

    center_probs = probs[:, center_offset]
    cand_windows = np.where(center_probs >= decision_threshold)[0]

    if cand_windows.size == 0:
        return np.array([], dtype=np.int64)

    det_indices = []
    last_det = -np.inf

    for w_idx in cand_windows:
        idx = int(starts[w_idx] + center_offset)
        # Refractory check: ignore if too close to previous spike
        if idx - last_det >= refractory_suppression:
            det_indices.append(idx)
            last_det = idx

    return np.array(det_indices, dtype=np.int64)


def run_inference_dataset(detector_model,
                          classifier_model,
                          path,
                          save_path,
                          refractory=45,
                          default_threshold=0.75):
    """
    Executes the full pipeline on a single .mat dataset file.
    Loads -> Denoises -> Detects -> Classifies -> Saves.
    """
    path = Path(path)
    dataset_name = path.stem
    real_name = dataset_name.split("_")[0]

    print(f"\n{'='*60}")
    print(f"PROCESSING {real_name}")
    print(f"{'='*60}")

    # 1. Denoise
    # Note: denoise_dataset prints its own status messages
    _, d_denoised = denoise_dataset(
        real_name, save=False, use_matched_filter=True)
    d = d_denoised.astype(np.float32)

    # 2. Normalize
    mean = d.mean()
    std = d.std()
    d_norm = (d - mean) / (std + 1e-8)

    # 3. Select Threshold
    thr = THRESHOLD_PER_DATASET.get(real_name, default_threshold)
    source = "Manual Override" if real_name in THRESHOLD_PER_DATASET else "Auto-Tuned"
    print(f"Threshold: {thr:.3f} ({source}) | Refractory: {refractory}")

    # 4. Detect
    print("Running detector...")
    probs, starts = sliding_window_probs(
        d_norm, detector_model, window_len=WINDOW_LEN_DET)

    detected = apply_refractory(
        probs=probs,
        starts=starts,
        decision_threshold=thr,
        refractory_suppression=refractory,
        center_offset=DET_LABEL_TOL
    )

    print(f"Detected spikes (raw): {len(detected)}")

    if len(detected) == 0:
        print("No spikes detected.")
        spio.savemat(save_path, {"Index": [], "Class": []})
        return

    # 5. Extract & Classify (WITH BOUNDS CHECKING FIX)
    N = len(d_norm)
    valid_indices = []
    valid_waveforms = []

    # Filter out spikes that are too close to the start or end
    for idx in detected:
        start = idx - PRE
        end = idx + POST

        # Check bounds (must match extract_waveform_64 logic exactly)
        if start >= 0 and end <= N:
            w = d_norm[start:end]
            if len(w) == (PRE + POST):  # Ensure length is exactly 64
                valid_indices.append(idx)
                valid_waveforms.append(w)

    # Convert to arrays
    valid_indices = np.array(valid_indices, dtype=np.int64)
    X = np.array(valid_waveforms, dtype=np.float32)

    # Add channel dimension if needed
    if X.ndim == 2:
        X = X[..., np.newaxis]

    print(
        f"Valid waveforms extracted: {len(X)} (dropped {len(detected) - len(X)} edge spikes)")

    if len(X) > 0:
        print(f"Classifying {len(X)} waveforms...")
        probs_clf = classifier_model.predict(X, verbose=0)
        pred_classes_0_4 = np.argmax(probs_clf, axis=1)
        pred_classes = pred_classes_0_4 + 1  # Convert to 1-5 class labels

        counts = np.bincount(pred_classes, minlength=6)[1:]
        print(f"Class distribution: {counts}")
    else:
        print("No valid waveforms extracted.")
        pred_classes = []

    # 6. Save (FIXED: Using valid_indices and adding 1 for MATLAB format)
    spio.savemat(save_path, {
        "Index": valid_indices + 1,  # Convert 0-based to 1-based Index
        "Class": np.array(pred_classes, dtype=np.int64)
    })
    print(f"Saved to: {save_path}")
