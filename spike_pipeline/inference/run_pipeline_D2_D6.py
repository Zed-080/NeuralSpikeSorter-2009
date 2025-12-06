import numpy as np
import scipy.io as spio
from pathlib import Path

from spike_pipeline.inference.extract_waveforms import extract_waveform_64
from spike_pipeline.denoise.wavelet_denoise import denoise_dataset

# --- CONFIGURATION ---
WINDOW_LEN_DET = 120  # Detector window length
DET_LABEL_TOL = 5

# --- MANUAL OVERRIDES ---
# If a dataset is listed here, this value is FORCED.
# If you comment these out, the pipeline uses the 'default_threshold' passed in (from tuning).
# THRESHOLD_PER_DATASET = {
#     "D2": 0.30,
#     "D3": 0.45,
#     "D4": 0.45,
#     "D5": 0.50,
#     "D6": 0.20,
# }
THRESHOLD_PER_DATASET = {
    "D2": 0.70,
    "D3": 0.75,
    "D4": 0.80,
    "D5": 0.85,
    "D6": 0.90,
}


def sliding_window_probs(d_norm, model, window_len=128, batch_size=2048):
    d_norm = np.asarray(d_norm, dtype=np.float32)
    N = d_norm.shape[0]
    starts = np.arange(0, N - window_len + 1, dtype=np.int64)
    probs = np.empty((len(starts), window_len), dtype=np.float32)

    for i in range(0, len(starts), batch_size):
        batch_indices = starts[i:i + batch_size]
        # Extract windows: (B, 128)
        batch_windows = np.stack([d_norm[s:s + window_len]
                                 for s in batch_indices])
        # Add channel dimension: (B, 128, 1)
        batch_X = batch_windows[..., np.newaxis]

        p = model.predict(batch_X, verbose=0)
        probs[i:i + len(batch_indices), :] = np.squeeze(p, axis=-1)

    return probs, starts


def apply_refractory(probs, starts, decision_threshold, refractory_suppression, center_offset=None):
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
        if idx - last_det >= refractory_suppression:
            det_indices.append(idx)
            last_det = idx

    return np.array(det_indices, dtype=np.int64)


def run_inference_dataset(detector_model,
                          classifier_model,
                          path,
                          save_path,
                          refractory=45,
                          default_threshold=0.75):  # <--- CRITICAL FIX HERE

    path = Path(path)
    dataset_name = path.stem
    real_name = dataset_name.split("_")[0]

    print(f"\n=== Processing {dataset_name} ===")

    # 1. Denoise
    print(f"Denoising {real_name}...")
    _, d_denoised = denoise_dataset(
        real_name, save=False, use_matched_filter=True)
    d = d_denoised.astype(np.float32)

    # 2. Normalize
    mean = d.mean()
    std = d.std()
    d_norm = (d - mean) / (std + 1e-8)

    # 3. SELECT THRESHOLD (The Logic)
    # Check dictionary first; if not found, use the default_threshold passed in.
    thr = THRESHOLD_PER_DATASET.get(real_name, default_threshold)

    source = "Manual Override" if real_name in THRESHOLD_PER_DATASET else "Auto-Tuned"
    print(f"Using threshold: {thr:.3f} ({source})")

    # 4. Detect
    print("Running sliding window detector...")
    probs, starts = sliding_window_probs(
        d_norm, detector_model, window_len=WINDOW_LEN_DET)

    detected = apply_refractory(
        probs=probs,
        starts=starts,
        decision_threshold=thr,
        refractory_suppression=refractory,
        center_offset=DET_LABEL_TOL
    )

    if len(detected) == 0:
        print("No spikes detected.")
        spio.savemat(save_path, {"Index": [], "Class": []})
        return

    # 5. Extract & Classify
    X = extract_waveform_64(d_norm, detected)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    if len(X) > 0:
        print("Classifying...")
        probs_clf = classifier_model.predict(X, verbose=0)
        pred_classes = np.argmax(probs_clf, axis=1) + 1
    else:
        pred_classes = []

    # 6. Save
    spio.savemat(save_path, {
        "Index": np.array(detected, dtype=np.int64),
        "Class": np.array(pred_classes, dtype=np.int64)
    })
    print(f"Saved {len(detected)} spikes to {save_path}")
