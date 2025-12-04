import numpy as np
import scipy.io as spio
from pathlib import Path

from spike_pipeline.inference.extract_waveforms import extract_waveform_64
from spike_pipeline.denoise.wavelet_denoise import denoise_dataset

# --- CONFIGURATION ---
WINDOW_LEN_DET = 128    # Match the trained model (128)
DET_LABEL_TOL = 5       # Alignment tolerance

# Specific thresholds per dataset
THRESHOLD_PER_DATASET = {
    "D2": 0.70,
    "D3": 0.75,
    "D4": 0.80,
    "D5": 0.85,
    "D6": 0.90,
}


def sliding_window_probs(d_norm, model, window_len=WINDOW_LEN_DET, batch_size=2048):
    """
    Computes per-timestep probabilities using the sequence-labeling model.
    """
    d_norm = np.asarray(d_norm, dtype=np.float32)
    N = d_norm.shape[0]

    # Generate start indices for all possible windows
    starts = np.arange(0, N - window_len + 1, dtype=np.int64)
    num_windows = starts.shape[0]

    probs = np.empty((num_windows, window_len), dtype=np.float32)

    for i in range(0, num_windows, batch_size):
        batch_indices = starts[i:i + batch_size]
        B = batch_indices.shape[0]

        # Extract windows: (B, 128)
        batch_windows = np.stack([d_norm[s:s + window_len]
                                 for s in batch_indices])

        # Add channel dimension: (B, 128, 1)
        batch_X = batch_windows[..., np.newaxis]

        # Predict -> Output is (B, 128, 1)
        p = model.predict(batch_X, verbose=0)

        # Squeeze back to (B, 128)
        probs[i:i + B, :] = np.squeeze(p, axis=-1)

    return probs, starts


def apply_refractory(probs, starts, decision_threshold, refractory_suppression, center_offset=None):
    """
    Selects spikes based on the center pixel of the sliding window.
    """
    if center_offset is None:
        center_offset = probs.shape[1] // 2

    # Look at the probability at the "spike index" (index 5) of the window
    center_probs = probs[:, center_offset]
    cand_windows = np.where(center_probs >= decision_threshold)[0]

    if cand_windows.size == 0:
        return np.array([], dtype=np.int64)

    det_indices = []
    last_det = -np.inf

    for w_idx in cand_windows:
        # Map window index back to sample index in the original signal
        idx = int(starts[w_idx] + center_offset)

        if idx - last_det < refractory_suppression:
            continue

        det_indices.append(idx)
        last_det = idx

    return np.array(det_indices, dtype=np.int64)


def run_inference_dataset(detector_model,
                          classifier_model,
                          path,
                          save_path,
                          refractory=45):

    path = Path(path)
    dataset_name = path.stem  # "D2"
    real_name = dataset_name.split("_")[0]  # Handle "D2_test" -> "D2"

    print(f"\n=== Processing {dataset_name} ===")

    # 1. PREPROCESSING (Matched Filter + Denoising)
    #    We ENABLE use_matched_filter because we retrained the model with it.
    print(f"Denoising {real_name}...")
    _, d_denoised = denoise_dataset(
        real_name, save=False, use_matched_filter=True)

    # 2. NORMALISATION
    d = d_denoised.astype(np.float32)
    mean = d.mean()
    std = d.std()
    d_norm = (d - mean) / (std + 1e-8)

    # 3. SELECT THRESHOLD
    thr = THRESHOLD_PER_DATASET.get(real_name, 0.75)
    print(f"Using threshold: {thr} for {dataset_name}")

    # 4. DETECTOR INFERENCE
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

    # 5. EXTRACT WAVEFORMS
    X = extract_waveform_64(d_norm, detected)

    # Reshape for classifier (N, 64, 1)
    if X.ndim == 2:
        X = X[..., np.newaxis]

    # 6. CLASSIFY
    if len(X) > 0:
        print("Classifying...")
        probs_clf = classifier_model.predict(X, verbose=0)
        pred_classes = np.argmax(probs_clf, axis=1) + 1  # 1..5
    else:
        pred_classes = []

    # 7. SAVE
    spio.savemat(save_path, {
        "Index": np.array(detected, dtype=np.int64),
        "Class": np.array(pred_classes, dtype=np.int64)
    })

    print(f"Saved {len(detected)} spikes to {save_path}")
