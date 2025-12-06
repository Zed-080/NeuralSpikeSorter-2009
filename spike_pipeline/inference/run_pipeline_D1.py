import numpy as np
from pathlib import Path  # <--- Need this
from spike_pipeline.data_loader import load_D1, global_normalize
from spike_pipeline.inference.detect_spikes import sliding_window_predict
from spike_pipeline.inference.extract_waveforms import extract_waveform_64
from spike_pipeline.inference.matching import match_predictions

# --- FIX PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data"


def run_D1_selfcheck(detector_model,
                     classifier_model,
                     threshold,
                     refractory,
                     path="D1.mat"):  # This string is just the filename now

    # Construct full path
    full_path = DATA_RAW / path

    if not full_path.exists():
        print(f"Error: Could not find {full_path}")
        return

    d_norm, Index_gt, Class_gt = load_D1(full_path)

    print("Running detector on D1...")
    detected_indices = sliding_window_predict(
        detector_model,
        d_norm,
        threshold=threshold,
        refractory=refractory,
        window=128
    )

    print(f"Detected spikes: {len(detected_indices)}")

    metrics = match_predictions(detected_indices, Index_gt, tolerance=50)
    print("Detector matching:", metrics)

    X = []
    kept_indices = []

    for idx in detected_indices:
        w = extract_waveform_64(d_norm, idx)
        if w is not None:
            X.append(w)
            kept_indices.append(idx)

    if len(X) == 0:
        print("No valid waveforms extracted.")
        return

    X = np.array(X)[..., np.newaxis]

    print("Running classifier...")
    probs = classifier_model.predict(X, verbose=0)
    pred_classes = np.argmax(probs, axis=1) + 1

    return detected_indices, pred_classes, metrics
