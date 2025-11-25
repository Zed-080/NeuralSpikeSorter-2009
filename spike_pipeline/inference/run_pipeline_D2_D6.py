import numpy as np
import scipy.io as spio
from spike_pipeline.data_loader import load_unlabelled
from spike_pipeline.inference.detect_spikes import sliding_window_predict
from spike_pipeline.inference.extract_waveforms import extract_waveform_64


def run_inference_dataset(detector_model,
                          classifier_model,
                          threshold,
                          refractory,
                          path,
                          save_path):

    # Load normalised signal
    d_norm = load_unlabelled(path)

    # ---------------------------
    # 1. Detect spike indices
    # ---------------------------
    detected = sliding_window_predict(
        detector_model,
        d_norm,
        threshold=threshold,
        refractory=refractory,
        window=128
    )

    # If no spikes, save empty output
    if len(detected) == 0:
        spio.savemat(save_path, {"Index": [], "Class": []})
        print(f"{path} → No spikes detected.")
        return

    # ---------------------------
    # 2. Extract ALL waveforms at once
    # ---------------------------
    X = extract_waveform_64(d_norm, detected)   # <--- FIXED

    # X must be shape (N, 64, 1)
    if X.ndim == 2:               # (N,64)
        X = X[:, :, np.newaxis]
    elif X.shape[1] == 1:         # (N,1,64)
        X = np.transpose(X, (0, 2, 1))

    # ---------------------------
    # 3. Classify ALL waveforms at once
    # ---------------------------
    probs = classifier_model.predict(X, verbose=0)
    pred_classes = np.argmax(probs, axis=1) + 1  # 1..5

    # ---------------------------
    # 4. Save output
    # ---------------------------
    spio.savemat(save_path, {
        "Index": np.array(detected, dtype=np.int64),
        "Class": pred_classes
    })

    # Print summary
    unique, counts = np.unique(pred_classes, return_counts=True)
    print(f"Class counts for {path}:")
    for cls in range(1, 6):
        print(
            f"  Class {cls}: {counts[unique.tolist().index(cls)] if cls in unique else 0} spikes")

    print(f"{path} → Saved {save_path} with {len(detected)} spikes.")
