import numpy as np
from spike_pipeline.data_loader import load_D1, global_normalize
from spike_pipeline.inference.detect_spikes import sliding_window_predict
from spike_pipeline.inference.extract_waveforms import extract_waveform_64
from spike_pipeline.inference.matching import match_predictions


def run_D1_selfcheck(detector_model,
                     classifier_model,
                     threshold,
                     refractory,
                     path="D1.mat"):

    d_norm, Index_gt, Class_gt = load_D1(path)

    print("Running detector on D1...")
    detected_indices = sliding_window_predict(
        detector_model,
        d_norm,
        threshold=threshold,
        refractory=refractory,
        window=128
    )

    print(f"Detected spikes: {len(detected_indices)}")

    # match spike positions
    metrics = match_predictions(detected_indices, Index_gt, tolerance=50)
    print("Detector matching:", metrics)

    # extract waveforms for classification
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
    pred_classes = np.argmax(probs, axis=1) + 1  # convert 0..4 → 1..5

    # match class predictions to GT (optional)
    return detected_indices, pred_classes, metrics
