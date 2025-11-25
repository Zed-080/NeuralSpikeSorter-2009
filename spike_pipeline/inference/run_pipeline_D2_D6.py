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

    d_norm = load_unlabelled(path)

    detected = sliding_window_predict(
        detector_model,
        d_norm,
        threshold=threshold,
        refractory=refractory,
        window=128
    )

    X = []
    final_indices = []

    for idx in detected:
        w = extract_waveform_64(d_norm, idx)
        if w is not None:
            X.append(w)
            final_indices.append(idx)

    if len(X) == 0:
        spio.savemat(save_path, {"Index": [], "Class": []})
        return

    X = np.array(X)[..., np.newaxis]

    probs = classifier_model.predict(X, verbose=0)
    pred_classes = np.argmax(probs, axis=1) + 1  # convert 0..4 → 1..5

    # Count spikes per class
    unique, counts = np.unique(pred_classes, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # Print detection summary
    print(f"Class counts for {path}:")
    for cls in range(1, 6):
        print(f"  Class {cls}: {class_counts.get(cls, 0)} spikes")

    spio.savemat(save_path, {
        "Index": np.array(final_indices, dtype=np.int64),
        "Class": pred_classes
    })

    print(f"{path} → Saved {save_path} with {len(final_indices)} spikes.")
