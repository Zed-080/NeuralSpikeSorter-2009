import numpy as np
from spike_pipeline.data_loader import load_D1
from spike_pipeline.inference.detect_spikes import sliding_window_predict  # type: ignore
from spike_pipeline.inference.matching import match_predictions  # type: ignore


def tune_detector_threshold(detector_model,
                            D1_path="D1.mat",
                            threshold_range=np.linspace(0.90, 0.995, 6),
                            refractory_range=range(30, 61, 5)):

    d_norm, Index_gt, _ = load_D1(D1_path)

    best_f1 = -1
    best_threshold = None
    best_refractory = None

    for t in threshold_range:
        for r in refractory_range:

            preds = sliding_window_predict(
                detector_model,
                d_norm,
                threshold=t,
                refractory=r
            )

            f1 = match_predictions(preds, Index_gt, tolerance=50)["f1"]

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_refractory = r

            print(f"Threshold={t:.2f}, Refr={r}, F1={f1:.3f}")

    print("\n=== Best Settings ===")
    print(f"Threshold: {best_threshold}")
    print(f"Refractory: {best_refractory}")
    print(f"F1 Score: {best_f1:.3f}")

    return best_threshold, best_refractory
