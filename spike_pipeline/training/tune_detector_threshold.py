import numpy as np

from spike_pipeline.data_loader.load_datasets import load_D1, load_unlabelled
from spike_pipeline.inference.detect_spikes import sliding_window_predict
from spike_pipeline.inference.matching import match_predictions
from spike_pipeline.utils.degradation import degrade_with_spectral_noise


def evaluate_f1(detector_model, signal, Index_gt, thr, refr):
    """Runs detector → matches predictions → returns F1."""
    preds = sliding_window_predict(
        detector_model,
        signal,
        threshold=thr,
        refractory=refr
    )
    return match_predictions(preds, Index_gt, tolerance=50)["f1"]


def tune_detector_threshold(detector_model,
                            D1_path="D1.mat",
                            threshold_range=np.linspace(0.90, 0.995, 6),
                            refractory_range=range(30, 61, 5)):
    """
    Multi-SNR threshold tuning:
      Clean D1 (weight 0.4)
      D1 degraded with D3 noise (weight 0.3)
      D1 degraded with D5 noise (weight 0.3)
    """

    # -------------------------------------------------------
    # 1. Load clean D1 (ground truth used for all 3 variants)
    # -------------------------------------------------------
    d_clean, Index_gt, _ = load_D1(D1_path)

    # -------------------------------------------------------
    # 2. Load noise reference datasets (D3, D5)
    # -------------------------------------------------------
    d_D3 = load_unlabelled("D3.mat")
    d_D5 = load_unlabelled("D5.mat")

    # -------------------------------------------------------
    # 3. Create degraded versions of D1
    # -------------------------------------------------------
    d_D1_D3 = degrade_with_spectral_noise(d_clean, d_D3, noise_scale=1.0)
    d_D1_D5 = degrade_with_spectral_noise(d_clean, d_D5, noise_scale=1.5)

    print("=== Multi-SNR Threshold Tuning ===")
    print("Weights: Clean=0.4 | D3noise=0.3 | D5noise=0.3\n")

    # Best so far
    best_score = -1
    best_thr = None
    best_refr = None

    # -------------------------------------------------------
    # 4. Grid search
    # -------------------------------------------------------
    for thr in threshold_range:
        for refr in refractory_range:

            # F1 on clean
            f1_clean = evaluate_f1(
                detector_model, d_clean, Index_gt, thr, refr)

            # F1 on degraded versions
            f1_D3 = evaluate_f1(detector_model, d_D1_D3, Index_gt, thr, refr)
            f1_D5 = evaluate_f1(detector_model, d_D1_D5, Index_gt, thr, refr)

            # Weighted score (balanced)
            weighted_f1 = (
                0.4 * f1_clean +
                0.3 * f1_D3 +
                0.3 * f1_D5
            )

            print(
                f"Thr={thr:.3f}, Refr={refr} | "
                f"Clean={f1_clean:.3f}, D3n={f1_D3:.3f}, D5n={f1_D5:.3f} "
                f"=> Weighted={weighted_f1:.3f}"
            )

            # Update best
            if weighted_f1 > best_score:
                best_score = weighted_f1
                best_thr = thr
                best_refr = refr

    # -------------------------------------------------------
    # 5. Print best settings
    # -------------------------------------------------------
    print("\n=== Best Multi-SNR Settings ===")
    print(f"Threshold: {best_thr}")
    print(f"Refractory: {best_refr}")
    print(f"Weighted F1: {best_score:.3f}")

    return best_thr, best_refr
