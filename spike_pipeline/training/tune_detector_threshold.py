import numpy as np
from spike_pipeline.data_loader.load_datasets import load_D1, load_unlabelled
from spike_pipeline.utils.degradation import degrade_with_spectral_noise
from spike_pipeline.inference.matching import match_predictions


# -----------------------------------------------------------
# 1. OPTIMIZED HELPERS (Splitting Prediction from Thresholding)
# -----------------------------------------------------------
def get_model_probabilities(model, d_norm, window=128):
    """
    Run the heavy CNN inference ONCE to get raw probabilities.
    """
    N = len(d_norm)
    # Build all windows (vectorized)
    # Note: For very large signals, you might need batching,
    # but for D1 (1.44M samples) this usually fits in Colab RAM.
    X = np.zeros((N - window, window, 1), dtype=np.float32)
    for i in range(N - window):
        X[i, :, 0] = d_norm[i:i+window]

    # Predict
    probs = model.predict(X, verbose=0).flatten()
    return probs


def predict_from_probs(probs, threshold, refractory):
    """
    Apply threshold and refractory to PRE-CALCULATED probabilities.
    This is instant (no CNN inference).
    """
    # 1. Threshold
    # specific indices where prob > threshold
    candidate_indices = np.where(probs >= threshold)[0]

    # 2. Refractory (Greedy suppression)
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int64)

    kept_indices = [candidate_indices[0]]
    for idx in candidate_indices[1:]:
        if idx - kept_indices[-1] >= refractory:
            kept_indices.append(idx)

    return np.array(kept_indices, dtype=np.int64)


# -----------------------------------------------------------
# 2. MAIN TUNING FUNCTION
# -----------------------------------------------------------
def tune_detector_threshold(detector_model,
                            D1_path="D1.mat",
                            threshold_range=np.linspace(0.5, 0.99, 10),
                            refractory_range=range(30, 61, 15)):

    # --- A. Setup Datasets ---
    print("Preparing datasets for tuning...")
    d_clean, Index_gt, _ = load_D1(D1_path)
    d_D3 = load_unlabelled("D3.mat")
    d_D5 = load_unlabelled("D5.mat")

    # Create degraded versions
    d_D1_D3 = degrade_with_spectral_noise(d_clean, d_D3, noise_scale=1.0)
    d_D1_D5 = degrade_with_spectral_noise(d_clean, d_D5, noise_scale=1.5)

    # --- B. PRE-CALCULATE PROBABILITIES (The Speed Fix) ---
    print("\nPre-calculating CNN probabilities (running inference 3 times)...")

    print("  1/3: Clean D1...")
    probs_clean = get_model_probabilities(detector_model, d_clean)

    print("  2/3: D1 + D3 Noise...")
    probs_D3 = get_model_probabilities(detector_model, d_D1_D3)

    print("  3/3: D1 + D5 Noise...")
    probs_D5 = get_model_probabilities(detector_model, d_D1_D5)

    print("Done! Now starting grid search (instant)...")
    print("Weights: Clean=0.4 | D3noise=0.3 | D5noise=0.3\n")

    best_score = -1
    best_thr = None
    best_refr = None

    # --- C. Fast Grid Search ---
    # Now we just loop over math, not model inference
    for thr in threshold_range:
        for refr in refractory_range:

            # 1. Clean Score
            preds = predict_from_probs(probs_clean, thr, refr)
            f1_clean = match_predictions(preds, Index_gt, tolerance=50)["f1"]

            # 2. D3 Noise Score
            preds = predict_from_probs(probs_D3, thr, refr)
            f1_D3 = match_predictions(preds, Index_gt, tolerance=50)["f1"]

            # 3. D5 Noise Score
            preds = predict_from_probs(probs_D5, thr, refr)
            f1_D5 = match_predictions(preds, Index_gt, tolerance=50)["f1"]

            # Weighted Average
            weighted_f1 = (0.4 * f1_clean + 0.3 * f1_D3 + 0.3 * f1_D5)

            print(f"Thr={thr:.3f}, Refr={refr} | F1_Mixed={weighted_f1:.3f}")

            if weighted_f1 > best_score:
                best_score = weighted_f1
                best_thr = thr
                best_refr = refr

    print("\n=== Best Multi-SNR Settings ===")
    print(f"Threshold: {best_thr}")
    print(f"Refractory: {best_refr}")
    print(f"Weighted F1: {best_score:.3f}")

    return best_thr, best_refr
