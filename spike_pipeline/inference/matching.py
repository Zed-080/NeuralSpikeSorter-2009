# ==============================================================================
# PERFORMANCE EVALUATION METRICS
# ==============================================================================
# Utilities for comparing predicted spikes against Ground Truth (D1).
#
# 1. MATCHING LOGIC
#    - Nearest Neighbor: For each predicted spike, we find the closest ground truth spike.
#    - Tolerance: A match is only valid if the distance is within +/- 50 samples.
#    - Uniqueness: Each ground truth spike can be matched at most once.
#
# 2. METRICS
#    - Precision: How many of our predictions were actual spikes? (TP / (TP + FP))
#    - Recall: How many of the real spikes did we find? (TP / (TP + FN))
#    - F1 Score: The harmonic mean of Precision and Recall. This is the primary
#      metric used for coursework marking.
# ==============================================================================

import numpy as np


def match_predictions(predicted, ground_truth, tolerance=50):
    """
    Matches predicted spikes to ground truth within a tolerance window.
    Returns a dictionary containing TP, FP, FN, Precision, Recall, and F1 Score.
    """
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)

    used_gt = set()
    tp = 0

    # Greedy matching of predicted spikes to nearest GT
    for p in predicted:
        diffs = np.abs(ground_truth - p)
        idx_min = np.argmin(diffs)

        if diffs[idx_min] <= tolerance:
            if idx_min not in used_gt:
                used_gt.add(idx_min)
                tp += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    # Add epsilon to avoid division by zero
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
