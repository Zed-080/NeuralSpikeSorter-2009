import numpy as np


def match_predictions(predicted, ground_truth, tolerance=50):
    """
    Matches each predicted spike to nearest ground truth spike within ±tolerance.
    Computes precision, recall, F1.
    """
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)

    used_gt = set()
    tp = 0

    # match predicted spikes
    for p in predicted:
        diffs = np.abs(ground_truth - p)
        idx_min = np.argmin(diffs)

        if diffs[idx_min] <= tolerance:
            if idx_min not in used_gt:
                used_gt.add(idx_min)
                tp += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

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
