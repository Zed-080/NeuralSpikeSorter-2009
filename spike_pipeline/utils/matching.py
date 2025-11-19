import numpy as np


def match_nearest(predicted, ground_truth, tolerance=50):
    """
    Count true positives based on nearest-neighbour match within tolerance.
    """
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)

    used_gt = set()
    tp = 0

    for p in predicted:
        diffs = np.abs(ground_truth - p)

        i_min = np.argmin(diffs)
        if diffs[i_min] <= tolerance:
            if i_min not in used_gt:
                used_gt.add(i_min)
                tp += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
