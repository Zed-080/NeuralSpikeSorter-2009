# This script is designed to visualize the effects of the detector threshold
# on spike identification performance using the labelled D1 dataset.
# The goal is to find an optimal balance between Precision and Recall.

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model  # type: ignore
from scipy.io import loadmat

# Import modules from your pipeline
from spike_pipeline.inference.detect_spikes import detect_spikes
from spike_pipeline.utils.matching import match_spikes, calculate_detection_metrics
from spike_pipeline.utils.normalization import z_score_normalize
from spike_pipeline.utils.signal_tools import butter_bandpass_filter

# --- Configuration (These values should match your training setup) ---
# NOTE: Paths are relative to the project root, regardless of where this script is located.
DATASET_PATH = 'D1.mat'
MODEL_PATH = 'outputs/detector_model.h5'
REFRACTORY_PERIOD = 20  # Samples (based on typical spike properties)
MATCHING_TOLERANCE = 50  # Samples (as per coursework spec for matching)
WINDOW_SIZE = 128  # Samples (based on your summary document)

# Signal processing configuration (from typical spike sorting pipelines)
FS = 25000  # 25 kHz, based on D2 table in feedback report
FILTER_LOWCUT = 300  # Hz
FILTER_HIGHCUT = 3000  # Hz
FILTER_ORDER = 3


def load_data_and_preprocess():
    """Loads D1 data and applies preprocessing steps (filtering, normalization)."""
    try:
        # Load the .mat file (D1 has the 'signal' and 'gt_index' fields)
        data = loadmat(DATASET_PATH)
        signal = data['signal'].flatten()
        gt_indices = data['gt_index'].flatten()
        print(
            f"Loaded D1: Signal length {len(signal)}, Ground Truth Spikes {len(gt_indices)}")

        # 1. Bandpass filter the signal (Crucial step for noise reduction)
        # Note: Your full pipeline likely already does this, but it must be done before detection here too.
        filtered_signal = butter_bandpass_filter(
            signal, FILTER_LOWCUT, FILTER_HIGHCUT, FS, order=FILTER_ORDER)

        # 2. Global Z-Score Normalization (As per your summary document)
        normalized_signal = z_score_normalize(filtered_signal)

        return normalized_signal, gt_indices

    except FileNotFoundError:
        print(
            f"ERROR: Could not find dataset file at {DATASET_PATH}. Please ensure D1.mat is in the correct location.")
        return None, None
    except KeyError:
        print("ERROR: D1.mat does not contain expected keys ('signal', 'gt_index').")
        return None, None


def analyze_thresholds(signal, gt_indices, model):
    """Iterates through thresholds, calculates metrics, and plots P/R curves."""

    # We are using a focused threshold range (0.5 to 0.99 in 25 steps)
    # to speed up analysis and focus on the high-confidence region,
    # which should help improve your currently low Precision score.
    thresholds = np.linspace(0.5, 0.99, 25)

    precisions = []
    recalls = []
    f1_scores = []

    print("\n--- Running Threshold Analysis ---")

    for threshold in thresholds:
        # 1. Run detection
        predicted_indices = detect_spikes(
            signal, model, WINDOW_SIZE, REFRACTORY_PERIOD, threshold)

        # 2. Match predictions to ground truth
        tp_gt, tp_pred, fp, fn = match_spikes(
            predicted_indices, gt_indices, MATCHING_TOLERANCE)

        # 3. Calculate metrics
        precision, recall, f1 = calculate_detection_metrics(tp_pred, fp, fn)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"T: {threshold:.2f} | Pred Spikes: {len(predicted_indices):<6} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

    # Plotting the P/R/F1 curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', marker='.')
    plt.plot(thresholds, recalls, label='Recall', marker='.')
    plt.plot(thresholds, f1_scores, label='F1 Score',
             marker='.', linestyle='--')

    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx]

    plt.axvline(x=best_threshold, color='r', linestyle=':',
                label=f'Optimal F1 Threshold ({best_threshold:.2f}, F1={best_f1:.3f})')

    plt.title('Detector Performance vs. Threshold (on D1)')
    plt.xlabel('Detection Threshold')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    return best_threshold


def visual_inspection(signal, gt_indices, model, threshold):
    """Plots a segment of the signal with ground truth and predicted spikes."""

    # Run detection with the suggested threshold
    predicted_indices = detect_spikes(
        signal, model, WINDOW_SIZE, REFRACTORY_PERIOD, threshold)

    # Define a small segment for visualization (e.g., first 5000 samples)
    start_sample = 10000
    end_sample = 15000
    segment = signal[start_sample:end_sample]
    time_points = np.arange(len(segment))

    # Filter indices to only include those in the segment
    segment_gt_indices = gt_indices[(gt_indices >= start_sample) & (
        gt_indices < end_sample)] - start_sample
    segment_pred_indices = predicted_indices[(predicted_indices >= start_sample) & (
        predicted_indices < end_sample)] - start_sample

    plt.figure(figsize=(14, 6))
    plt.plot(time_points, segment, color='gray',
             alpha=0.7, label='Normalized Signal')

    # Plot Ground Truth Spikes
    plt.scatter(segment_gt_indices, segment[segment_gt_indices],
                color='green', marker='o', s=100, label='Ground Truth Spikes', zorder=5)

    # Plot Predicted Spikes
    plt.scatter(segment_pred_indices, segment[segment_pred_indices],
                color='red', marker='x', s=100, label='Predicted Spikes', zorder=6)

    plt.title(
        f'Visual Spike Detection Inspection (Samples {start_sample} - {end_sample})')
    plt.xlabel('Sample Index (Relative to Segment Start)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

    # Print statistics for this segment
    print(
        f"\n--- Visual Inspection Statistics (Threshold={threshold:.2f}) ---")
    print(f"Ground Truth Spikes in Segment: {len(segment_gt_indices)}")
    print(f"Predicted Spikes in Segment: {len(segment_pred_indices)}")

    # Calculate metrics for the segment for comparison (Optional, but useful)
    segment_gt_abs = gt_indices[(
        gt_indices >= start_sample) & (gt_indices < end_sample)]
    segment_pred_abs = predicted_indices[(
        predicted_indices >= start_sample) & (predicted_indices < end_sample)]
    tp_gt, tp_pred, fp, fn = match_spikes(
        segment_pred_abs, segment_gt_abs, MATCHING_TOLERANCE)
    precision, recall, f1 = calculate_detection_metrics(tp_pred, fp, fn)
    print(f"Segment Metrics: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(
            f"ERROR: Detector model not found at {MODEL_PATH}. Please run main_train.py first to train the detector.")
        return

    # Load Model
    detector_model = load_model(MODEL_PATH)

    # Load and Preprocess Data
    signal, gt_indices = load_data_and_preprocess()

    if signal is None:
        return

    # STEP 1: Analyze performance across a range of thresholds
    best_threshold = analyze_thresholds(signal, gt_indices, detector_model)

    # STEP 2: Visually inspect a segment using the recommended threshold
    if best_threshold:
        print(
            f"\n--- Visualizing with Optimal F1 Threshold ({best_threshold:.2f}) ---")
        visual_inspection(signal, gt_indices, detector_model, best_threshold)

    print("\nNext Steps:")
    print("1. Find the optimal threshold from the plot (Highest F1 Score).")
    print("2. The plots will show you which threshold maximizes precision (reduces false positives).")
    print("3. Update your 'spike_pipeline/training/tune_detector_threshold.py' script to target this new, optimal threshold value.")


if __name__ == '__main__':
    # Ensure the path for the detector model is accessible
    # This might require you to run 'main_train.py' first if you haven't saved a model yet.
    # Set the backend to Agg for non-interactive environments (if needed)
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
