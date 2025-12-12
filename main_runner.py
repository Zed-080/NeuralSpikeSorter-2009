# ==============================================================================
# MASTER PIPELINE RUNNER (TRAINING & INFERENCE)
# ==============================================================================
# This script executes the entire end-to-end pipeline:
#
# 1. DATA GENERATION: Creates training datasets from D1 (Clean + Noisy versions).
# 2. TRAINING:
#    - Trains the Detector CNN (Spike vs Noise).
#    - Trains the Classifier CNN (5-Class Neuron Type).
# 3. TUNING: Automatically finds the optimal decision threshold and refractory
#    period by maximizing F1 score on the clean D1 dataset.
# 4. INFERENCE: Runs the trained models on D2-D6 to generate submission files.
# ==============================================================================

from main_infer import main_infer
from spike_pipeline.training.tune_detector_threshold import tune_detector_threshold
from spike_pipeline.training.train_classifier import train_classifier
from spike_pipeline.training.train_detector import train_detector
from spike_pipeline.data_loader.generate_datasets import main as build_all_data
import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- CONFIGURATION ---
RUN_TUNING = True  # Set False to skip tuning and use cached config


def main():
    """
    Main entry point. Sequentially runs data generation, training, tuning,
    and final inference on the unlabelled datasets.
    """
    print("\n" + "="*60)
    print("STEP 1: GENERATING DATASETS")
    print("="*60)
    build_all_data()

    print("\n" + "="*60)
    print("STEP 2: TRAINING DETECTOR")
    print("="*60)
    det_model = train_detector()

    print("\n" + "="*60)
    print("STEP 3: TUNING THRESHOLDS")
    print("="*60)

    if RUN_TUNING:
        print(">> Running automated threshold tuning...")
        # Saves to outputs/detector_config.npz
        tune_detector_threshold(det_model)
    else:
        print(">> SKIPPING tuning. Using cached config if available.")

    print("\n" + "="*60)
    print("STEP 4: TRAINING CLASSIFIER")
    print("="*60)
    train_classifier()

    print("\n" + "="*60)
    print("STEP 5: INFERENCE & SUBMISSION")
    print("="*60)
    # Runs inference on D2-D6 using the tuned (or manual) thresholds
    main_infer()

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
