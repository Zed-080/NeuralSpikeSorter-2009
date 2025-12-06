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

# --- IMPORT MODULES ---

# --- CONFIGURATION ---
RUN_TUNING = False  # Set False to skip tuning and use cached config


def main():
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
