import os
import shutil
import zipfile
import numpy as np
import tensorflow as tf
from scipy.io import savemat

# Import modules from your spike_pipeline
from spike_pipeline.data_loader.load_datasets import load_unlabelled
from spike_pipeline.inference.run_pipeline_D2_D6 import run_inference_dataset

# --- Configuration ---
OUTPUT_DIR = 'outputs'
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
SUBMISSION_DIR = 'submission_datasets'
SUBMISSION_ZIP_NAME = 'NeuralSpikeSorter_Submission.zip'

DETECTOR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'spike_detector_model.keras')
CLASSIFIER_MODEL_PATH = os.path.join(
    OUTPUT_DIR, 'spike_classifier_model.keras')
TUNING_PARAMS_PATH = os.path.join(OUTPUT_DIR, 'detector_params.txt')


def load_tuning_params():
    """Loads the refractory period (Threshold is now ignored/hardcoded per dataset)."""
    try:
        with open(TUNING_PARAMS_PATH, 'r') as f:
            content = f.read().strip()

        # We only care about refractory here.
        # Threshold is handled inside run_pipeline_D2_D6 via the dictionary.
        threshold_str, refractory_str = content.split(',')
        return int(refractory_str.strip())
    except Exception as e:
        print(f"Defaulting to refractory=45 due to error: {e}")
        return 45


def prepare_submission_zip():
    # ... (Keep this function exactly as it is) ...
    print("-" * 50)
    print("STARTING SUBMISSION PREPARATION")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    dataset_files = [f"D{i}.mat" for i in range(2, 7)]
    copied_files = []
    for filename in dataset_files:
        source_path = os.path.join(PREDICTIONS_DIR, filename)
        target_path = os.path.join(SUBMISSION_DIR, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            copied_files.append(filename)

    zip_filepath = SUBMISSION_ZIP_NAME
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, SUBMISSION_DIR)
                zf.write(full_path, relative_path)
    print(f"SUCCESS: Submission ZIP created: {zip_filepath}")


def main_infer():
    # 1. Load trained models
    try:
        detector_model = tf.keras.models.load_model(DETECTOR_MODEL_PATH)
        classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Could not load models: {e}")
        return

    # 2. Load tuned refractory period
    refractory_period = load_tuning_params()

    # 3. Ensure prediction output directory exists
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # 4. Process datasets D2-D6
    datasets = ['D2', 'D3', 'D4', 'D5', 'D6']

    for dataset_name in datasets:
        print(f"\n--- Processing {dataset_name}.mat ---")
        input_filepath = f'{dataset_name}.mat'
        output_filepath = os.path.join(PREDICTIONS_DIR, f'{dataset_name}.mat')

        # CHANGED: Removed 'threshold' argument.
        # The pipeline now selects it automatically based on 'D2', 'D3', etc.
        run_inference_dataset(
            detector_model=detector_model,
            classifier_model=classifier_model,
            path=input_filepath,
            save_path=output_filepath,
            refractory=refractory_period
        )

    # 5. Prepare ZIP
    prepare_submission_zip()


if __name__ == '__main__':
    main_infer()
