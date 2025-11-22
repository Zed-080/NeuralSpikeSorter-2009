import os
import shutil
import zipfile
import numpy as np
import tensorflow as tf
from scipy.io import savemat

# Import modules from your spike_pipeline
from spike_pipeline.data_loader.load_datasets import load_unlabelled
# from spike_pipeline.inference.run_pipeline_D2_D6 import run_pipeline_D2_D6
from spike_pipeline.inference.run_pipeline_D2_D6 import run_inference_dataset

# --- Configuration ---
# Directory where models are saved and where outputs (predictions) will be placed.
OUTPUT_DIR = 'outputs'
# Assuming prediction .mat files are generated in a subdirectory of OUTPUT_DIR
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
SUBMISSION_DIR = 'submission_datasets'
SUBMISSION_ZIP_NAME = 'NeuralSpikeSorter_Submission.zip'

# Paths to trained models and tuning parameters
DETECTOR_MODEL_PATH = os.path.join(OUTPUT_DIR, 'spike_detector_model.keras')
CLASSIFIER_MODEL_PATH = os.path.join(
    OUTPUT_DIR, 'spike_classifier_model.keras')
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, 'detector_threshold.txt')
REFRACTORY_PATH = os.path.join(OUTPUT_DIR, 'refractory_period.txt')


def load_tuning_params():
    """Loads the best tuned threshold and refractory period."""
    try:
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        with open(REFRACTORY_PATH, 'r') as f:
            refractory_period = int(f.read().strip())
        print(
            f"Loaded detector parameters: Threshold={threshold:.4f}, Refractory={refractory_period}")
        return threshold, refractory_period
    except FileNotFoundError:
        print("ERROR: Tuning parameters not found. Run main_train.py first!")
        return None, None
    except Exception as e:
        print(f"ERROR loading tuning parameters: {e}")
        return None, None


def prepare_submission_zip():
    """
    Creates the final submission folder, copies the required D2-D6 .mat files into 
    it, and zips the folder for final upload.
    """
    print("-" * 50)
    print("STARTING SUBMISSION PREPARATION")

    # 1. Ensure the submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Define the datasets to be included in the submission
    # D2.mat, D3.mat, ..., D6.mat
    dataset_files = [f"D{i}.mat" for i in range(2, 7)]

    copied_files = []

    # 3. Copy the files from the predictions directory to the submission directory
    for filename in dataset_files:
        source_path = os.path.join(PREDICTIONS_DIR, filename)
        target_path = os.path.join(SUBMISSION_DIR, filename)

        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            copied_files.append(filename)
            print(f"  -> Copied {filename}")
        else:
            print(
                f"  WARNING: Prediction file {filename} not found at {source_path}. Skipping.")

    print(
        f"\nCopied {len(copied_files)} out of 5 expected files into {SUBMISSION_DIR}")

    # 4. Create the final ZIP file
    zip_filepath = SUBMISSION_ZIP_NAME

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Walk through the submission directory and add files to the zip
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                # Determine the path inside the ZIP file (relative to submission folder)
                relative_path = os.path.relpath(full_path, SUBMISSION_DIR)
                zf.write(full_path, relative_path)

    print("-" * 50)
    print(f"✅ SUCCESS: Submission ZIP created!")
    print(f"Zip file: {zip_filepath}")
    print(f"You can upload this ZIP file for marking.")
    print("-" * 50)


def main_infer():
    """Main function for the inference pipeline (Task 2)."""

    # 1. Load trained models
    try:
        detector_model = tf.keras.models.load_model(DETECTOR_MODEL_PATH)
        classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print(
            f"ERROR: Could not load trained models. Check paths or run main_train.py first. Error: {e}")
        return

    # 2. Load tuned parameters
    threshold, refractory_period = load_tuning_params()
    if threshold is None:
        return

    # 3. Ensure prediction output directory exists
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # 4. Process datasets D2-D6
    # The run_pipeline_D2_D6 function is expected to handle the loading, detection,
    # classification, and saving of the D2-D6.mat files into PREDICTIONS_DIR.
    datasets = ['D2', 'D3', 'D4', 'D5', 'D6']

    for dataset_name in datasets:
        print(f"\n--- Processing {dataset_name}.mat ---")

        # Run the full pipeline and save the results
        # Assuming run_inference_dataset saves the output .mat file with Index and Class
        # directly into PREDICTIONS_DIR
        input_filepath = f'{dataset_name}.mat'
        output_filepath = os.path.join(PREDICTIONS_DIR, f'{dataset_name}.mat')

        run_inference_dataset(
            detector_model=detector_model,
            classifier_model=classifier_model,
            threshold=threshold,
            refractory=refractory_period,
            path=input_filepath,       # Pass the input file path
            save_path=output_filepath  # Pass the output file path
        )
        print(f"Finished processing {dataset_name}. Results saved.")

    # 5. Prepare the final ZIP file for submission
    prepare_submission_zip()


if __name__ == '__main__':
    main_infer()
