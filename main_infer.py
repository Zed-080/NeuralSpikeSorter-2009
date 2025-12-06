import os
import shutil
import zipfile
import numpy as np
import tensorflow as tf
from scipy.io import savemat

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
TUNING_PARAMS_PATH = os.path.join(OUTPUT_DIR, 'detector_config.npz')


def load_tuning_params():
    """Returns (threshold, refractory) from config, or defaults."""
    try:
        cfg = np.load(TUNING_PARAMS_PATH)
        thr = float(cfg["decision_threshold"])
        refr = int(cfg["refractory_suppression"])
        print(f"Loaded tuning config: Thr={thr:.3f}, Refr={refr}")
        return thr, refr
    except Exception as e:
        print(f"Warning: Could not load tuning config ({e}). Using defaults.")
        return 0.75, 45


def prepare_submission_zip():
    print("-" * 50)
    print("STARTING SUBMISSION PREPARATION")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    dataset_files = [f"D{i}.mat" for i in range(2, 7)]
    for filename in dataset_files:
        source_path = os.path.join(PREDICTIONS_DIR, filename)
        target_path = os.path.join(SUBMISSION_DIR, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)

    zip_filepath = SUBMISSION_ZIP_NAME
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, SUBMISSION_DIR)
                zf.write(full_path, relative_path)
    print(f"SUCCESS: Submission ZIP created: {zip_filepath}")


def main_infer():
    # 1. Load Models
    det_model = tf.keras.models.load_model(DETECTOR_MODEL_PATH)
    clf_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)

    # 2. Load Tuned Params
    best_thr, best_refr = load_tuning_params()

    # 3. Run Inference
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    for ds in ['D2', 'D3', 'D4', 'D5', 'D6']:
        run_inference_dataset(
            detector_model=det_model,
            classifier_model=clf_model,
            path=f"{ds}.mat",
            save_path=os.path.join(PREDICTIONS_DIR, f"{ds}.mat"),
            refractory=best_refr,
            default_threshold=best_thr  # Matches the new argument
        )

    # 4. Prepare ZIP
    prepare_submission_zip()


if __name__ == '__main__':
    main_infer()
