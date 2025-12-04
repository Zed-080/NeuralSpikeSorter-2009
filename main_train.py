from spike_pipeline.data_loader import (
    build_detector_dataset,
    build_classifier_dataset,  # Ensure this exists in your project
)
from spike_pipeline.training import (
    train_detector,
    train_classifier,
)


def main():
    # 1. Build Data
    print("=== Building Detector Dataset (Sequence Mode) ===")
    # This now calls the NEW code we just wrote
    build_detector_dataset("D1.mat", save_prefix="outputs/")

    print("\n=== Building Classifier Dataset ===")
    # This remains unchanged (standard classification)
    build_classifier_dataset("D1.mat", save_prefix="outputs/")

    # 2. Train Detector
    print("\n=== Training Detector (Sequence Model) ===")
    # This trains the new Conv1D->Conv1D model
    train_detector(
        X_path="outputs/X_detector.npy",
        y_path="outputs/y_detector.npy",
        save_path="outputs/spike_detector_model.keras"
    )

    # 3. Train Classifier
    print("\n=== Training Classifier ===")
    train_classifier(
        X_path="outputs/X_classifier.npy",
        y_path="outputs/y_classifier.npy",
        y_raw_path="outputs/y_classifier_raw.npy",
        save_path="outputs/spike_classifier_model.keras"
    )

    print("\n🎉 TRAINING COMPLETE 🎉")
    print("You can now run main_infer.py to generate predictions.")


if __name__ == "__main__":
    main()
