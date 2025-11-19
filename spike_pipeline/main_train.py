from spike_pipeline.data_loader import (
    build_detector_dataset,
    build_classifier_dataset,
)

from spike_pipeline.training import (
    train_detector,
    tune_detector_threshold,
    train_classifier,
)


def main():
    print("=== Building Detector Dataset ===")
    build_detector_dataset("D1.mat", save_prefix="outputs/")

    print("\n=== Building Classifier Dataset ===")
    build_classifier_dataset("D1.mat", save_prefix="outputs/")

    print("\n=== Training Detector CNN ===")
    detector_model = train_detector(
        X_path="outputs/X_detector.npy",
        y_path="outputs/y_detector.npy",
        save_path="outputs/spike_detector_model.keras"
    )

    print("\n=== Tuning Detector Threshold & Refractory ===")
    best_threshold, best_refractory = tune_detector_threshold(
        detector_model,
        D1_path="D1.mat"
    )

    # save the tuned parameters
    with open("outputs/detector_params.txt", "w") as f:
        f.write(f"{best_threshold},{best_refractory}")

    print("\n=== Training Classifier CNN ===")
    classifier_model = train_classifier(
        X_path="outputs/X_classifier.npy",
        y_path="outputs/y_classifier.npy",
        y_raw_path="outputs/y_classifier_raw.npy",
        save_path="outputs/spike_classifier_model.keras"
    )

    print("\n🎉 TRAINING COMPLETE 🎉")
    print(f"Tuned threshold: {best_threshold}")
    print(f"Tuned refractory: {best_refractory}")


if __name__ == "__main__":
    main()
