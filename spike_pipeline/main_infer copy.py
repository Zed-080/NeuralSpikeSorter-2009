import numpy as np
from tensorflow.keras.models import load_model

from spike_pipeline.inference import run_inference_dataset


def load_detector_params(path="outputs/detector_params.txt"):
    with open(path, "r") as f:
        t, r = f.read().split(",")
        return float(t), int(float(r))


def main():
    print("=== Loading Models ===")
    detector_model = load_model("outputs/spike_detector_model.keras")
    classifier_model = load_model("outputs/spike_classifier_model.keras")

    print("=== Loading Tuned Parameters ===")
    threshold, refractory = load_detector_params()
    print(f"Threshold = {threshold}, Refractory = {refractory}")

    print("\n=== Running Inference on D2–D6 ===")

    for i in range(2, 7):
        path = f"D{i}.mat"
        save_path = f"outputs/D{i}.mat"
        run_inference_dataset(
            detector_model,
            classifier_model,
            threshold,
            refractory,
            path,
            save_path
        )

    print("\n🎉 INFERENCE COMPLETE — All MAT files saved in outputs/ 🎉")


if __name__ == "__main__":
    main()
