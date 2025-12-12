# ==============================================================================
# DETECTOR MODEL TRAINING
# ==============================================================================
# Trains the binary CNN (Spike vs. Background) using the generated datasets.
#
# 1. INPUT DATA
#    - X_detector: (N, 120, 1) windows containing spikes or noise.
#    - y_detector: (N, 1) binary labels (1=Spike, 0=Noise).
#
# 2. TRAINING STRATEGY
#    - Split: 80% Training, 20% Validation (Random shuffle).
#    - Optimizer: Adam with default learning rate.
#    - Regularization: Early Stopping monitors 'val_loss' to prevent overfitting.
#      (Stops if no improvement for 3 epochs, restores best weights).
#
# 3. OUTPUT
#    - Saves the trained model to 'outputs/spike_detector_model.keras'.
# ==============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from spike_pipeline.models import build_detector_model


def train_detector(X_path="outputs/X_detector.npy",
                   y_path="outputs/y_detector.npy",
                   save_path="outputs/spike_detector_model.keras"):
    """
    Loads detector data, trains the binary CNN, and saves the best model.
    Returns the trained Keras model object.
    """
    # 1. Load Data
    print(f"Loading detector data from {X_path}...")
    X = np.load(X_path)
    y = np.load(y_path)

    # 2. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # 3. Build Model
    model = build_detector_model(window=X.shape[1])

    # 4. Configure Training
    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    # 5. Save
    model.save(save_path)
    print(f"Detector model saved to: {save_path}")

    return model
