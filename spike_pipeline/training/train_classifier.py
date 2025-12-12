# ==============================================================================
# CLASSIFIER MODEL TRAINING
# ==============================================================================
# Trains the 5-class CNN to distinguish between different neuron types.
#
# 1. INPUT DATA
#    - X: (N, 64, 1) centered spike waveforms.
#    - y: (N, 5) One-Hot encoded labels for training.
#    - y_raw: (N,) Integer labels (0-4) used for stratified splitting.
#
# 2. TRAINING STRATEGY
#    - Split: Stratified 80/20 split. This ensures rare neuron classes are
#      represented equally in training and validation sets.
#    - Metrics: Accuracy, plus a full Classification Report (Precision/Recall/F1)
#      and Confusion Matrix printed after training.
#
# 3. OUTPUT
#    - Saves the trained model to 'outputs/spike_classifier_model.keras'.
# ==============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from spike_pipeline.models import build_classifier_model


def train_classifier(X_path="outputs/X_classifier.npy",
                     y_path="outputs/y_classifier.npy",
                     y_raw_path="outputs/y_classifier_raw.npy",
                     save_path="outputs/spike_classifier_model.keras"):
    """
    Trains the multi-class classifier using stratified splitting.
    Prints a detailed performance report (F1 per class) after training.
    """
    # 1. Load Data
    print(f"Loading classifier data from {X_path}...")
    X = np.load(X_path)
    y = np.load(y_path)
    y_raw = np.load(y_raw_path)

    # 2. Stratified Split (Crucial for imbalanced classes)
    # random_state=42 ensures the validation set is identical every run
    X_train, X_val, y_train, y_val, _, y_raw_val = train_test_split(
        X, y, y_raw,
        test_size=0.2,
        stratify=y_raw,
        shuffle=True,
        random_state=42
    )

    # 3. Build Model
    model = build_classifier_model(window=X.shape[1], num_classes=5)

    # 4. Configure Training
    es = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    print("Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    # 5. Save
    model.save(save_path)
    print(f"Classifier model saved to: {save_path}")

    # 6. Evaluation Report
    print("\nValidation Classification Report:")
    y_val_probs = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_probs, axis=1)

    # digits=3 gives 3 decimal places for precision/recall
    print(classification_report(y_raw_val, y_val_pred, digits=3))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_raw_val, y_val_pred))

    return model
