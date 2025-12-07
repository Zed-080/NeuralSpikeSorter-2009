import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from spike_pipeline.models import build_detector_model


def train_detector(X_path="outputs/X_detector.npy",
                   y_path="outputs/y_detector.npy",
                   save_path="outputs/spike_detector_model.keras"):

    X = np.load(X_path)
    y = np.load(y_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = build_detector_model(window=X.shape[1])

    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    model.save(save_path)
    print(f"Detector model saved to: {save_path}")

    return model
