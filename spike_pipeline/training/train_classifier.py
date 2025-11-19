import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from spike_pipeline.models import build_classifier_model


def train_classifier(X_path="outputs/X_classifier.npy",
                     y_path="outputs/y_classifier.npy",
                     y_raw_path="outputs/y_classifier_raw.npy",
                     save_path="outputs/spike_classifier_model.keras"):

    X = np.load(X_path)
    y = np.load(y_path)
    y_raw = np.load(y_raw_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y_raw
    )

    model = build_classifier_model(window=X.shape[1], num_classes=5)

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )

    model.save(save_path)
    print(f"Classifier model saved to: {save_path}")

    return model
