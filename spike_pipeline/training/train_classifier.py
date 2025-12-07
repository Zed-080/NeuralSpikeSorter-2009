import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from spike_pipeline.models import build_classifier_model


# def train_classifier(X_path="outputs/X_classifier.npy",
#                      y_path="outputs/y_classifier.npy",
#                      y_raw_path="outputs/y_classifier_raw.npy",
#                      save_path="outputs/spike_classifier_model.keras"):

#     X = np.load(X_path)
#     y = np.load(y_path)
#     y_raw = np.load(y_raw_path)

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y,
#         test_size=0.2,
#         stratify=y_raw,
#         shuffle=True,
#         random_state=42
#     )

#     model = build_classifier_model(window=X.shape[1], num_classes=5)

#     es = EarlyStopping(
#         monitor="val_loss",
#         patience=5,
#         restore_best_weights=True
#     )

#     model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=20,
#         batch_size=64,
#         callbacks=[es],
#         verbose=1
#     )

#     model.save(save_path)
#     print(f"Classifier model saved to: {save_path}")

#     return model

def train_classifier(X_path="outputs/X_classifier.npy",
                     y_path="outputs/y_classifier.npy",
                     y_raw_path="outputs/y_classifier_raw.npy",
                     save_path="outputs/spike_classifier_model.keras"):

    X = np.load(X_path)
    y = np.load(y_path)
    y_raw = np.load(y_raw_path)

    # --- FIX: Match src exactly (random_state=42, return raw labels) ---
    X_train, X_val, y_train, y_val, _, y_raw_val = train_test_split(
        X, y, y_raw,
        test_size=0.2,
        stratify=y_raw,
        shuffle=True,
        random_state=42  # <--- Ensures identical validation set every time
    )

    model = build_classifier_model(window=X.shape[1], num_classes=5)

    es = EarlyStopping(
        monitor="val_loss",
        patience=4,
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

    # --- ADDED: Evaluation Report (Matches src) ---
    print("\nValidation Classification Report:")
    y_val_probs = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_probs, axis=1)

    print(classification_report(y_raw_val, y_val_pred, digits=3))
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_raw_val, y_val_pred))

    return model
