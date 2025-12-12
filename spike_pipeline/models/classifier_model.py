from tensorflow.keras import layers, models, optimizers  # type: ignore


def build_classifier_model(window=64, num_classes=5):
    """
    Exact match to 'src' repo architecture.
    """
    inputs = layers.Input(shape=(window, 1))

    # Layer 1
    x = layers.Conv1D(32, kernel_size=5, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Layer 2
    x = layers.Conv1D(64, kernel_size=5, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Global Pooling (Squashes time dimension)
    x = layers.GlobalAveragePooling1D()(x)

    # Dense Layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="spike_classifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
