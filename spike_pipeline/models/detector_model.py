from tensorflow.keras import layers, models, optimizers  # type: ignore


def build_detector_model(window=128):
    """
    Spike detector model:
      Input: (window=128, 1)
      Output: probability of spike (sigmoid)
    """

    inputs = layers.Input(shape=(window, 1))

    x = layers.Conv1D(32, kernel_size=5, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # downsample with strided conv
    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # classifier head
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
