from tensorflow.keras import layers, models, optimizers  # type: ignore


def build_detector_model(window=128):
    """
    Sequence Labeling Spike Detector (Matches 'src' pipeline).

    Input:  (Batch, window, 1)
    Output: (Batch, window, 1) -> A probability curve over time.
    """
    inputs = layers.Input(shape=(window, 1))

    # 1. Feature Extraction (Keep time dimension -> padding="same", strides=1)
    #    Note: 'src' uses 16 -> 32 -> 64 filters
    x = layers.Conv1D(16, kernel_size=5, padding="same",
                      activation="relu")(inputs)
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)

    # REMOVED: Strided convolutions (they shrink the time axis)
    # REMOVED: GlobalAveragePooling1D (it destroys the time axis)
    # REMOVED: Dense layers (they destroy the time axis)

    # 2. Per-Timestep Classification
    #    We use a 1x1 convolution to map the 64 features at each time step
    #    to a single probability (sigmoid).
    outputs = layers.Conv1D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="sigmoid"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="spike_detector")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
