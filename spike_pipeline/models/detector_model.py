# ==============================================================================
# SPIKE DETECTOR ARCHITECTURE (CNN)
# ==============================================================================
# Defines the neural network used for identifying spike locations in the signal.
#
# 1. ARCHITECTURE TYPE: Sequence Labeling
#    - Unlike a standard classifier that outputs one label per window, this model
#      outputs a probability *curve* matching the input length.
#    - Input:  (Batch, 120, 1) -> Raw signal window
#    - Output: (Batch, 120, 1) -> Probability of a spike at each time step
#
# 2. KEY DESIGN CHOICES
#    - Padding="same": Essential to keep the time axis length constant (120 -> 120).
#    - No Pooling: We do not use MaxPool or GlobalPool because we need to preserve
#      temporal resolution to pinpoint exactly *when* the spike occurs.
#    - 1x1 Convolution: The final layer acts like a Dense layer applied to every
#      time step independently, producing a point-wise classification.
# ==============================================================================

from tensorflow.keras import layers, models, optimizers  # type: ignore


def build_detector_model(window=120):
    """
    Constructs the Sequence Labeling Spike Detector model.
    Returns a compiled Keras model ready for training.
    """
    inputs = layers.Input(shape=(window, 1))

    # 1. Feature Extraction Layers
    # We use 3 Conv1D layers to learn increasingly complex temporal features.
    # 'same' padding ensures the output length matches the input length.
    x = layers.Conv1D(16, kernel_size=5, padding="same",
                      activation="relu")(inputs)
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)

    # Note: We deliberately avoid Strides or Pooling here to prevent
    # losing temporal precision.

    # 2. Per-Timestep Classification
    # A 1x1 Convolution maps the 64 features at each time step down to
    # a single scalar probability (0-1).
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
