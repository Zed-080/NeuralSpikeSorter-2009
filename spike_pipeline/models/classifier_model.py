# ==============================================================================
# SPIKE CLASSIFIER ARCHITECTURE (CNN)
# ==============================================================================
# Defines the neural network used for distinguishing between neuron types (Classes 1-5).
#
# 1. ARCHITECTURE TYPE: Waveform Classifier
#    - Input:  (Batch, 64, 1) -> Centered spike waveform
#    - Output: (Batch, 5)     -> Probability distribution over classes
#
# 2. KEY DESIGN CHOICES
#    - Batch Normalization: Used after convolutions to stabilize training and
#      allow higher learning rates.
#    - Global Average Pooling: Compresses the entire time dimension into a single
#      feature vector. This makes the model robust to small temporal misalignments
#      (shift invariance).
#    - Dropout: Applied before the final layer to prevent overfitting on the D1 training set.
# ==============================================================================

from tensorflow.keras import layers, models, optimizers  # type: ignore


def build_classifier_model(window=64, num_classes=5):
    """
    Constructs the Multi-Class Waveform Classifier model.
    Returns a compiled Keras model ready for training.
    """
    inputs = layers.Input(shape=(window, 1))

    # 1. Convolutional Feature Extraction
    # Block 1
    x = layers.Conv1D(32, kernel_size=5, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv1D(64, kernel_size=5, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # 2. Dimensionality Reduction
    # Global Pooling squashes (Batch, Time, Feat) -> (Batch, Feat)
    x = layers.GlobalAveragePooling1D()(x)

    # 3. Classification Head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)  # Regularization

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="spike_classifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
