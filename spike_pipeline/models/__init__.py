"""
Neural Network definitions for the Spike Sorting pipeline.
Exposes builders for the Detector (Binary Sequence Labeling) and 
Classifier (Multi-Class Waveform Classification).
"""

from .detector_model import build_detector_model
from .classifier_model import build_classifier_model

__all__ = [
    "build_detector_model",
    "build_classifier_model",
]
