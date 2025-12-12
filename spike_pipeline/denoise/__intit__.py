"""
Denoising module for the Neural Spike Sorter.
Contains algorithms for signal enhancement including Matched Filtering 
and Wavelet-based noise reduction.
"""

from .matched_filter import build_average_spike_template, matched_filter_enhance
from .wavelet_denoise import denoise_dataset, wavelet_denoise

__all__ = [
    "build_average_spike_template",
    "matched_filter_enhance",
    "denoise_dataset",
    "wavelet_denoise",
]
