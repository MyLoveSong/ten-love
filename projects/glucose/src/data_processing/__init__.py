"""
Data processing module for glucose prediction system.
Provides advanced data preprocessing, augmentation, and validation.
"""

from .data_augmentation import GlucoseDataAugmenter
from .rare_event_augment import RareEventAugmenter, RareEventDetector, SMOTEGlucose

try:
    from .wavelet_features import WaveletFeatureExtractor, WaveletTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover - optional PyWavelets path
    if exc.name != "pywt":
        raise
    WaveletFeatureExtractor = None
    WaveletTokenizer = None

__all__ = [
    'GlucoseDataAugmenter',
    'RareEventAugmenter',
    'RareEventDetector',
    'SMOTEGlucose'
]

if WaveletFeatureExtractor is not None:
    __all__.extend(['WaveletFeatureExtractor', 'WaveletTokenizer'])
