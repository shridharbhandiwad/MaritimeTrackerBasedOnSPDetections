"""
Maritime Radar Preprocessing Module
=================================

Preprocessing components for maritime radar data including:
- CFAR detection algorithms
- Doppler filtering and processing
- Coordinate normalization and projection
"""

from .cfar import (
    CFARDetector, CACFARDetector, SOCFARDetector, GOCFARDetector, OSCFARDetector,
    adaptive_cfar_selection, classify_clutter_environment
)

from .doppler_filter import (
    DopplerProcessor, AngleGating, CoordinateProcessor,
    extract_detection_features
)

__all__ = [
    'CFARDetector', 'CACFARDetector', 'SOCFARDetector', 'GOCFARDetector', 'OSCFARDetector',
    'adaptive_cfar_selection', 'classify_clutter_environment',
    'DopplerProcessor', 'AngleGating', 'CoordinateProcessor',
    'extract_detection_features'
]