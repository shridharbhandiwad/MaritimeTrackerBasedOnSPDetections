"""
Maritime Radar Tracking System with Deep Learning
================================================

A comprehensive deep learning-based tracker for maritime radar systems
designed to robustly track maritime targets while rejecting sea clutter.

Components:
- Preprocessing: CFAR, Doppler filtering, coordinate normalization
- Core tracking: DeepSORT, Transformer-based tracking, GNN classification
- Post-processing: Clutter rejection, track smoothing, confirmation logic
- Evaluation: OSPA, MOTA/MOTP, fragmentation analysis
"""

__version__ = "1.0.0"
__author__ = "Maritime Radar Tracking Team"

from .preprocessing import *
from .models import *
from .tracking import *
from .evaluation import *
from .utils import *