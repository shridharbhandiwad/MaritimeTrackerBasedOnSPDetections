"""
CFAR (Constant False Alarm Rate) Detection Algorithms
====================================================

Implementation of various CFAR algorithms for maritime radar target detection:
- CA-CFAR (Cell Averaging)
- SO-CFAR (Smallest Of)
- GO-CFAR (Greatest Of)
- OS-CFAR (Ordered Statistics)
"""

import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional, Union
import warnings


class CFARDetector:
    """Base class for CFAR detection algorithms."""
    
    def __init__(self, 
                 guard_cells: int = 2,
                 reference_cells: int = 16,
                 pfa: float = 1e-6,
                 threshold_factor: Optional[float] = None):
        """
        Initialize CFAR detector.
        
        Args:
            guard_cells: Number of guard cells on each side of CUT
            reference_cells: Number of reference cells on each side
            pfa: Probability of false alarm
            threshold_factor: Custom threshold factor (overrides pfa if provided)
        """
        self.guard_cells = guard_cells
        self.reference_cells = reference_cells
        self.pfa = pfa
        self.threshold_factor = threshold_factor
        
    def _calculate_threshold_factor(self, n_ref: int) -> float:
        """Calculate threshold factor based on PFA and number of reference cells."""
        if self.threshold_factor is not None:
            return self.threshold_factor
        
        # For CA-CFAR with exponential noise
        alpha = n_ref * (self.pfa ** (-1/n_ref) - 1)
        return alpha
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform CFAR detection on input data.
        
        Args:
            data: Input radar data (range-azimuth or range-doppler)
            
        Returns:
            detections: Binary detection map
            thresholds: Adaptive threshold values
        """
        raise NotImplementedError


class CACFARDetector(CFARDetector):
    """Cell Averaging CFAR detector."""
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform CA-CFAR detection."""
        if data.ndim == 1:
            return self._detect_1d(data)
        elif data.ndim == 2:
            return self._detect_2d(data)
        else:
            raise ValueError("Input data must be 1D or 2D")
    
    def _detect_1d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """1D CA-CFAR detection."""
        n_samples = len(data)
        detections = np.zeros(n_samples, dtype=bool)
        thresholds = np.zeros(n_samples)
        
        window_size = 2 * (self.guard_cells + self.reference_cells) + 1
        half_window = window_size // 2
        
        alpha = self._calculate_threshold_factor(2 * self.reference_cells)
        
        for i in range(half_window, n_samples - half_window):
            # Define reference cells (excluding guard cells and CUT)
            left_start = i - half_window
            left_end = i - self.guard_cells
            right_start = i + self.guard_cells + 1
            right_end = i + half_window + 1
            
            # Calculate noise level from reference cells
            ref_cells = np.concatenate([
                data[left_start:left_end],
                data[right_start:right_end]
            ])
            
            noise_level = np.mean(ref_cells)
            threshold = alpha * noise_level
            thresholds[i] = threshold
            
            # Detection test
            detections[i] = data[i] > threshold
        
        return detections, thresholds
    
    def _detect_2d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """2D CA-CFAR detection."""
        rows, cols = data.shape
        detections = np.zeros_like(data, dtype=bool)
        thresholds = np.zeros_like(data)
        
        # Apply 1D CFAR along each dimension
        for i in range(rows):
            det_row, thresh_row = self._detect_1d(data[i, :])
            detections[i, :] = det_row
            thresholds[i, :] = thresh_row
        
        return detections, thresholds


class SOCFARDetector(CFARDetector):
    """Smallest Of CFAR detector for clutter edge situations."""
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform SO-CFAR detection."""
        return self._detect_1d(data) if data.ndim == 1 else self._detect_2d(data)
    
    def _detect_1d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """1D SO-CFAR detection."""
        n_samples = len(data)
        detections = np.zeros(n_samples, dtype=bool)
        thresholds = np.zeros(n_samples)
        
        window_size = 2 * (self.guard_cells + self.reference_cells) + 1
        half_window = window_size // 2
        
        alpha = self._calculate_threshold_factor(self.reference_cells)
        
        for i in range(half_window, n_samples - half_window):
            # Left and right reference windows
            left_start = i - half_window
            left_end = i - self.guard_cells
            right_start = i + self.guard_cells + 1
            right_end = i + half_window + 1
            
            left_cells = data[left_start:left_end]
            right_cells = data[right_start:right_end]
            
            # Calculate noise levels
            left_noise = np.mean(left_cells)
            right_noise = np.mean(right_cells)
            
            # Take smallest (most conservative)
            noise_level = min(left_noise, right_noise)
            threshold = alpha * noise_level
            thresholds[i] = threshold
            
            detections[i] = data[i] > threshold
        
        return detections, thresholds
    
    def _detect_2d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """2D SO-CFAR detection."""
        rows, cols = data.shape
        detections = np.zeros_like(data, dtype=bool)
        thresholds = np.zeros_like(data)
        
        for i in range(rows):
            det_row, thresh_row = self._detect_1d(data[i, :])
            detections[i, :] = det_row
            thresholds[i, :] = thresh_row
        
        return detections, thresholds


class GOCFARDetector(CFARDetector):
    """Greatest Of CFAR detector for multiple target situations."""
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform GO-CFAR detection."""
        return self._detect_1d(data) if data.ndim == 1 else self._detect_2d(data)
    
    def _detect_1d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """1D GO-CFAR detection."""
        n_samples = len(data)
        detections = np.zeros(n_samples, dtype=bool)
        thresholds = np.zeros(n_samples)
        
        window_size = 2 * (self.guard_cells + self.reference_cells) + 1
        half_window = window_size // 2
        
        alpha = self._calculate_threshold_factor(self.reference_cells)
        
        for i in range(half_window, n_samples - half_window):
            left_start = i - half_window
            left_end = i - self.guard_cells
            right_start = i + self.guard_cells + 1
            right_end = i + half_window + 1
            
            left_cells = data[left_start:left_end]
            right_cells = data[right_start:right_end]
            
            left_noise = np.mean(left_cells)
            right_noise = np.mean(right_cells)
            
            # Take greatest (less conservative)
            noise_level = max(left_noise, right_noise)
            threshold = alpha * noise_level
            thresholds[i] = threshold
            
            detections[i] = data[i] > threshold
        
        return detections, thresholds
    
    def _detect_2d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """2D GO-CFAR detection."""
        rows, cols = data.shape
        detections = np.zeros_like(data, dtype=bool)
        thresholds = np.zeros_like(data)
        
        for i in range(rows):
            det_row, thresh_row = self._detect_1d(data[i, :])
            detections[i, :] = det_row
            thresholds[i, :] = thresh_row
        
        return detections, thresholds


class OSCFARDetector(CFARDetector):
    """Ordered Statistics CFAR detector."""
    
    def __init__(self, k: int = None, **kwargs):
        """
        Initialize OS-CFAR detector.
        
        Args:
            k: Order statistic index (default: 3/4 of reference cells)
        """
        super().__init__(**kwargs)
        self.k = k
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform OS-CFAR detection."""
        return self._detect_1d(data) if data.ndim == 1 else self._detect_2d(data)
    
    def _detect_1d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """1D OS-CFAR detection."""
        n_samples = len(data)
        detections = np.zeros(n_samples, dtype=bool)
        thresholds = np.zeros(n_samples)
        
        window_size = 2 * (self.guard_cells + self.reference_cells) + 1
        half_window = window_size // 2
        
        n_ref = 2 * self.reference_cells
        k = self.k if self.k is not None else int(0.75 * n_ref)
        alpha = self._calculate_threshold_factor(n_ref)
        
        for i in range(half_window, n_samples - half_window):
            left_start = i - half_window
            left_end = i - self.guard_cells
            right_start = i + self.guard_cells + 1
            right_end = i + half_window + 1
            
            ref_cells = np.concatenate([
                data[left_start:left_end],
                data[right_start:right_end]
            ])
            
            # Sort and take k-th order statistic
            sorted_refs = np.sort(ref_cells)
            noise_level = sorted_refs[k-1]  # k-th order statistic (0-indexed)
            
            threshold = alpha * noise_level
            thresholds[i] = threshold
            
            detections[i] = data[i] > threshold
        
        return detections, thresholds
    
    def _detect_2d(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """2D OS-CFAR detection."""
        rows, cols = data.shape
        detections = np.zeros_like(data, dtype=bool)
        thresholds = np.zeros_like(data)
        
        for i in range(rows):
            det_row, thresh_row = self._detect_1d(data[i, :])
            detections[i, :] = det_row
            thresholds[i, :] = thresh_row
        
        return detections, thresholds


def adaptive_cfar_selection(data: np.ndarray, 
                           detectors: list = None,
                           clutter_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptively select CFAR algorithm based on local clutter characteristics.
    
    Args:
        data: Input radar data
        detectors: List of CFAR detectors to choose from
        clutter_map: Optional clutter type map (0: homogeneous, 1: edge, 2: multiple targets)
        
    Returns:
        detections: Combined detection map
        thresholds: Combined threshold map
    """
    if detectors is None:
        detectors = [
            CACFARDetector(),  # For homogeneous clutter
            SOCFARDetector(),  # For clutter edges
            GOCFARDetector()   # For multiple targets
        ]
    
    if clutter_map is None:
        # Simple clutter classification based on local variance
        clutter_map = classify_clutter_environment(data)
    
    detections = np.zeros_like(data, dtype=bool)
    thresholds = np.zeros_like(data)
    
    for detector_idx, detector in enumerate(detectors):
        mask = (clutter_map == detector_idx)
        if np.any(mask):
            det, thresh = detector.detect(data)
            detections[mask] = det[mask]
            thresholds[mask] = thresh[mask]
    
    return detections, thresholds


def classify_clutter_environment(data: np.ndarray, 
                                window_size: int = 16) -> np.ndarray:
    """
    Classify clutter environment for adaptive CFAR selection.
    
    Args:
        data: Input radar data
        window_size: Size of analysis window
        
    Returns:
        clutter_map: Classification map (0: homogeneous, 1: edge, 2: multiple targets)
    """
    if data.ndim == 1:
        return classify_clutter_1d(data, window_size)
    elif data.ndim == 2:
        clutter_map = np.zeros(data.shape, dtype=int)
        for i in range(data.shape[0]):
            clutter_map[i, :] = classify_clutter_1d(data[i, :], window_size)
        return clutter_map
    else:
        raise ValueError("Input data must be 1D or 2D")


def classify_clutter_1d(data: np.ndarray, window_size: int = 16) -> np.ndarray:
    """1D clutter environment classification."""
    n_samples = len(data)
    clutter_map = np.zeros(n_samples, dtype=int)
    half_window = window_size // 2
    
    for i in range(half_window, n_samples - half_window):
        window_data = data[i-half_window:i+half_window+1]
        
        # Calculate statistics
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        cv = std_val / (mean_val + 1e-10)  # Coefficient of variation
        
        # Edge detection using gradient
        gradient = np.abs(np.gradient(window_data))
        max_gradient = np.max(gradient)
        
        # Target detection using peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(window_data, height=mean_val + 2*std_val)
        n_targets = len(peaks)
        
        # Classification logic
        if max_gradient > 2 * std_val:
            clutter_map[i] = 1  # Edge environment
        elif n_targets > 2:
            clutter_map[i] = 2  # Multiple targets
        else:
            clutter_map[i] = 0  # Homogeneous clutter
    
    return clutter_map