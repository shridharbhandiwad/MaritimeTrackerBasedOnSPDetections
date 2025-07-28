"""
Doppler Filtering and Preprocessing for Maritime Radar
====================================================

Implementation of Doppler filtering, angle gating, and coordinate normalization
specifically designed for maritime radar applications with sea clutter rejection.
"""

import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from typing import Tuple, Optional, Dict, List
import warnings


class DopplerProcessor:
    """Doppler processing for maritime radar data."""
    
    def __init__(self, 
                 prf: float,
                 carrier_freq: float = 9.4e9,
                 coherent_integration_time: float = 0.1,
                 min_doppler: float = -100.0,
                 max_doppler: float = 100.0):
        """
        Initialize Doppler processor.
        
        Args:
            prf: Pulse repetition frequency (Hz)
            carrier_freq: Radar carrier frequency (Hz)
            coherent_integration_time: Coherent integration time (s)
            min_doppler: Minimum Doppler frequency (Hz)
            max_doppler: Maximum Doppler frequency (Hz)
        """
        self.prf = prf
        self.carrier_freq = carrier_freq
        self.cit = coherent_integration_time
        self.min_doppler = min_doppler
        self.max_doppler = max_doppler
        
        # Calculate wavelength and maximum unambiguous velocity
        self.wavelength = 3e8 / carrier_freq
        self.max_velocity = self.wavelength * prf / 4  # m/s
        
        # Number of pulses for coherent integration
        self.n_pulses = int(prf * coherent_integration_time)
        
    def doppler_fft(self, pulse_data: np.ndarray, 
                   window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Doppler FFT on pulse data.
        
        Args:
            pulse_data: Input pulse data [n_pulses, n_range_bins]
            window_type: Window function type
            
        Returns:
            doppler_spectrum: Doppler spectrum [n_doppler_bins, n_range_bins]
            doppler_freqs: Doppler frequency bins
        """
        n_pulses, n_range = pulse_data.shape
        
        # Apply window function
        if window_type == 'hann':
            window = np.hanning(n_pulses)
        elif window_type == 'hamming':
            window = np.hamming(n_pulses)
        elif window_type == 'blackman':
            window = np.blackman(n_pulses)
        else:
            window = np.ones(n_pulses)
        
        # Apply window to each range bin
        windowed_data = pulse_data * window[:, np.newaxis]
        
        # Perform FFT along pulse dimension
        doppler_spectrum = fft(windowed_data, axis=0)
        doppler_spectrum = fftshift(doppler_spectrum, axes=0)
        
        # Calculate Doppler frequency bins
        doppler_freqs = fftshift(fftfreq(n_pulses, 1/self.prf))
        
        return doppler_spectrum, doppler_freqs
    
    def velocity_to_doppler(self, velocity: float) -> float:
        """Convert radial velocity to Doppler frequency."""
        return 2 * velocity / self.wavelength
    
    def doppler_to_velocity(self, doppler: float) -> float:
        """Convert Doppler frequency to radial velocity."""
        return doppler * self.wavelength / 2
    
    def create_doppler_filter(self, doppler_freqs: np.ndarray,
                             filter_type: str = 'bandpass',
                             sea_clutter_params: Optional[Dict] = None) -> np.ndarray:
        """
        Create Doppler filter for sea clutter rejection.
        
        Args:
            doppler_freqs: Doppler frequency bins
            filter_type: Type of filter ('bandpass', 'notch', 'adaptive')
            sea_clutter_params: Parameters for sea clutter filtering
            
        Returns:
            filter_response: Filter response function
        """
        if sea_clutter_params is None:
            sea_clutter_params = {
                'sea_state': 3,  # Sea state (0-9)
                'wind_speed': 10,  # Wind speed (m/s)
                'radar_height': 20,  # Radar height (m)
            }
        
        # Estimate sea clutter Doppler spread
        sea_clutter_spread = self._estimate_sea_clutter_spread(sea_clutter_params)
        
        if filter_type == 'bandpass':
            # Simple bandpass filter
            filter_response = np.ones_like(doppler_freqs)
            mask = (np.abs(doppler_freqs) < sea_clutter_spread)
            filter_response[mask] = 0.1  # Attenuate sea clutter region
            
        elif filter_type == 'notch':
            # Notch filter at zero Doppler
            notch_width = sea_clutter_spread
            filter_response = np.ones_like(doppler_freqs)
            mask = (np.abs(doppler_freqs) < notch_width)
            filter_response[mask] = 0.01
            
        elif filter_type == 'adaptive':
            # Adaptive filter based on sea state
            filter_response = self._adaptive_sea_clutter_filter(
                doppler_freqs, sea_clutter_params)
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return filter_response
    
    def _estimate_sea_clutter_spread(self, params: Dict) -> float:
        """Estimate sea clutter Doppler spread based on environmental conditions."""
        sea_state = params.get('sea_state', 3)
        wind_speed = params.get('wind_speed', 10)
        radar_height = params.get('radar_height', 20)
        
        # Empirical model for sea clutter Doppler spread
        # Based on sea state and wind conditions
        base_spread = 0.5 + 0.3 * sea_state  # Hz
        wind_factor = 1 + 0.1 * (wind_speed - 10) / 10  # Wind effect
        height_factor = 1 + 0.05 * (radar_height - 20) / 20  # Height effect
        
        spread = base_spread * wind_factor * height_factor
        return max(spread, 0.1)  # Minimum spread
    
    def _adaptive_sea_clutter_filter(self, doppler_freqs: np.ndarray, 
                                    params: Dict) -> np.ndarray:
        """Create adaptive sea clutter filter."""
        sea_state = params.get('sea_state', 3)
        
        # Multi-stage filter design
        filter_response = np.ones_like(doppler_freqs)
        
        # Stage 1: Zero Doppler notch
        notch_width = 0.2 + 0.1 * sea_state
        mask1 = (np.abs(doppler_freqs) < notch_width)
        filter_response[mask1] = 0.01
        
        # Stage 2: Low Doppler attenuation
        low_doppler_width = 1.0 + 0.5 * sea_state
        mask2 = (np.abs(doppler_freqs) < low_doppler_width) & ~mask1
        filter_response[mask2] = 0.1 + 0.1 * (np.abs(doppler_freqs[mask2]) / low_doppler_width)
        
        # Stage 3: Smooth transition
        transition_width = 2.0
        mask3 = (np.abs(doppler_freqs) < low_doppler_width + transition_width) & ~mask2 & ~mask1
        transition_factor = (np.abs(doppler_freqs[mask3]) - low_doppler_width) / transition_width
        filter_response[mask3] = 0.2 + 0.8 * transition_factor
        
        return filter_response
    
    def apply_doppler_filter(self, doppler_spectrum: np.ndarray,
                           filter_response: np.ndarray) -> np.ndarray:
        """Apply Doppler filter to spectrum."""
        return doppler_spectrum * filter_response[:, np.newaxis]
    
    def doppler_clustering(self, doppler_spectrum: np.ndarray,
                          doppler_freqs: np.ndarray,
                          threshold: float = 0.7) -> List[Dict]:
        """
        Cluster detections in Doppler domain.
        
        Args:
            doppler_spectrum: Doppler spectrum [n_doppler, n_range]
            doppler_freqs: Doppler frequency bins
            threshold: Detection threshold
            
        Returns:
            clusters: List of cluster dictionaries
        """
        # Magnitude spectrum
        magnitude = np.abs(doppler_spectrum)
        
        # Find peaks above threshold
        peak_mask = magnitude > threshold * np.max(magnitude)
        
        # Get peak coordinates
        doppler_indices, range_indices = np.where(peak_mask)
        
        if len(doppler_indices) == 0:
            return []
        
        # Cluster nearby peaks
        clusters = []
        visited = np.zeros(len(doppler_indices), dtype=bool)
        
        for i, (d_idx, r_idx) in enumerate(zip(doppler_indices, range_indices)):
            if visited[i]:
                continue
                
            # Start new cluster
            cluster = {
                'doppler_indices': [d_idx],
                'range_indices': [r_idx],
                'doppler_freqs': [doppler_freqs[d_idx]],
                'magnitudes': [magnitude[d_idx, r_idx]]
            }
            visited[i] = True
            
            # Find nearby peaks
            for j, (d_idx2, r_idx2) in enumerate(zip(doppler_indices, range_indices)):
                if visited[j]:
                    continue
                    
                # Check if peaks are close in Doppler-range space
                doppler_dist = abs(d_idx - d_idx2)
                range_dist = abs(r_idx - r_idx2)
                
                if doppler_dist <= 2 and range_dist <= 3:  # Clustering thresholds
                    cluster['doppler_indices'].append(d_idx2)
                    cluster['range_indices'].append(r_idx2)
                    cluster['doppler_freqs'].append(doppler_freqs[d_idx2])
                    cluster['magnitudes'].append(magnitude[d_idx2, r_idx2])
                    visited[j] = True
            
            # Calculate cluster statistics
            cluster['mean_doppler'] = np.mean(cluster['doppler_freqs'])
            cluster['mean_range_idx'] = np.mean(cluster['range_indices'])
            cluster['max_magnitude'] = np.max(cluster['magnitudes'])
            cluster['size'] = len(cluster['doppler_indices'])
            
            clusters.append(cluster)
        
        return clusters


class AngleGating:
    """Angle gating for maritime radar."""
    
    def __init__(self, 
                 antenna_beamwidth: float = 1.5,
                 sector_limits: Optional[Tuple[float, float]] = None):
        """
        Initialize angle gating.
        
        Args:
            antenna_beamwidth: 3dB beamwidth (degrees)
            sector_limits: Azimuth sector limits (min_deg, max_deg)
        """
        self.beamwidth = antenna_beamwidth
        self.sector_limits = sector_limits
    
    def apply_sector_gating(self, data: np.ndarray, 
                           azimuths: np.ndarray) -> np.ndarray:
        """Apply azimuth sector gating."""
        if self.sector_limits is None:
            return data
        
        min_az, max_az = self.sector_limits
        
        # Handle azimuth wrapping
        if max_az < min_az:  # Sector crosses 0 degrees
            mask = (azimuths >= min_az) | (azimuths <= max_az)
        else:
            mask = (azimuths >= min_az) & (azimuths <= max_az)
        
        # Apply gating
        gated_data = data.copy()
        gated_data[~mask, :] = 0
        
        return gated_data
    
    def angle_dependent_threshold(self, azimuths: np.ndarray,
                                 base_threshold: float = 1.0,
                                 land_sectors: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Calculate angle-dependent detection thresholds.
        
        Args:
            azimuths: Azimuth angles (degrees)
            base_threshold: Base threshold value
            land_sectors: List of (min_az, max_az) tuples for land sectors
            
        Returns:
            thresholds: Angle-dependent thresholds
        """
        thresholds = np.full_like(azimuths, base_threshold)
        
        if land_sectors is not None:
            for min_az, max_az in land_sectors:
                if max_az < min_az:  # Wrapping case
                    mask = (azimuths >= min_az) | (azimuths <= max_az)
                else:
                    mask = (azimuths >= min_az) & (azimuths <= max_az)
                
                # Increase threshold in land sectors to reduce false alarms
                thresholds[mask] *= 2.0
        
        return thresholds


class CoordinateProcessor:
    """Coordinate projection and normalization for maritime radar."""
    
    def __init__(self, 
                 radar_position: Tuple[float, float] = (0.0, 0.0),
                 coordinate_system: str = 'cartesian'):
        """
        Initialize coordinate processor.
        
        Args:
            radar_position: Radar position (lat, lon) or (x, y)
            coordinate_system: 'cartesian' or 'geographic'
        """
        self.radar_position = radar_position
        self.coordinate_system = coordinate_system
    
    def polar_to_cartesian(self, ranges: np.ndarray, 
                          azimuths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert polar coordinates to Cartesian."""
        # Convert azimuth to mathematical convention (CCW from East)
        theta_rad = np.radians(90 - azimuths)
        
        x = ranges * np.cos(theta_rad)
        y = ranges * np.sin(theta_rad)
        
        return x, y
    
    def cartesian_to_polar(self, x: np.ndarray, 
                          y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Cartesian coordinates to polar."""
        ranges = np.sqrt(x**2 + y**2)
        
        # Convert to navigation convention (CW from North)
        theta_math = np.arctan2(y, x)
        azimuths = np.degrees(90 - theta_math)
        
        # Normalize azimuth to [0, 360)
        azimuths = np.mod(azimuths, 360)
        
        return ranges, azimuths
    
    def normalize_coordinates(self, detections: np.ndarray,
                            method: str = 'minmax',
                            feature_ranges: Optional[Dict] = None) -> np.ndarray:
        """
        Normalize detection coordinates for neural network input.
        
        Args:
            detections: Detection array [n_detections, n_features]
            method: Normalization method ('minmax', 'zscore', 'robust')
            feature_ranges: Optional feature ranges for normalization
            
        Returns:
            normalized_detections: Normalized detection array
        """
        if feature_ranges is None:
            feature_ranges = {
                'range': (0, 50000),      # meters
                'azimuth': (0, 360),      # degrees
                'doppler': (-100, 100),   # Hz
                'rcs': (-40, 40),         # dBsm
                'snr': (0, 40)            # dB
            }
        
        normalized = detections.copy()
        
        for i, (feature, (min_val, max_val)) in enumerate(feature_ranges.items()):
            if i >= detections.shape[1]:
                break
                
            feature_data = detections[:, i]
            
            if method == 'minmax':
                normalized[:, i] = (feature_data - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                normalized[:, i] = (feature_data - mean_val) / (std_val + 1e-8)
            elif method == 'robust':
                median_val = np.median(feature_data)
                mad = np.median(np.abs(feature_data - median_val))
                normalized[:, i] = (feature_data - median_val) / (mad + 1e-8)
        
        return normalized
    
    def project_to_local_grid(self, detections: np.ndarray,
                             grid_size: Tuple[int, int] = (256, 256),
                             extent: Tuple[float, float, float, float] = (-25000, 25000, -25000, 25000)) -> np.ndarray:
        """
        Project detections to local grid for CNN processing.
        
        Args:
            detections: Detection array [n_detections, >=2] (x, y, ...)
            grid_size: Output grid size (height, width)
            extent: Grid extent (x_min, x_max, y_min, y_max) in meters
            
        Returns:
            grid: Detection grid [height, width]
        """
        x_min, x_max, y_min, y_max = extent
        height, width = grid_size
        
        # Extract x, y coordinates
        x_coords = detections[:, 0]
        y_coords = detections[:, 1]
        
        # Convert to grid indices
        x_indices = np.floor((x_coords - x_min) / (x_max - x_min) * width).astype(int)
        y_indices = np.floor((y_coords - y_min) / (y_max - y_min) * height).astype(int)
        
        # Clip to grid bounds
        x_indices = np.clip(x_indices, 0, width - 1)
        y_indices = np.clip(y_indices, 0, height - 1)
        
        # Create grid
        grid = np.zeros((height, width))
        
        # Add detections to grid (accumulate if multiple detections in same cell)
        for i in range(len(detections)):
            if len(detections[i]) > 2:
                # Use additional features (e.g., RCS, SNR) as grid values
                grid[y_indices[i], x_indices[i]] += detections[i, 2]
            else:
                grid[y_indices[i], x_indices[i]] += 1
        
        return grid


def extract_detection_features(pulse_data: np.ndarray,
                              range_bins: np.ndarray,
                              azimuth_bins: np.ndarray,
                              detection_mask: np.ndarray,
                              doppler_processor: Optional[DopplerProcessor] = None) -> np.ndarray:
    """
    Extract comprehensive features from radar detections.
    
    Args:
        pulse_data: Raw pulse data [n_pulses, n_range, n_azimuth]
        range_bins: Range bin values
        azimuth_bins: Azimuth bin values
        detection_mask: Binary detection mask
        doppler_processor: Optional Doppler processor
        
    Returns:
        features: Detection features [n_detections, n_features]
    """
    # Find detection locations
    detection_indices = np.where(detection_mask)
    n_detections = len(detection_indices[0])
    
    if n_detections == 0:
        return np.empty((0, 8))
    
    # Initialize feature array
    # Features: [range, azimuth, magnitude, phase, doppler, rcs_estimate, snr, persistence]
    features = np.zeros((n_detections, 8))
    
    for i, (pulse_idx, range_idx, az_idx) in enumerate(zip(*detection_indices)):
        # Basic geometric features
        features[i, 0] = range_bins[range_idx]  # Range
        features[i, 1] = azimuth_bins[az_idx]   # Azimuth
        
        # Signal features
        complex_signal = pulse_data[pulse_idx, range_idx, az_idx]
        features[i, 2] = np.abs(complex_signal)     # Magnitude
        features[i, 3] = np.angle(complex_signal)   # Phase
        
        # Doppler features (if processor available)
        if doppler_processor is not None and pulse_data.shape[0] > 1:
            pulse_window = pulse_data[:, range_idx, az_idx]
            doppler_spectrum, doppler_freqs = doppler_processor.doppler_fft(
                pulse_window.reshape(-1, 1))
            peak_idx = np.argmax(np.abs(doppler_spectrum))
            features[i, 4] = doppler_freqs[peak_idx]  # Doppler frequency
        
        # RCS estimate (simplified)
        range_val = features[i, 0]
        magnitude = features[i, 2]
        # Simple RCS estimation (would need calibration in practice)
        features[i, 5] = 20 * np.log10(magnitude) - 40 * np.log10(range_val / 1000)
        
        # SNR estimate
        local_window = pulse_data[max(0, pulse_idx-2):pulse_idx+3,
                                 max(0, range_idx-2):range_idx+3,
                                 max(0, az_idx-2):az_idx+3]
        noise_level = np.std(local_window[local_window != complex_signal])
        features[i, 6] = 20 * np.log10(magnitude / (noise_level + 1e-10))
        
        # Persistence (would need multiple frames)
        features[i, 7] = 1.0  # Placeholder
    
    return features