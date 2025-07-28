"""
Maritime Radar Data Simulator
============================

Simulation of maritime radar data including:
- Sea clutter modeling with various sea states
- Moving target generation with realistic trajectories
- Radar system modeling (range, azimuth, Doppler)
- Environmental effects (weather, multipath)
- Dataset generation for training and evaluation
"""

import numpy as np
import scipy.signal as signal
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json


@dataclass
class RadarParameters:
    """Radar system parameters."""
    carrier_freq: float = 9.4e9  # Hz (X-band)
    bandwidth: float = 10e6      # Hz
    prf: float = 1000.0         # Hz
    peak_power: float = 100e3    # Watts
    antenna_gain: float = 35.0   # dB
    noise_figure: float = 3.0    # dB
    system_losses: float = 5.0   # dB
    
    # Antenna parameters
    azimuth_beamwidth: float = 1.5  # degrees
    elevation_beamwidth: float = 20.0  # degrees
    antenna_height: float = 20.0  # meters
    
    # Range parameters
    max_range: float = 50000.0  # meters
    range_resolution: float = 15.0  # meters
    
    # Processing parameters
    coherent_integration_time: float = 0.1  # seconds
    

@dataclass
class EnvironmentParameters:
    """Environmental parameters affecting radar performance."""
    sea_state: int = 3  # Beaufort scale (0-9)
    wind_speed: float = 15.0  # m/s
    wind_direction: float = 0.0  # degrees
    wave_height: float = 2.0  # meters
    temperature: float = 15.0  # Celsius
    humidity: float = 70.0  # percent
    rain_rate: float = 0.0  # mm/hr
    

@dataclass
class TargetParameters:
    """Maritime target parameters."""
    initial_position: Tuple[float, float]  # (x, y) in meters
    initial_velocity: Tuple[float, float]  # (vx, vy) in m/s
    rcs_base: float = 100.0  # m^2 (base RCS)
    rcs_fluctuation: float = 5.0  # dB std
    length: float = 50.0  # meters
    width: float = 10.0  # meters
    target_type: str = 'ship'  # 'ship', 'boat', 'submarine'
    maneuver_capability: float = 0.1  # probability of maneuvering per frame


class SeaClutterModel:
    """Sea clutter modeling for maritime radar."""
    
    def __init__(self, radar_params: RadarParameters, env_params: EnvironmentParameters):
        self.radar_params = radar_params
        self.env_params = env_params
        self.wavelength = 3e8 / radar_params.carrier_freq
        
    def generate_clutter_rcs(self, range_bins: np.ndarray, azimuth_bins: np.ndarray) -> np.ndarray:
        """
        Generate sea clutter RCS map.
        
        Args:
            range_bins: Range bin centers (meters)
            azimuth_bins: Azimuth bin centers (degrees)
            
        Returns:
            clutter_rcs: Clutter RCS values [n_azimuth, n_range] (dBsm)
        """
        n_azimuth = len(azimuth_bins)
        n_range = len(range_bins)
        
        # Base clutter level depends on sea state
        base_clutter_level = -30 - 5 * self.env_params.sea_state  # dBsm/m^2
        
        # Range dependence (R^-3 for distributed clutter)
        range_factor = -30 * np.log10(range_bins / 1000.0)  # Normalize to km
        
        # Angular dependence (lower at grazing angles)
        # Assuming flat earth model for simplicity
        grazing_angles = np.degrees(np.arctan(self.radar_params.antenna_height / range_bins))
        grazing_factor = 10 * np.log10(np.sin(np.radians(grazing_angles)) + 0.01)
        
        # Create 2D grid
        range_grid, azimuth_grid = np.meshgrid(range_bins, azimuth_bins)
        range_factor_2d = np.interp(range_grid, range_bins, range_factor)
        grazing_factor_2d = np.interp(range_grid, range_bins, grazing_factor)
        
        # Wind direction effect
        wind_effect = 3 * np.cos(np.radians(azimuth_grid - self.env_params.wind_direction))
        
        # Combine effects
        clutter_rcs = (base_clutter_level + range_factor_2d + 
                      grazing_factor_2d + wind_effect)
        
        # Add spatial correlation and texture
        clutter_rcs = self._add_spatial_correlation(clutter_rcs)
        
        return clutter_rcs
    
    def generate_clutter_doppler(self, clutter_rcs: np.ndarray, 
                                range_bins: np.ndarray, 
                                azimuth_bins: np.ndarray) -> np.ndarray:
        """
        Generate Doppler spectrum for sea clutter.
        
        Args:
            clutter_rcs: Clutter RCS map
            range_bins: Range bins
            azimuth_bins: Azimuth bins
            
        Returns:
            doppler_spectrum: Doppler frequencies for each range-azimuth cell
        """
        n_azimuth, n_range = clutter_rcs.shape
        
        # Bragg scattering gives dominant Doppler components
        bragg_velocity = np.sqrt(9.81 * self.wavelength / (4 * np.pi))  # m/s
        
        # Wind effect on surface waves
        wind_component = self.env_params.wind_speed * 0.1  # Simple model
        
        # Generate Doppler for each cell
        doppler_spectrum = np.zeros((n_azimuth, n_range))
        
        for i, azimuth in enumerate(azimuth_bins):
            # Wind direction relative to radar look angle
            relative_wind = azimuth - self.env_params.wind_direction
            wind_radial = self.env_params.wind_speed * np.cos(np.radians(relative_wind))
            
            # Dominant Doppler frequency
            dominant_doppler = 2 * (bragg_velocity + wind_component * np.sign(wind_radial)) / self.wavelength
            
            # Add spread based on sea state
            doppler_spread = 0.5 + 0.2 * self.env_params.sea_state
            
            # Generate spectrum for this azimuth
            for j in range(n_range):
                if clutter_rcs[i, j] > -60:  # Only for significant clutter
                    # Multiple spectral lines for rough sea
                    doppler_spectrum[i, j] = dominant_doppler + np.random.normal(0, doppler_spread)
        
        return doppler_spectrum
    
    def _add_spatial_correlation(self, clutter_map: np.ndarray) -> np.ndarray:
        """Add spatial correlation to clutter map."""
        # Apply 2D smoothing filter for spatial correlation
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        smoothed = signal.convolve2d(clutter_map, kernel, mode='same', boundary='symm')
        
        # Add random texture
        texture = np.random.normal(0, 2, clutter_map.shape)  # 2 dB std
        
        return smoothed + texture


class TargetMotionModel:
    """Maritime target motion modeling."""
    
    def __init__(self, target_params: TargetParameters):
        self.target_params = target_params
        self.position = np.array(target_params.initial_position, dtype=np.float64)
        self.velocity = np.array(target_params.initial_velocity, dtype=np.float64)
        self.acceleration = np.array([0.0, 0.0])
        
        # Track history
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        
    def update(self, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update target motion.
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            position: Updated position (x, y)
            velocity: Updated velocity (vx, vy)
        """
        # Check for maneuver
        if np.random.random() < self.target_params.maneuver_capability * dt:
            self._execute_maneuver()
        
        # Update kinematics
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt
        
        # Add some random motion (sea state effects)
        position_noise = np.random.normal(0, 1, 2)  # 1 meter std
        velocity_noise = np.random.normal(0, 0.1, 2)  # 0.1 m/s std
        
        self.position += position_noise
        self.velocity += velocity_noise
        
        # Record trajectory
        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        
        return self.position.copy(), self.velocity.copy()
    
    def _execute_maneuver(self):
        """Execute a random maneuver."""
        if self.target_params.target_type == 'ship':
            # Ship maneuvers: turn or speed change
            if np.random.random() < 0.5:
                # Turn maneuver
                turn_rate = np.random.uniform(-5, 5)  # degrees per second
                current_heading = np.arctan2(self.velocity[1], self.velocity[0])
                new_heading = current_heading + np.radians(turn_rate)
                speed = np.linalg.norm(self.velocity)
                self.velocity = speed * np.array([np.cos(new_heading), np.sin(new_heading)])
            else:
                # Speed change
                speed_change = np.random.uniform(-2, 2)  # m/s
                current_speed = np.linalg.norm(self.velocity)
                new_speed = max(1, current_speed + speed_change)  # Minimum 1 m/s
                self.velocity = (new_speed / current_speed) * self.velocity
        
        elif self.target_params.target_type == 'boat':
            # Boats are more agile
            heading_change = np.random.uniform(-20, 20)  # degrees
            speed_change = np.random.uniform(-5, 5)  # m/s
            
            current_heading = np.arctan2(self.velocity[1], self.velocity[0])
            new_heading = current_heading + np.radians(heading_change)
            current_speed = np.linalg.norm(self.velocity)
            new_speed = max(0.5, current_speed + speed_change)
            
            self.velocity = new_speed * np.array([np.cos(new_heading), np.sin(new_heading)])
    
    def get_rcs(self, radar_azimuth: float) -> float:
        """
        Get target RCS based on aspect angle.
        
        Args:
            radar_azimuth: Radar azimuth to target (degrees)
            
        Returns:
            rcs: Target RCS (dBsm)
        """
        # Calculate aspect angle
        target_heading = np.degrees(np.arctan2(self.velocity[1], self.velocity[0]))
        aspect_angle = abs(radar_azimuth - target_heading)
        aspect_angle = min(aspect_angle, 360 - aspect_angle)  # Take smaller angle
        
        # RCS depends on aspect angle
        if aspect_angle < 30 or aspect_angle > 150:
            # Bow/stern aspects - lower RCS
            rcs_factor = -5  # dB
        elif 60 < aspect_angle < 120:
            # Beam aspect - higher RCS
            rcs_factor = 3  # dB
        else:
            # Intermediate aspects
            rcs_factor = 0  # dB
        
        # Add fluctuation
        fluctuation = np.random.normal(0, self.target_params.rcs_fluctuation)
        
        base_rcs_db = 10 * np.log10(self.target_params.rcs_base)
        return base_rcs_db + rcs_factor + fluctuation


class MaritimeRadarSimulator:
    """Complete maritime radar simulator."""
    
    def __init__(self, radar_params: RadarParameters, env_params: EnvironmentParameters):
        self.radar_params = radar_params
        self.env_params = env_params
        
        # Initialize components
        self.sea_clutter = SeaClutterModel(radar_params, env_params)
        self.targets = []
        
        # Simulation grid
        self.range_bins = np.arange(100, radar_params.max_range, radar_params.range_resolution)
        self.azimuth_bins = np.arange(0, 360, radar_params.azimuth_beamwidth)
        
        # Generate base clutter map
        self.clutter_rcs_map = self.sea_clutter.generate_clutter_rcs(
            self.range_bins, self.azimuth_bins)
        self.clutter_doppler_map = self.sea_clutter.generate_clutter_doppler(
            self.clutter_rcs_map, self.range_bins, self.azimuth_bins)
    
    def add_target(self, target_params: TargetParameters):
        """Add a target to the simulation."""
        target = TargetMotionModel(target_params)
        self.targets.append(target)
    
    def simulate_frame(self, frame_time: float) -> Dict:
        """
        Simulate one radar frame.
        
        Args:
            frame_time: Current simulation time (seconds)
            
        Returns:
            frame_data: Dictionary containing detection data
        """
        # Update all targets
        target_states = []
        for target in self.targets:
            position, velocity = target.update()
            target_states.append({
                'position': position,
                'velocity': velocity,
                'target': target
            })
        
        # Generate detections
        detections = []
        ground_truth = []
        
        # Process targets
        for i, state in enumerate(target_states):
            pos = state['position']
            vel = state['velocity']
            target = state['target']
            
            # Convert to polar coordinates
            range_val = np.linalg.norm(pos)
            azimuth_val = np.degrees(np.arctan2(pos[1], pos[0]))
            azimuth_val = azimuth_val % 360  # Normalize to [0, 360)
            
            # Check if in coverage
            if range_val > self.radar_params.max_range:
                continue
            
            # Calculate Doppler
            radial_velocity = np.dot(vel, pos) / range_val
            doppler_freq = 2 * radial_velocity / (3e8 / self.radar_params.carrier_freq)
            
            # Get target RCS
            rcs_db = target.get_rcs(azimuth_val)
            
            # Calculate SNR
            snr = self._calculate_snr(range_val, rcs_db)
            
            # Detection probability based on SNR
            pd = self._detection_probability(snr)
            
            if np.random.random() < pd:
                # Add measurement noise
                range_noise = np.random.normal(0, self.radar_params.range_resolution / 4)
                azimuth_noise = np.random.normal(0, self.radar_params.azimuth_beamwidth / 4)
                doppler_noise = np.random.normal(0, 0.5)  # 0.5 Hz std
                
                detection = {
                    'range': range_val + range_noise,
                    'azimuth': azimuth_val + azimuth_noise,
                    'doppler': doppler_freq + doppler_noise,
                    'rcs': rcs_db,
                    'snr': snr,
                    'target_id': i,
                    'frame_time': frame_time
                }
                
                # Convert to Cartesian for processing
                x = detection['range'] * np.cos(np.radians(detection['azimuth']))
                y = detection['range'] * np.sin(np.radians(detection['azimuth']))
                vx = radial_velocity * np.cos(np.radians(detection['azimuth']))
                vy = radial_velocity * np.sin(np.radians(detection['azimuth']))
                
                detection.update({
                    'x': x,
                    'y': y,
                    'vx': vx,
                    'vy': vy
                })
                
                detections.append(detection)
            
            # Always add to ground truth
            gt = {
                'id': i,
                'x': pos[0],
                'y': pos[1],
                'vx': vel[0],
                'vy': vel[1],
                'range': range_val,
                'azimuth': azimuth_val,
                'doppler': doppler_freq,
                'rcs': rcs_db,
                'frame_time': frame_time
            }
            ground_truth.append(gt)
        
        # Add sea clutter detections
        clutter_detections = self._generate_clutter_detections(frame_time)
        
        return {
            'frame_time': frame_time,
            'target_detections': detections,
            'clutter_detections': clutter_detections,
            'all_detections': detections + clutter_detections,
            'ground_truth': ground_truth,
            'environment': {
                'sea_state': self.env_params.sea_state,
                'wind_speed': self.env_params.wind_speed,
                'wind_direction': self.env_params.wind_direction
            }
        }
    
    def _calculate_snr(self, range_val: float, target_rcs_db: float) -> float:
        """Calculate target SNR."""
        # Radar equation
        pt = self.radar_params.peak_power  # Watts
        gt = 10**(self.radar_params.antenna_gain / 10)  # Linear
        sigma = 10**(target_rcs_db / 10)  # m^2
        
        # Received power
        pr = (pt * gt**2 * (3e8 / self.radar_params.carrier_freq)**2 * sigma) / \
             ((4 * np.pi)**3 * range_val**4)
        
        # Noise power
        k = 1.38e-23  # Boltzmann constant
        t0 = 290  # K
        nf = 10**(self.radar_params.noise_figure / 10)
        bn = self.radar_params.bandwidth
        pn = k * t0 * nf * bn
        
        # SNR
        snr_linear = pr / pn
        snr_db = 10 * np.log10(snr_linear)
        
        # Account for processing gain
        processing_gain = 10 * np.log10(self.radar_params.prf * self.radar_params.coherent_integration_time)
        
        return snr_db + processing_gain
    
    def _detection_probability(self, snr_db: float) -> float:
        """Calculate detection probability based on SNR."""
        # Simple model: Pd = 0.5 * (1 + erf((SNR - SNR_threshold) / sqrt(2) / sigma))
        snr_threshold = 13.0  # dB
        sigma = 2.0  # dB
        
        from scipy.special import erf
        pd = 0.5 * (1 + erf((snr_db - snr_threshold) / (np.sqrt(2) * sigma)))
        return max(0, min(1, pd))
    
    def _generate_clutter_detections(self, frame_time: float) -> List[Dict]:
        """Generate false alarms from sea clutter."""
        clutter_detections = []
        
        # False alarm rate based on threshold and clutter level
        base_far = 1e-6  # Base false alarm rate per resolution cell
        
        for i, azimuth in enumerate(self.azimuth_bins[::5]):  # Sample every 5th azimuth
            for j, range_val in enumerate(self.range_bins[::3]):  # Sample every 3rd range
                # Get clutter level at this cell
                clutter_level = self.clutter_rcs_map[min(i, len(self.azimuth_bins)-1), 
                                                   min(j, len(self.range_bins)-1)]
                
                # Higher clutter increases false alarm rate
                far = base_far * 10**(clutter_level / 20)
                
                if np.random.random() < far:
                    # Generate false alarm
                    range_noise = np.random.normal(0, self.radar_params.range_resolution / 2)
                    azimuth_noise = np.random.normal(0, self.radar_params.azimuth_beamwidth / 2)
                    
                    # Clutter Doppler characteristics
                    clutter_doppler = self.clutter_doppler_map[min(i, len(self.azimuth_bins)-1),
                                                             min(j, len(self.range_bins)-1)]
                    doppler_noise = np.random.normal(0, 1.0)  # 1 Hz std for clutter
                    
                    detection = {
                        'range': range_val + range_noise,
                        'azimuth': azimuth + azimuth_noise,
                        'doppler': clutter_doppler + doppler_noise,
                        'rcs': clutter_level + np.random.normal(0, 3),  # 3 dB std
                        'snr': clutter_level + 15,  # Rough estimate
                        'target_id': -1,  # Clutter marker
                        'frame_time': frame_time,
                        'is_clutter': True
                    }
                    
                    # Convert to Cartesian
                    x = detection['range'] * np.cos(np.radians(detection['azimuth']))
                    y = detection['range'] * np.sin(np.radians(detection['azimuth']))
                    
                    detection.update({
                        'x': x,
                        'y': y,
                        'vx': 0.0,  # Clutter has no net velocity
                        'vy': 0.0
                    })
                    
                    clutter_detections.append(detection)
        
        return clutter_detections
    
    def run_simulation(self, duration: float, dt: float = 1.0) -> List[Dict]:
        """
        Run complete simulation.
        
        Args:
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            
        Returns:
            simulation_data: List of frame data dictionaries
        """
        simulation_data = []
        current_time = 0.0
        
        while current_time < duration:
            frame_data = self.simulate_frame(current_time)
            simulation_data.append(frame_data)
            current_time += dt
        
        return simulation_data
    
    def save_scenario(self, filename: str):
        """Save simulation scenario configuration."""
        scenario = {
            'radar_params': {
                'carrier_freq': self.radar_params.carrier_freq,
                'bandwidth': self.radar_params.bandwidth,
                'prf': self.radar_params.prf,
                'peak_power': self.radar_params.peak_power,
                'antenna_gain': self.radar_params.antenna_gain,
                'noise_figure': self.radar_params.noise_figure,
                'max_range': self.radar_params.max_range,
                'range_resolution': self.radar_params.range_resolution,
                'azimuth_beamwidth': self.radar_params.azimuth_beamwidth
            },
            'env_params': {
                'sea_state': self.env_params.sea_state,
                'wind_speed': self.env_params.wind_speed,
                'wind_direction': self.env_params.wind_direction,
                'wave_height': self.env_params.wave_height
            },
            'targets': [
                {
                    'initial_position': target.target_params.initial_position,
                    'initial_velocity': target.target_params.initial_velocity,
                    'rcs_base': target.target_params.rcs_base,
                    'target_type': target.target_params.target_type
                }
                for target in self.targets
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(scenario, f, indent=2)


def create_test_scenario() -> MaritimeRadarSimulator:
    """Create a test scenario with multiple targets and realistic conditions."""
    # Radar configuration
    radar_params = RadarParameters(
        carrier_freq=9.4e9,
        max_range=30000,
        range_resolution=15.0,
        azimuth_beamwidth=1.0
    )
    
    # Environmental conditions
    env_params = EnvironmentParameters(
        sea_state=4,
        wind_speed=12.0,
        wind_direction=45.0
    )
    
    # Create simulator
    simulator = MaritimeRadarSimulator(radar_params, env_params)
    
    # Add targets
    targets = [
        TargetParameters(
            initial_position=(5000, 8000),
            initial_velocity=(8, -3),
            rcs_base=150,
            target_type='ship'
        ),
        TargetParameters(
            initial_position=(-7000, 12000),
            initial_velocity=(5, 5),
            rcs_base=80,
            target_type='ship'
        ),
        TargetParameters(
            initial_position=(15000, -3000),
            initial_velocity=(-12, 8),
            rcs_base=25,
            target_type='boat'
        )
    ]
    
    for target_params in targets:
        simulator.add_target(target_params)
    
    return simulator