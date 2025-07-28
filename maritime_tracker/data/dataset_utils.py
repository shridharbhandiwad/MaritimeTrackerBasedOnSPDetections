"""
Dataset Utilities for Maritime Radar Tracking
============================================

Utilities for loading, preprocessing, and managing maritime radar datasets:
- Real dataset loaders (IPIX, CSIR, RASEX)
- Data preprocessing and augmentation
- Training/validation data preparation
- Ground truth annotation tools
"""

import numpy as np
import pandas as pd
import h5py
import pickle
import os
import json
from typing import List, Dict, Tuple, Optional, Union, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import warnings


class MaritimeRadarDataset(Dataset):
    """PyTorch dataset for maritime radar data."""
    
    def __init__(self, 
                 detection_sequences: List[np.ndarray],
                 ground_truth_sequences: List[List[Dict]],
                 transform: Optional[callable] = None,
                 sequence_length: int = 10,
                 overlap: float = 0.5):
        """
        Initialize dataset.
        
        Args:
            detection_sequences: List of detection arrays [seq_len, n_detections, n_features]
            ground_truth_sequences: List of ground truth sequences
            transform: Optional data transformation
            sequence_length: Length of input sequences
            overlap: Overlap between consecutive sequences
        """
        self.detection_sequences = detection_sequences
        self.ground_truth_sequences = ground_truth_sequences
        self.transform = transform
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        # Generate sequence indices
        self.sequence_indices = self._generate_sequence_indices()
        
    def _generate_sequence_indices(self) -> List[Tuple[int, int]]:
        """Generate indices for sequence extraction."""
        indices = []
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        for seq_idx, detection_seq in enumerate(self.detection_sequences):
            seq_len = len(detection_seq)
            for start_idx in range(0, seq_len - self.sequence_length + 1, step_size):
                indices.append((seq_idx, start_idx))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample."""
        seq_idx, start_idx = self.sequence_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract detection sequence
        detections = self.detection_sequences[seq_idx][start_idx:end_idx]
        
        # Extract ground truth
        gt_sequence = self.ground_truth_sequences[seq_idx][start_idx:end_idx]
        
        # Convert to tensors
        detection_tensor = torch.FloatTensor(detections)
        
        # Prepare ground truth
        gt_tensor = self._prepare_ground_truth(gt_sequence)
        
        sample = {
            'detections': detection_tensor,
            'ground_truth': gt_tensor,
            'sequence_id': seq_idx,
            'start_frame': start_idx
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _prepare_ground_truth(self, gt_sequence: List[Dict]) -> torch.Tensor:
        """Prepare ground truth tensor."""
        max_objects = 10  # Maximum number of objects per frame
        n_frames = len(gt_sequence)
        
        # Initialize tensor [n_frames, max_objects, features]
        # Features: [x, y, vx, vy, track_id, is_valid]
        gt_tensor = torch.zeros(n_frames, max_objects, 6)
        
        for frame_idx, frame_gt in enumerate(gt_sequence):
            for obj_idx, obj in enumerate(frame_gt[:max_objects]):
                gt_tensor[frame_idx, obj_idx, 0] = obj.get('x', 0)
                gt_tensor[frame_idx, obj_idx, 1] = obj.get('y', 0)
                gt_tensor[frame_idx, obj_idx, 2] = obj.get('vx', 0)
                gt_tensor[frame_idx, obj_idx, 3] = obj.get('vy', 0)
                gt_tensor[frame_idx, obj_idx, 4] = obj.get('id', -1)
                gt_tensor[frame_idx, obj_idx, 5] = 1  # is_valid
        
        return gt_tensor


class IPIXDatasetLoader:
    """Loader for IPIX radar dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize IPIX dataset loader.
        
        Args:
            data_path: Path to IPIX dataset directory
        """
        self.data_path = data_path
        
    def load_sequence(self, sequence_name: str) -> Tuple[np.ndarray, Dict]:
        """
        Load IPIX radar sequence.
        
        Args:
            sequence_name: Name of the sequence to load
            
        Returns:
            radar_data: Complex radar data [n_pulses, n_range, n_azimuth]
            metadata: Sequence metadata
        """
        sequence_path = os.path.join(self.data_path, sequence_name)
        
        # Load IPIX data (typically in .mat format)
        try:
            data = loadmat(sequence_path)
            
            # Extract radar data and metadata
            if 'data' in data:
                radar_data = data['data']
            elif 'radarData' in data:
                radar_data = data['radarData']
            else:
                # Try to find the main data array
                data_keys = [k for k in data.keys() if not k.startswith('_')]
                radar_data = data[data_keys[0]]
            
            # Extract metadata
            metadata = {
                'prf': data.get('prf', 1000.0),
                'range_resolution': data.get('range_res', 15.0),
                'azimuth_resolution': data.get('azimuth_res', 1.0),
                'carrier_freq': data.get('fc', 9.4e9),
                'sequence_name': sequence_name
            }
            
            return radar_data, metadata
            
        except Exception as e:
            print(f"Error loading IPIX sequence {sequence_name}: {e}")
            return None, None
    
    def extract_detections(self, radar_data: np.ndarray, 
                          metadata: Dict,
                          cfar_params: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Extract detections from IPIX radar data.
        
        Args:
            radar_data: Complex radar data
            metadata: Sequence metadata
            cfar_params: CFAR detection parameters
            
        Returns:
            detection_sequence: List of detection arrays for each frame
        """
        from ..preprocessing import CACFARDetector, extract_detection_features
        
        if cfar_params is None:
            cfar_params = {'guard_cells': 2, 'reference_cells': 16, 'pfa': 1e-6}
        
        # Initialize CFAR detector
        cfar = CACFARDetector(**cfar_params)
        
        n_pulses, n_range, n_azimuth = radar_data.shape
        detection_sequence = []
        
        # Process each pulse
        for pulse_idx in range(n_pulses):
            pulse_data = radar_data[pulse_idx]
            
            # Apply CFAR detection
            magnitude = np.abs(pulse_data)
            detections, _ = cfar.detect(magnitude)
            
            # Extract detection features
            if np.any(detections):
                # Create range and azimuth bins
                range_bins = np.arange(n_range) * metadata['range_resolution']
                azimuth_bins = np.arange(n_azimuth) * metadata['azimuth_resolution']
                
                # Extract features
                features = extract_detection_features(
                    radar_data[pulse_idx:pulse_idx+1],
                    range_bins, azimuth_bins, detections[np.newaxis, :, :])
                
                detection_sequence.append(features)
            else:
                # No detections in this frame
                detection_sequence.append(np.empty((0, 8)))
        
        return detection_sequence


class CSIRDatasetLoader:
    """Loader for CSIR maritime radar dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_sequence(self, sequence_name: str) -> Tuple[List[np.ndarray], List[Dict]]:
        """Load CSIR dataset sequence."""
        # Implementation depends on CSIR data format
        # This is a template - actual implementation would depend on data structure
        
        detections_file = os.path.join(self.data_path, sequence_name, 'detections.csv')
        gt_file = os.path.join(self.data_path, sequence_name, 'ground_truth.json')
        
        detection_sequence = []
        ground_truth_sequence = []
        
        try:
            # Load detections
            if os.path.exists(detections_file):
                df = pd.read_csv(detections_file)
                
                # Group by frame
                for frame_id in df['frame_id'].unique():
                    frame_detections = df[df['frame_id'] == frame_id]
                    
                    # Convert to numpy array
                    features = frame_detections[['range', 'azimuth', 'doppler', 'rcs', 'snr']].values
                    detection_sequence.append(features)
            
            # Load ground truth
            if os.path.exists(gt_file):
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                    ground_truth_sequence = gt_data.get('tracks', [])
            
            return detection_sequence, ground_truth_sequence
            
        except Exception as e:
            print(f"Error loading CSIR sequence {sequence_name}: {e}")
            return [], []


class RASEXDatasetLoader:
    """Loader for RASEX radar dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_hdf5_sequence(self, filename: str) -> Tuple[np.ndarray, Dict]:
        """Load RASEX HDF5 data file."""
        filepath = os.path.join(self.data_path, filename)
        
        try:
            with h5py.File(filepath, 'r') as f:
                # Load radar data
                radar_data = f['radar_data'][:]
                
                # Load metadata
                metadata = {}
                if 'metadata' in f:
                    for key in f['metadata'].keys():
                        metadata[key] = f['metadata'][key][()]
                
                return radar_data, metadata
                
        except Exception as e:
            print(f"Error loading RASEX file {filename}: {e}")
            return None, None


class DataAugmentation:
    """Data augmentation for maritime radar tracking."""
    
    def __init__(self, 
                 noise_std: float = 0.1,
                 position_jitter: float = 5.0,
                 velocity_jitter: float = 0.5,
                 rotation_angle: float = 10.0):
        """
        Initialize data augmentation.
        
        Args:
            noise_std: Standard deviation of additive noise
            position_jitter: Position jitter in meters
            velocity_jitter: Velocity jitter in m/s
            rotation_angle: Maximum rotation angle in degrees
        """
        self.noise_std = noise_std
        self.position_jitter = position_jitter
        self.velocity_jitter = velocity_jitter
        self.rotation_angle = rotation_angle
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to sample."""
        detections = sample['detections'].clone()
        ground_truth = sample['ground_truth'].clone()
        
        # Add noise to detections
        if len(detections) > 0:
            noise = torch.randn_like(detections) * self.noise_std
            detections += noise
        
        # Add jitter to ground truth positions and velocities
        if len(ground_truth) > 0:
            valid_mask = ground_truth[:, :, 5] > 0  # is_valid mask
            
            # Position jitter
            pos_jitter = torch.randn(ground_truth.shape[0], ground_truth.shape[1], 2) * self.position_jitter
            ground_truth[:, :, 0:2] += pos_jitter * valid_mask.unsqueeze(-1)
            
            # Velocity jitter
            vel_jitter = torch.randn(ground_truth.shape[0], ground_truth.shape[1], 2) * self.velocity_jitter
            ground_truth[:, :, 2:4] += vel_jitter * valid_mask.unsqueeze(-1)
        
        # Apply rotation (optional)
        if np.random.random() < 0.3:  # 30% chance of rotation
            angle = np.random.uniform(-self.rotation_angle, self.rotation_angle)
            detections, ground_truth = self._apply_rotation(detections, ground_truth, angle)
        
        sample['detections'] = detections
        sample['ground_truth'] = ground_truth
        
        return sample
    
    def _apply_rotation(self, detections: torch.Tensor, 
                       ground_truth: torch.Tensor, 
                       angle_deg: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotation transformation."""
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        
        # Rotate detections (assuming first two features are x, y)
        if len(detections) > 0 and detections.shape[-1] >= 2:
            for frame_idx in range(detections.shape[0]):
                if len(detections[frame_idx]) > 0:
                    positions = detections[frame_idx, :, :2]
                    rotated_positions = torch.mm(positions, rotation_matrix.T)
                    detections[frame_idx, :, :2] = rotated_positions
        
        # Rotate ground truth
        if len(ground_truth) > 0:
            valid_mask = ground_truth[:, :, 5] > 0
            for frame_idx in range(ground_truth.shape[0]):
                frame_mask = valid_mask[frame_idx]
                if torch.any(frame_mask):
                    # Rotate positions
                    positions = ground_truth[frame_idx, frame_mask, :2]
                    rotated_positions = torch.mm(positions, rotation_matrix.T)
                    ground_truth[frame_idx, frame_mask, :2] = rotated_positions
                    
                    # Rotate velocities
                    velocities = ground_truth[frame_idx, frame_mask, 2:4]
                    rotated_velocities = torch.mm(velocities, rotation_matrix.T)
                    ground_truth[frame_idx, frame_mask, 2:4] = rotated_velocities
        
        return detections, ground_truth


class TrackingDatasetBuilder:
    """Builder for tracking datasets from various sources."""
    
    def __init__(self, output_dir: str):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def build_from_simulation(self, 
                             simulation_data: List[Dict],
                             dataset_name: str = 'simulated_maritime') -> str:
        """
        Build dataset from simulation data.
        
        Args:
            simulation_data: List of simulation frame data
            dataset_name: Name for the dataset
            
        Returns:
            dataset_path: Path to saved dataset
        """
        detection_sequences = []
        ground_truth_sequences = []
        
        # Extract sequences
        current_detections = []
        current_gt = []
        
        for frame_data in simulation_data:
            # Extract detections (convert to feature array)
            detections = frame_data['all_detections']
            detection_features = []
            
            for det in detections:
                features = [
                    det['range'], det['azimuth'], det['x'], det['y'],
                    det['doppler'], det['rcs'], det['snr'], 
                    1.0 if not det.get('is_clutter', False) else 0.0  # target/clutter flag
                ]
                detection_features.append(features)
            
            detection_array = np.array(detection_features) if detection_features else np.empty((0, 8))
            current_detections.append(detection_array)
            current_gt.append(frame_data['ground_truth'])
        
        detection_sequences.append(current_detections)
        ground_truth_sequences.append(current_gt)
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, f'{dataset_name}.pkl')
        
        dataset_dict = {
            'detection_sequences': detection_sequences,
            'ground_truth_sequences': ground_truth_sequences,
            'metadata': {
                'n_sequences': len(detection_sequences),
                'n_frames': len(simulation_data),
                'feature_names': ['range', 'azimuth', 'x', 'y', 'doppler', 'rcs', 'snr', 'is_target']
            }
        }
        
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"Dataset saved to {dataset_path}")
        return dataset_path
    
    def build_from_real_data(self, 
                            data_loader: Union[IPIXDatasetLoader, CSIRDatasetLoader, RASEXDatasetLoader],
                            sequence_names: List[str],
                            dataset_name: str) -> str:
        """Build dataset from real radar data."""
        detection_sequences = []
        ground_truth_sequences = []
        
        for seq_name in sequence_names:
            print(f"Processing sequence: {seq_name}")
            
            if isinstance(data_loader, IPIXDatasetLoader):
                radar_data, metadata = data_loader.load_sequence(seq_name)
                if radar_data is not None:
                    detections = data_loader.extract_detections(radar_data, metadata)
                    detection_sequences.append(detections)
                    # IPIX typically doesn't have ground truth, so create empty
                    ground_truth_sequences.append([[] for _ in range(len(detections))])
            
            elif isinstance(data_loader, CSIRDatasetLoader):
                detections, gt = data_loader.load_sequence(seq_name)
                if detections and gt:
                    detection_sequences.append(detections)
                    ground_truth_sequences.append(gt)
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, f'{dataset_name}.pkl')
        
        dataset_dict = {
            'detection_sequences': detection_sequences,
            'ground_truth_sequences': ground_truth_sequences,
            'metadata': {
                'n_sequences': len(detection_sequences),
                'source': type(data_loader).__name__,
                'sequence_names': sequence_names
            }
        }
        
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"Real data dataset saved to {dataset_path}")
        return dataset_path
    
    @staticmethod
    def load_dataset(dataset_path: str) -> Tuple[List[np.ndarray], List[List[Dict]], Dict]:
        """Load processed dataset."""
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        return (dataset_dict['detection_sequences'], 
                dataset_dict['ground_truth_sequences'],
                dataset_dict['metadata'])


def create_train_val_split(dataset_path: str, 
                          train_ratio: float = 0.8,
                          random_state: int = 42) -> Tuple[str, str]:
    """
    Create train/validation split from dataset.
    
    Args:
        dataset_path: Path to dataset file
        train_ratio: Ratio of data for training
        random_state: Random seed
        
    Returns:
        train_dataset_path: Path to training dataset
        val_dataset_path: Path to validation dataset
    """
    # Load dataset
    detection_sequences, ground_truth_sequences, metadata = TrackingDatasetBuilder.load_dataset(dataset_path)
    
    # Split sequences
    n_sequences = len(detection_sequences)
    train_indices, val_indices = train_test_split(
        range(n_sequences), train_size=train_ratio, random_state=random_state)
    
    # Create train dataset
    train_detections = [detection_sequences[i] for i in train_indices]
    train_gt = [ground_truth_sequences[i] for i in train_indices]
    
    # Create validation dataset
    val_detections = [detection_sequences[i] for i in val_indices]
    val_gt = [ground_truth_sequences[i] for i in val_indices]
    
    # Save splits
    base_path = os.path.splitext(dataset_path)[0]
    train_path = f"{base_path}_train.pkl"
    val_path = f"{base_path}_val.pkl"
    
    # Save training set
    train_dict = {
        'detection_sequences': train_detections,
        'ground_truth_sequences': train_gt,
        'metadata': {**metadata, 'split': 'train', 'n_sequences': len(train_detections)}
    }
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_dict, f)
    
    # Save validation set
    val_dict = {
        'detection_sequences': val_detections,
        'ground_truth_sequences': val_gt,
        'metadata': {**metadata, 'split': 'validation', 'n_sequences': len(val_detections)}
    }
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_dict, f)
    
    print(f"Train dataset saved to {train_path} ({len(train_detections)} sequences)")
    print(f"Validation dataset saved to {val_path} ({len(val_detections)} sequences)")
    
    return train_path, val_path


def create_dataloader(dataset_path: str,
                     batch_size: int = 16,
                     sequence_length: int = 10,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     augment: bool = True) -> DataLoader:
    """
    Create PyTorch DataLoader for training.
    
    Args:
        dataset_path: Path to dataset file
        batch_size: Batch size
        sequence_length: Sequence length for training
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    # Load dataset
    detection_sequences, ground_truth_sequences, metadata = TrackingDatasetBuilder.load_dataset(dataset_path)
    
    # Create transform
    transform = DataAugmentation() if augment else None
    
    # Create dataset
    dataset = MaritimeRadarDataset(
        detection_sequences=detection_sequences,
        ground_truth_sequences=ground_truth_sequences,
        transform=transform,
        sequence_length=sequence_length
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn
    )
    
    return dataloader


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching variable-length sequences."""
    # This is a simple implementation - could be enhanced for variable-length sequences
    detections = torch.stack([item['detections'] for item in batch])
    ground_truth = torch.stack([item['ground_truth'] for item in batch])
    sequence_ids = torch.tensor([item['sequence_id'] for item in batch])
    start_frames = torch.tensor([item['start_frame'] for item in batch])
    
    return {
        'detections': detections,
        'ground_truth': ground_truth,
        'sequence_ids': sequence_ids,
        'start_frames': start_frames
    }