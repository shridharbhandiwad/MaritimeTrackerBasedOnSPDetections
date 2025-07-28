#!/usr/bin/env python3
"""
Maritime Radar Data Loader
==========================

Utility functions to load and work with generated maritime radar datasets.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_dataset_index(data_dir: str = 'data') -> Dict[str, Any]:
    """Load the dataset index file."""
    index_path = Path(data_dir) / 'dataset_index.json'
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found at {index_path}")
    
    with open(index_path, 'r') as f:
        return json.load(f)


def list_available_datasets(data_dir: str = 'data') -> List[str]:
    """Return a list of all available dataset names."""
    try:
        index = load_dataset_index(data_dir)
        return list(index.keys())
    except FileNotFoundError:
        return []


def print_available_datasets(data_dir: str = 'data') -> None:
    """Print all available datasets."""
    try:
        index = load_dataset_index(data_dir)
        print("Available Datasets:")
        print("=" * 50)
        
        for dataset_name, info in index.items():
            print(f"  {dataset_name}")
            print(f"    Type: {info['type']}")
            print(f"    Description: {info['description']}")
            print(f"    Path: {info['path']}")
            print()
    except FileNotFoundError:
        print("No datasets found. Run generate_datasets.py first to create datasets.")


def load_dataset(dataset_name: str, data_dir: str = 'data') -> Dict[str, Any]:
    """Load a complete dataset including simulation data and metadata."""
    try:
        # Load simulation data
        frames = load_simulation_data(dataset_name, data_dir)
        
        # Load metadata if available
        try:
            metadata = load_metadata(dataset_name, data_dir)
        except FileNotFoundError:
            metadata = {}
        
        # Load statistics if available
        try:
            statistics = load_statistics(dataset_name, data_dir)
        except FileNotFoundError:
            statistics = {}
        
        return {
            'frames': frames,
            'metadata': metadata,
            'statistics': statistics,
            'dataset_name': dataset_name
        }
    except Exception as e:
        raise Exception(f"Failed to load dataset '{dataset_name}': {str(e)}")


def load_simulation_data(dataset_name: str, data_dir: str = 'data') -> List[Dict[str, Any]]:
    """Load simulation data for a specific dataset."""
    index = load_dataset_index(data_dir)
    
    if dataset_name not in index:
        available = list(index.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    dataset_path = Path(index[dataset_name]['path'])
    simulation_file = dataset_path / 'simulation_data.pkl'
    
    if not simulation_file.exists():
        raise FileNotFoundError(f"Simulation data not found at {simulation_file}")
    
    with open(simulation_file, 'rb') as f:
        return pickle.load(f)


def load_metadata(dataset_name: str, data_dir: str = 'data') -> Dict[str, Any]:
    """Load metadata for a specific dataset."""
    index = load_dataset_index(data_dir)
    
    if dataset_name not in index:
        available = list(index.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    dataset_path = Path(index[dataset_name]['path'])
    metadata_file = dataset_path / 'metadata.json'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_statistics(dataset_name: str, data_dir: str = 'data') -> Dict[str, Any]:
    """Load statistics for a specific dataset."""
    index = load_dataset_index(data_dir)
    
    if dataset_name not in index:
        available = list(index.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    dataset_path = Path(index[dataset_name]['path'])
    stats_file = dataset_path / 'statistics.json'
    
    if not stats_file.exists():
        raise FileNotFoundError(f"Statistics not found at {stats_file}")
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def extract_detections_and_ground_truth(simulation_data: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[List[Dict]]]:
    """
    Extract detections and ground truth from simulation data.
    
    Returns:
        detections: List of detection arrays for each frame [n_detections, 8]
        ground_truth: List of ground truth target states for each frame
    """
    detections = []
    ground_truth = []
    
    for frame_data in simulation_data:
        # Extract detections
        frame_detections = []
        for det in frame_data['all_detections']:
            features = [
                det['range'], det['azimuth'], det['x'], det['y'],
                det['doppler'], det['rcs'], det['snr'],
                1.0 if not det.get('is_clutter', False) else 0.0
            ]
            frame_detections.append(features)
        
        if frame_detections:
            detections.append(np.array(frame_detections))
        else:
            detections.append(np.empty((0, 8)))
        
        # Extract ground truth
        ground_truth.append(frame_data['ground_truth'])
    
    return detections, ground_truth


def print_dataset_summary(dataset_name: str, data_dir: str = 'data') -> None:
    """Print a comprehensive summary of a dataset."""
    try:
        metadata = load_metadata(dataset_name, data_dir)
        stats = load_statistics(dataset_name, data_dir)
        
        print(f"Dataset Summary: {dataset_name}")
        print("=" * 50)
        
        print(f"Scenario: {metadata['scenario_name']}")
        print(f"Duration: {metadata['duration']} seconds")
        print(f"Time step: {metadata['dt']} seconds")
        print(f"Total frames: {metadata['num_frames']}")
        print(f"Generated: {metadata['generated_at']}")
        
        print(f"\nRadar Parameters:")
        radar_params = metadata['radar_params']
        print(f"  Frequency: {radar_params['carrier_freq']/1e9:.1f} GHz")
        print(f"  Max range: {radar_params['max_range']/1000:.1f} km")
        print(f"  Range resolution: {radar_params['range_resolution']} m")
        print(f"  Azimuth beamwidth: {radar_params['azimuth_beamwidth']}°")
        
        print(f"\nEnvironment:")
        env_params = metadata['env_params']
        print(f"  Sea state: {env_params['sea_state']}")
        print(f"  Wind speed: {env_params['wind_speed']} m/s")
        print(f"  Wind direction: {env_params['wind_direction']}°")
        
        print(f"\nTargets: {metadata['num_targets']}")
        
        print(f"\nDetection Statistics:")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Target detections: {stats['target_detections']}")
        print(f"  Clutter detections: {stats['clutter_detections']}")
        print(f"  Clutter ratio: {stats['clutter_ratio']:.1%}")
        print(f"  Avg detections/frame: {stats['avg_detections_per_frame']:.1f}")
        print(f"  Avg targets/frame: {stats['avg_targets_per_frame']:.1f}")
        
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")


def get_training_datasets(data_dir: str = 'data') -> List[str]:
    """Get list of training dataset names."""
    index = load_dataset_index(data_dir)
    return [name for name, info in index.items() if info['type'] == 'training']


def get_validation_datasets(data_dir: str = 'data') -> List[str]:
    """Get list of validation dataset names."""
    index = load_dataset_index(data_dir)
    return [name for name, info in index.items() if info['type'] == 'validation']


def get_test_datasets(data_dir: str = 'data') -> List[str]:
    """Get list of test dataset names."""
    index = load_dataset_index(data_dir)
    return [name for name, info in index.items() if info['type'] == 'test']


def main():
    """Demo of data loading functionality."""
    print("Maritime Radar Data Loader Demo")
    print("=" * 40)
    
    # List available datasets
    print_available_datasets()
    
    # Try to load simple test data if available
    try:
        index = load_dataset_index()
        if 'simple_test' in index:
            print("\nLoading simple_test dataset...")
            print_dataset_summary('simple_test')
            
            # Load and show first frame
            simulation_data = load_simulation_data('simple_test')
            print(f"\nFirst frame example:")
            frame = simulation_data[0]
            print(f"  All detections: {len(frame['all_detections'])}")
            print(f"  Target detections: {len(frame['target_detections'])}")
            print(f"  Clutter detections: {len(frame['clutter_detections'])}")
            print(f"  Ground truth targets: {len(frame['ground_truth'])}")
        else:
            print("\nNo datasets found. Run 'python generate_datasets.py' first.")
            
    except FileNotFoundError:
        print("\nNo datasets found. Run 'python generate_datasets.py' first.")


if __name__ == '__main__':
    main()